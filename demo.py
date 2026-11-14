import os
import sys
import time
import subprocess
import faulthandler
import torch
import seq_queries
from utilities.load_data import load_data, ConfigDataLoad
from utilities.load_model import get_models
from tools.delta_nets import get_delta_net
from tools.visualize import RunRecorder
from tools.statistics import GWADStatistics
import json
from argparse import ArgumentParser

"""
This Demo version of GWAD only supports CIFAR10, but not restricted. models and weights are downloaded from 
https://github.com/huyvnphan/PyTorch_CIFAR10
Modify load_data and load_model in utilities to add other models and dataset     
"""
data_type = 'cifar10'

""" check cuda is available """
print(torch.__version__)
print(f"Is CUDA supported by this system? {torch.cuda.is_available()} ")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_cfg(cfg):
    cfg_attack, cfg_gwad = cfg["attack"], cfg["gwad"]
    if cfg_gwad["screen"]["on"] and cfg_attack["adaptive"]["name"] == 'batch':
        batch_size = cfg_attack["adaptive"]["batch_size"]
        fifo_depth = cfg_gwad["screen"]["n"]
        assert batch_size < fifo_depth, "screener fifo depth must be greater than batch size"
        pool_size = cfg_attack["adaptive"]["pool_size"]
        assert fifo_depth*2 <= pool_size, "benign pool size must be greater than fifo depth x 2 to avoid duplications"


def get_run_name(args, cfg):
    run_name = "benign" if args.scenario == "benign" else args.attack + "_" + args.scenario
    return run_name + "_" + cfg["gwad"]["model"]["name"]


def run_workers(args, cfg):
    """
    --workers N : attack the images in N parallel processes of this script (--shard k/N) and merge their results.
    each image is independent (GWAD is reset per image) and seeded with seed + test index, so every image gives
    the same result as in a single process run with the same --seed. attack i (0-based, benign pool excluded)
    is done by shard i % N. a batch=1 query leaves the GPU mostly idle, so N processes share one GPU well.

    watchdog: a worker process occasionally hangs (one core spinning, no progress) when several processes share
    the GPU. every worker touches a heartbeat file during its queries; a worker without a heartbeat for
    --worker_timeout seconds is killed and its shard restarted as a new part that skips the images the killed
    part already committed. images are seeded, so a redone image gives the same result.
    """
    n = args.workers
    if args.scenario == "benign":
        raise ValueError("--workers is not supported for the benign scenario (one continuous query sequence)")
    if args.seed is None:
        args.seed = 0
        print("--workers : per-image seeding enabled with --seed 0")

    run_name = get_run_name(args, cfg)
    if args.save_dir is not None:
        run_dir = RunRecorder.new_dir(args.save_dir, run_name)
        RunRecorder.write_config(run_dir, {"args": vars(args), "cfg": cfg, "device": str(device),
                                           "torch": torch.__version__})
    else:
        run_dir = os.path.join("stats", "shards", "%s_%s" % (time.strftime('%Y%m%d-%H%M%S'), run_name))
    os.makedirs(run_dir, exist_ok=True)

    # every process starting a full CPU thread pool would oversubscribe the cores
    # (16-core laptop, 4 workers: 1-2 threads per worker were fastest, torch default threads the slowest)
    env = dict(os.environ)
    threads = max(1, args.worker_threads)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"  # live part logs

    base = [sys.executable, os.path.abspath(__file__), "--data", args.data, "--attack", args.attack,
            "--scenario", args.scenario, "--save_every", str(args.save_every), "--seed", str(args.seed),
            "--run_dir", run_dir, "--stall_dump", str(max(10, args.worker_timeout // 2))]
    if args.n_images is not None:
        base += ["--n_images", str(args.n_images)]
    if args.model is not None:
        base += ["--model", args.model]
    if args.early_stop:
        base += ["--early_stop"]
    if args.save_dir is not None:
        base += ["--save_dir", args.save_dir]

    def launch(k, attempt):
        part = "%d.%d" % (k, attempt)
        skip_file = os.path.join(run_dir, "skip.part%s.txt" % part)
        with open(skip_file, "w") as f:
            f.write("\n".join(str(i) for i in sorted(done_images(run_dir, k))))
        hb = os.path.join(run_dir, "heartbeat.part%s" % part)
        open(hb, "w").close()  # start-up (CUDA init, data loading) counts against the timeout too
        log = open(os.path.join(run_dir, "part%s.log" % part), "w")
        cmd = base + ["--shard", "%d/%d" % (k, n), "--part", part, "--skip_file", skip_file, "--heartbeat_file", hb]
        return {"k": k, "attempt": attempt, "part": part, "hb": hb, "log": log,
                "proc": subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)}

    print("--workers : %d processes, %d CPU threads each, watchdog %ds, logs in %s"
          % (n, threads, args.worker_timeout, run_dir))
    t0 = time.perf_counter()
    running = [launch(k, 0) for k in range(n)]
    failed = []
    try:
        while running:
            time.sleep(2)
            for w in list(running):
                rc = w["proc"].poll()
                if rc is None:
                    try:
                        idle = time.time() - os.path.getmtime(w["hb"])
                    except OSError:
                        idle = 0
                    if idle < args.worker_timeout:
                        continue
                    print("--workers : part %s made no progress for %.0fs, killing it (stack dump in part%s.log)"
                          % (w["part"], idle, w["part"]))
                    w["proc"].kill()
                    w["proc"].wait()
                    rc = "stalled"
                w["log"].close()
                running.remove(w)
                if rc == 0:
                    continue
                if w["attempt"] < args.worker_retries:
                    print("--workers : part %s ended (%s), restarting shard %d without its %d finished images"
                          % (w["part"], rc, w["k"], len(done_images(run_dir, w["k"]))))
                    running.append(launch(w["k"], w["attempt"] + 1))
                else:
                    print("--workers : shard %d failed %d times, giving up" % (w["k"], w["attempt"] + 1))
                    failed.append(w["k"])
    except KeyboardInterrupt:
        # stop the workers; their committed images are merged below
        for w in running:
            w["proc"].kill()
            w["proc"].wait()
            w["log"].close()
        failed.append("interrupted")
    print("--workers : finished in %.1fs" % (time.perf_counter() - t0))
    merge_parts(run_dir, run_name, args.save_dir is not None)
    if failed:
        sys.exit("some shards failed (%s), see %s/part<k>.<attempt>.log" % (failed, run_dir))


def part_states(run_dir):
    # {part id: state} of every part that committed at least one image
    states = {}
    for name in os.listdir(run_dir):
        if name.startswith("state.part") and name.endswith(".pt"):
            states[name[len("state.part"):-len(".pt")]] = torch.load(os.path.join(run_dir, name))
    return states


def done_images(run_dir, k):
    # test indices committed by any part of shard k
    done = set()
    for part, st in part_states(run_dir).items():
        if part.split(".")[0] == str(k):
            done.update(st["done"])
    return done


def merge_parts(run_dir, run_name, recorded):
    # stats/dist/*_<i>.txt are written by the parts with the global attack index i;
    # stats/hist (mean histogram) = sum of the part histograms / total number of attacked images
    states = part_states(run_dir)
    total = GWADStatistics()
    hist_file = None
    seen = set()
    for part, st in states.items():
        dup = seen.intersection(st["done"])
        assert not dup, "images %s were committed by more than one part" % sorted(dup)
        seen.update(st["done"])
        total.hist += st["hist"]
        total.runs += st["runs"]
        hist_file = st["hist_file"]
    if hist_file is not None and total.runs > 0:
        seq_queries.save_hist(hist_file, total)
        print("merge : %d images, mean histogram saved in %s" % (total.runs, hist_file))
    if recorded:
        # parts that never committed an image may still have left (empty / partial) recorder files
        parts = set()
        for name in os.listdir(run_dir):
            for prefix, suffix in (("recorder.part", ".pkl"), ("gwad_timeline.part", ".csv")):
                if name.startswith(prefix) and name.endswith(suffix):
                    parts.add(name[len(prefix):-len(suffix)])
        RunRecorder.merge(run_dir, run_name, {p: set(states[p]["done"]) if p in states else set() for p in parts})
    for name in os.listdir(run_dir):
        if name.startswith(("state.part", "skip.part", "heartbeat.part")):
            os.remove(os.path.join(run_dir, name))


class Heartbeat:
    """
    called on every query of a worker: at most every `interval` seconds it touches the heartbeat file watched
    by the parent and re-arms faulthandler, which dumps the stacks of all threads into the part log if the
    process makes no progress for `dump_after` seconds (shortly before the parent kills it).
    """
    def __init__(self, path, dump_after, interval=5.0):
        self.path, self.dump_after, self.interval = path, dump_after, interval
        self.next = 0.0

    def __call__(self):
        now = time.monotonic()
        if now < self.next:
            return
        self.next = now + self.interval
        try:
            os.utime(self.path)
        except OSError:
            open(self.path, "w").close()
        faulthandler.dump_traceback_later(self.dump_after, repeat=False, file=sys.stderr)


def main(args):
    json_file = "configs/" + args.data + "/" + args.attack + "/" + "cfg_" + args.scenario + ".json"
    cfg = json.load(open(json_file))
    if args.model is not None:
        cfg["gwad"]["model"]["name"] = args.model
    check_cfg(cfg)

    if args.workers > 1:
        return run_workers(args, cfg)
    if args.seed is not None:
        # cudnn may pick another conv algorithm (tiny float differences) depending on GPU memory / other processes,
        # which changes hard labels of queries close to the decision boundary (e.g. already misclassified images)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    shard, part, part_state, skip, heartbeat = None, None, None, set(), None
    if args.shard is not None:
        shard = tuple(int(v) for v in args.shard.split("/"))
        assert 0 <= shard[0] < shard[1], "--shard k/N needs 0 <= k < N"
        assert args.seed is not None, "--shard needs --seed so that results do not depend on the shard"
        assert args.run_dir is not None, "--shard needs --run_dir"
        part = args.part if args.part is not None else "%d.0" % shard[0]
        part_state = os.path.join(args.run_dir, "state.part%s.pt" % part)
        if args.skip_file is not None:
            skip = {int(v) for v in open(args.skip_file).read().split()}
        if args.heartbeat_file is not None:
            heartbeat = Heartbeat(args.heartbeat_file, args.stall_dump)

    """ load threat model """
    target_model = get_models(data_type, cfg["gwad"]["model"]["name"])
    target_model.eval()
    target_model.to(device)

    """ load datasat """
    data_load_cfg = ConfigDataLoad(args.data)
    data_load_cfg.train.shuffle = True
    data_load_cfg.train.batch_size = 1
    data_load_cfg.test.shuffle = False
    data_load_cfg.test.batch_size = 1
    train_ldr, test_ldr, classes, _ = load_data(data_load_cfg)

    """ load gwad classifier """
    delta_net = get_delta_net(device)

    """ result recorder """
    recorder = None
    if args.save_dir is not None:
        run_name = get_run_name(args, cfg)
        run_config = {"args": vars(args), "cfg": cfg, "device": str(device), "torch": torch.__version__}
        recorder = RunRecorder(args.save_dir, run_name, run_config, cfg["gwad"]["model"],
                               classes, delta_net.classes, args.save_every,
                               run_dir=args.run_dir, part=part)

    """ test scenario """
    # every query is a pure forward pass (black-box attacks never backprop), so skip autograd bookkeeping
    with torch.inference_mode():
        try:
            if args.scenario == "benign":
                seq_queries.benign(device, cfg, args.data, target_model, delta_net, test_ldr, args.n_images, recorder)
            else:
                seq_queries.attack(device, cfg, args.data, target_model, delta_net, test_ldr, args.n_images, recorder,
                                   seed=args.seed, shard=shard, part_state=part_state,
                                   skip=skip, heartbeat=heartbeat, stop=args.early_stop)
        finally:
            # keep results of finished images even if the run is interrupted
            if recorder is not None:
                recorder.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--attack', type=str)
    parser.add_argument('--scenario', type=str)
    # benign: number of benign queries, attack: number of attacked images (benign pool excluded)
    parser.add_argument('--n_images', type=int, default=None)
    # target model, overrides gwad.model.name in json config
    parser.add_argument('--model', type=str, default=None, choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet'])
    # save summary images and GWAD timeline in <save_dir>/<timestamp>_<run name>/ (default dir: results)
    parser.add_argument('--save_dir', type=str, nargs='?', const='results', default=None)
    # save every N-th attack query image in queries/ (0: off), only with --save_dir
    parser.add_argument('--save_every', type=int, default=500)
    # seed every attacked image with seed + test index (default: unseeded, one RNG stream across images)
    parser.add_argument('--seed', type=int, default=None)
    # stop attacking an image once the attack's own stop criterion is met (e.g. RayS: adversarial within epsilon)
    # instead of spending the whole query budget
    parser.add_argument('--early_stop', action='store_true')
    # attack the images in N parallel processes sharing the GPU and merge the results (implies --seed 0)
    parser.add_argument('--workers', type=int, default=1)
    # CPU threads per worker process (OMP/MKL_NUM_THREADS)
    parser.add_argument('--worker_threads', type=int, default=2)
    # restart a worker that made no progress for this many seconds (it is killed, its finished images are kept)
    parser.add_argument('--worker_timeout', type=int, default=60)
    # restarts per shard before giving up
    parser.add_argument('--worker_retries', type=int, default=3)
    # internal, set by --workers: attack only share k of N as part <k>.<attempt> and write results in run_dir
    parser.add_argument('--shard', type=str, default=None)
    parser.add_argument('--part', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--skip_file', type=str, default=None)
    parser.add_argument('--heartbeat_file', type=str, default=None)
    parser.add_argument('--stall_dump', type=int, default=90)

    arg = parser.parse_args()

    main(parser.parse_args())


