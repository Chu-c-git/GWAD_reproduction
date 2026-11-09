import os
import torch
import time
import random
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools.statistics import GWADStatistics, AttackStatistics
from tools.attack_model import AttackModel as ATTACK_MODEL
from gwad import GWAD
from algorithm.attack.base import Query


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    img = img.cpu()
    return img.detach().numpy()


class Statistics:
    def __init__(self):
        self.gwad = GWADStatistics()
        self.attack = AttackStatistics()
        return

    def increment_runs(self):
        self.gwad.increment_runs()
        self.attack.increment_runs()
        return


def show_predicrions(gwad_def, hx_record, total_queries):
    # show gwad prediction record
    hx = gwad_def.get_predictions()
    hx = hx.numpy()
    queries, delta_queries = gwad_def.get_queries()
    total_queries += delta_queries
    print("GWAD_defence : delta-net predicted {} HoDS during {} queries".format(int(np.sum(hx)), queries))
    print("GWAD_defence : accumulated predictions {}/{}".format(int(np.sum(hx_record) + np.sum(hx)), total_queries))
    for i in range(len(hx_record)):
        hx_record[i] += hx[i]
        print("%s[%d], " % (gwad_def.classes[i], hx_record[i]), end='')
    print("")

    return total_queries


def hist_path(stats_dir, model_name, feat_type, data_type):
    return '%s/hist/%s_%s_%s.txt' % (stats_dir, data_type, model_name, feat_type)


def save_hist(hist_file, gwad_stats):
    np.savetxt(hist_file, gwad_stats.mean_hist(), delimiter=',')


def save_distributions(stats_dir, model, stats, feat_type, data_type, run_idx=None, sharded=False):
    # run_idx: global (1-based) index of the attacked image, defaults to the local run count.
    # sharded: (--workers/--shard) the mean histogram is written by the parent after all parts finish
    # from the per-part histograms saved by commit_part, so it is not written here
    if not sharded:
        save_hist(hist_path(stats_dir, model.name, feat_type, data_type), stats.gwad)

    dist = stats.gwad.distribution
    run_idx = stats.gwad.runs if run_idx is None else run_idx
    dist_file = '%s/dist/%s_%s_%s' % (stats_dir, data_type, feat_type, run_idx)
    np.savetxt('%s.txt' % (dist_file), dist, delimiter=',')


def commit_part(part_state, stats, done, hist_file):
    # commit point of a sharded run: written atomically after every finished image (after its dist file and
    # recorder record). the parent merges only images listed in 'done', and a retry of a killed part skips them
    tmp = part_state + '.tmp'
    torch.save({'hist': stats.gwad.hist.clone(), 'runs': stats.gwad.runs, 'done': list(done),
                'hist_file': hist_file}, tmp)
    os.replace(tmp, part_state)


def seed_image(seed, test_idx):
    # per-image seed so each image sees the same random stream regardless of
    # which images ran before it (needed for sharded runs to match a single process run)
    s = seed + test_idx
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def benign(device, cfg, data_type, model, delta_net, loader, n_images=None, recorder=None):
    stats_dir = 'stats'
    cfg_gwad = cfg["gwad"]
    attack_name = 'benign'

    # ready statistics
    stats = Statistics()

    # instantiate gwad
    gwad_def = GWAD(device, cfg_gwad, stats.gwad, mode='defence', model=model, delta_net=delta_net)

    # reset gwad
    gwad_def.reset()

    # record whole benign sequence as one image
    gwad_def.recorder = recorder
    if recorder is not None:
        recorder.start_image(0)

    # main loop
    hx_record = np.zeros(len(delta_net.classes))
    total_queries = 0
    idx = 0
    for data, true_class in loader:
        if n_images is not None and idx >= n_images:
            break
        if idx%10 == 0:
            print(end='\r')
            print("{}/{}".format(idx, len(loader)), end='')
        data = data.to(device)
        q = Query(t='benign', x=data)
        gwad_def.run(q)
        idx += 1

    if recorder is not None:
        recorder.end_sequence(gwad_def.first_detect)

    stats.increment_runs()

    # show gwad prediction record
    show_predicrions(gwad_def, hx_record, total_queries)

    # show defence statistics
    stats.gwad.show_screen()

    # get and save mean histogram in text file
    save_distributions(stats_dir, model, stats, attack_name, data_type)


def attack(device, cfg, data_type, model, delta_net, loader, n_images=None, recorder=None,
           seed=None, shard=None, part_state=None, skip=(), heartbeat=None):
    # seed  : None keeps the global RNG stream running across images (original behaviour),
    #         otherwise every image is attacked after seed_image(seed, test_idx)
    # shard : (k, n) attack only images whose attack index i (0-based, benign pool excluded) has i % n == k.
    #         every shard walks the same loader, so the benign pool of the 'batch' scenario is identical
    # part_state : state file of this worker process, see commit_part (merged by demo.merge_parts)
    # skip  : test indices already finished by an earlier (killed) part of this shard
    # heartbeat : callable invoked on every query and loader step, lets the parent detect a stalled worker
    stats_dir = 'stats'
    cfg_attack, cfg_gwad = cfg["attack"], cfg["gwad"]
    attack_name = cfg_attack["name"]
    adapt_type = cfg_attack["adaptive"]["name"]
    batch_size = cfg_attack["adaptive"]["batch_size"]
    pool_size = cfg_attack["adaptive"]["pool_size"]
    move_rate = cfg_attack["adaptive"]["move_rate"]
    q_budgets = [cfg_attack["query_budget"]]

    # ready statistics
    stats = Statistics()

    # instantiate gwad models
    # - defence model  : to generate adversarial examples
    # - evaluate model : to inject generated adversarial example
    gwad_def = GWAD(device, cfg_gwad, stats.gwad, mode='defence', model=model, delta_net=delta_net)
    gwad_evl = GWAD(device, cfg_gwad, stats.gwad, mode='evaluate', model=model, delta_net=delta_net)
    gwad_def.recorder = recorder

    # show setting
    print("\nsequence of queries : attack \n{} {} {} {}".format(data_type, model.name, attack_name, delta_net.name))

    # main loop
    hx_record = np.zeros(len(delta_net.classes))
    total_gwad_predict = 0
    total_attack_queries = 0
    x2_pool = []
    cnt = 0
    done = []
    hist_file = hist_path(stats_dir, model.name, attack_name, data_type)
    d_model = gwad_def.run
    if heartbeat is not None:
        def d_model(q):
            heartbeat()
            return gwad_def.run(q)
    for test_idx, (data, true_class) in enumerate(loader):
        if heartbeat is not None:
            heartbeat()
        data, true_class = data.to(device), true_class.to(device)

        # make benign example pool for multi-agent adaptive attack
        if adapt_type == 'batch':
            if len(x2_pool) < pool_size:
                x2_pool.append(data)
                continue

        if n_images is not None and cnt >= n_images:
            break

        # skip images of other shards
        if (shard is not None and cnt % shard[1] != shard[0]) or test_idx in skip:
            cnt += 1
            continue

        if seed is not None:
            seed_image(seed, test_idx)

        # reset gwad
        gwad_def.reset()
        if recorder is not None:
            recorder.start_image(test_idx)

        # instantiate attack model to conduct fresh attack
        attack = ATTACK_MODEL(data_type, device, stats.attack, cfg_attack,
                              d_model=d_model, q_budgets=q_budgets, stop=False)

        # set adaptive attack
        attack.set_adaptive(adapt_type, move_rate, batch_size, x2_pool)

        # perform adversarial attack on current data
        t0 = time.perf_counter()
        adv = attack.run(data, true_class)
        t1 = time.perf_counter()

        # predict adversarial example
        q = Query(t='attack', x=adv[0])
        predict = gwad_evl.run(q)
        if shard is not None and gwad_evl.examples_ready():
            # gwad_evl is never reset, so it keeps the adversarial examples of previous images; once 3 of them
            # pass its screener it adds a ds to this image's stats, which differs between shards
            print("warning: evaluate GWAD added a ds of earlier images to image %d (not shard-invariant)" % test_idx)

        # update attack statistics
        attack.update_stats(data, adv[0], true_class, predict)

        # save summary image of this attack
        if recorder is not None:
            recorder.end_image(data, adv[0], true_class.item(), int(stats.attack.hx),
                               float(stats.attack.dist0), float(stats.attack.dist1),
                               gwad_def.get_predictions(), gwad_def.first_detect, gwad_def.query_cnt)

        # increment count
        stats.increment_runs()

        # show gwad prediction record
        total_gwad_predict = show_predicrions(gwad_def, hx_record, total_gwad_predict)

        # show attack statistics
        print("Attack stats : num - time [true] : [adv][val/suc, dist, ratio, [i1] [i2] [i3]]")
        print("{} - {:.3f}s [{}] : ".format(stats.gwad.runs, t1-t0, true_class.item()), end='')
        stats.attack.show_stats()

        cnt += 1
        total_attack_queries += stats.attack.iter0[0]
        print("average attack queries - {}".format(int(total_attack_queries/stats.attack.runs)))

        # show defence statistics
        stats.gwad.show_screen()

        # get and save mean histogram in text file
        save_distributions(stats_dir, model, stats, attack_name, data_type, run_idx=cnt,
                           sharded=part_state is not None)
        if part_state is not None:
            if recorder is not None:
                recorder.commit_image()
            done.append(test_idx)
            commit_part(part_state, stats, done, hist_file)