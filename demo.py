import torch
import seq_queries
from utilities.load_data import load_data, ConfigDataLoad
from utilities.load_model import get_models
from tools.delta_nets import get_delta_net
from tools.visualize import RunRecorder
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


def main(args):
    json_file = "configs/" + args.data + "/" + args.attack + "/" + "cfg_" + args.scenario + ".json"
    cfg = json.load(open(json_file))
    if args.model is not None:
        cfg["gwad"]["model"]["name"] = args.model
    check_cfg(cfg)

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
        run_name = "benign" if args.scenario == "benign" else args.attack + "_" + args.scenario
        run_name += "_" + cfg["gwad"]["model"]["name"]
        run_config = {"args": vars(args), "cfg": cfg, "device": str(device), "torch": torch.__version__}
        recorder = RunRecorder(args.save_dir, run_name, run_config, cfg["gwad"]["model"],
                               classes, delta_net.classes, args.save_every)

    """ test scenario """
    try:
        if args.scenario == "benign":
            seq_queries.benign(device, cfg, args.data, target_model, delta_net, test_ldr, args.n_images, recorder)
        else:
            seq_queries.attack(device, cfg, args.data, target_model, delta_net, test_ldr, args.n_images, recorder)
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

    arg = parser.parse_args()

    main(parser.parse_args())


