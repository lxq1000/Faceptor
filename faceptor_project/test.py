
import os
import torch
import argparse
from torch import distributed
from core.utils import printlog
from core.config import Config
from core.solver import solver_entry

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    printlog("args:", args)

    C = Config(args, rank, local_rank, world_size)
    S = solver_entry(C)
    S.test()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Test Framework for Faceptor')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--expname', type=str, default=None, help='experiment name, output folder')
    parser.add_argument("--start_time", type=str, default="20230000_000000")
    parser.add_argument("--now", type=str, default="20230000_000000")
    parser.add_argument("--test_iter", type=str, default="newest")
    parser.add_argument("--out_dir", default='', type=str)
    parser.add_argument("--port", type=str, default="5671")
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--load_iter', default='newest', type=str)
    parser.add_argument('--load_ignore', nargs='+', default=[], type=str)

    

    main(parser.parse_args())


