import argparse
import copy
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from mmcv import Config
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
import torch.multiprocessing as mp
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    print(f"Setting up distributed environment: rank {rank}, world size {world_size}")
    os.environ['MASTER_ADDR'] = 'localhost' # actual ip-address of the master
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    print("Cleaning up distributed environment.")
    torch.distributed.destroy_process_group()

def main():
    print("Starting main process.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    # use wandb or not, default is True
    parser.add_argument("--nowandb", action="store_true", help="use wandb or not")
    # resume_id
    parser.add_argument("--resume", type=str, help="resume wandb logging")
    args, opts = parser.parse_known_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size =torch.cuda.device_count()
    setup(args.local_rank, world_size)
    # mp.spawn(setup, args=(args.local_rank, world_size,), nprocs=4)
    # setup_process, args=(world_size, "29500"), nprocs=4)


    print("Loading configuration.")
    configs.load(args.config, recursive=True)
    configs.update(opts)
    print("####configs####")

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(args.local_rank)
    print('####local_rank####')

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    print("Building dataset.")
    datasets = [build_dataset(cfg.data.train)]

    print("Building model.")
    print(f"Debug: local_rank = {args.local_rank}, nowandb = {args.nowandb}")

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(args.local_rank)
    # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print(f"Debug: local_rank = {args.local_rank}, nowandb = {args.nowandb}")
    if args.local_rank == 0 and not args.nowandb:
        import wandb
        # get the name
        group_name = cfg.run_dir.split("/")[-1]
        # run_name = group_name + datetime.now().strftime("-%d/%m_%H:%M")
        save_dir = f'../runs/{group_name}'
        os.makedirs(save_dir, exist_ok=True)

        wandb.init(
            project="diffuser",
            entity="ldtho-97",
            config=cfg._cfg_dict,
            sync_tensorboard=True,
            dir = save_dir,
            name=group_name,
            resume="must" if args.resume else None,
            id=args.resume if args.resume else None,
        )
        # log the transfusion.py file
        # wandb.save("../mmdet3d/models/fusers/diffuser.py")


        # wandb.watch(model, optimizer,
        #             log="all", log_freq=100,log_graph=True)



    logger.info(f"Model:\n{model}")
    print("Starting training process.")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp
    )

    cleanup()


if __name__ == "__main__":
    main()
# RDMAV_FORK_SAFE=1 python -m torch.distributed.launch  --nproc_per_node=4  --use_env  tools/train.py  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/diffuser-cam640.yaml  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth  --load_from pretrained/lidar-only-det.pth  --run-dir runs/test
