import argparse
import copy
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    # use wandb or not, default is True
    parser.add_argument("--nowandb", action="store_true", help="use wandb or not")
    # resume_id
    parser.add_argument("--resume", type=str, help="resume wandb logging")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

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

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    if dist.is_master() and not args.nowandb:
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
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
        dist = dist,
        args = args
    )


if __name__ == "__main__":
    main()
