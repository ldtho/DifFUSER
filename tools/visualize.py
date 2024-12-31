import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.01)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--viz_intermediate", action="store_true")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)
    if cfg.model.fuser.type == "Diffuser":
        cfg.model.fuser.return_intermediate = args.viz_intermediate

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()
    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            if "boxes_3d_intermediate" in outputs[0]:
                bboxes_intermediate = [b.tensor.numpy() for b in outputs[0]["boxes_3d_intermediate"]]
                scores_intermediate = [s.numpy() for s in outputs[0]["scores_3d_intermediate"]]
                labels_intermediate = [l.numpy() for l in outputs[0]["labels_3d_intermediate"]]

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
                if "boxes_3d_intermediate" in outputs[0]:
                    indices_intermediate = [np.isin(l, args.bbox_classes) for l in labels_intermediate]
                    bboxes_intermediate = [b[i] for b, i in zip(bboxes_intermediate, indices_intermediate)]
                    scores_intermediate = [s[i] for s, i in zip(scores_intermediate, indices_intermediate)]
                    labels_intermediate = [l[i] for l, i in zip(labels_intermediate, indices_intermediate)]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
                if "boxes_3d_intermediate" in outputs[0]:
                    indices_intermediate = [s >= args.bbox_score for s in scores_intermediate]
                    bboxes_intermediate = [b[i] for b, i in zip(bboxes_intermediate, indices_intermediate)]
                    scores_intermediate = [s[i] for s, i in zip(scores_intermediate, indices_intermediate)]
                    labels_intermediate = [l[i] for l, i in zip(labels_intermediate, indices_intermediate)]

            if "boxes_3d_intermediate" in outputs[0]:
                bboxes_intermediate = [LiDARInstance3DBoxes(b, box_dim=9) for b in bboxes_intermediate]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(bool)
            masks_intermediate = None
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks_intermediate = None
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
            if outputs[0]['masks_bev_intermediate'] is not None:
                masks_intermediate = [m.numpy() > args.map_score for m in outputs[0]['masks_bev_intermediate']]
        else:
            masks = None

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_{'last' if args.mode == 'pred' else ''}.png"),
                masks,
                classes=cfg.map_classes,
            )
            if masks_intermediate is not None:
                for i, mask in enumerate(masks_intermediate):
                    visualize_map(
                        os.path.join(args.out_dir, "map", f"{name}_{i+1}.png"),
                        mask,
                        classes=cfg.map_classes,
                    )

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}_{'last' if args.mode == 'pred' else ''}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )
                if args.mode == 'pred':
                    if "boxes_3d_intermediate" in outputs[0]:
                        for i, (bboxes_i,labels_i) in enumerate(zip(bboxes_intermediate,labels_intermediate)):
                            visualize_camera(
                                os.path.join(args.out_dir, f"camera-{k}", f"{name}_{i+1}.png"),
                                image,
                                bboxes=bboxes_i,
                                labels=labels_i,
                                transform=metas["lidar2image"][k],
                                classes=cfg.object_classes,
                            )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}_{'last' if args.mode == 'pred' else ''}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )
            if args.mode == 'pred':
                if "boxes_3d_intermediate" in outputs[0]:
                    for i, (bboxes_i,labels_i) in enumerate(zip(bboxes_intermediate,labels_intermediate)):
                        visualize_lidar(
                            os.path.join(args.out_dir, "lidar", f"{name}_{i+1}.png"),
                            lidar,
                            bboxes=bboxes_i,
                            labels=labels_i,
                            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                            classes=cfg.object_classes,
                        )



if __name__ == "__main__":
    main()
