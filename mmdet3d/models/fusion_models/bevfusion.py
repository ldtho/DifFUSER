from typing import Any, Dict
import time

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel
COUNT_FLOPS = False

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
            self.fuser_bn_act = nn.Sequential(
                nn.BatchNorm2d(self.fuser.out_channels),
                nn.ReLU(),
            )
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        self.loss_scale['noise_loss'] = 0.1
        self.loss_scale['object'] = 0.1

        self.times = {
            'extract_camera_features': [],
            'extract_lidar_features': [],
            'fuser': [],
            'decoder': [],
            'head': [],
            # Add other modules if necessary
        }

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        if COUNT_FLOPS:
            print("FLOPS analysis: Camera backbone")
            # print(flop_count_table(FlopCountAnalysis(self.encoders["camera"]["backbone"], x)))
        x = self.encoders["camera"]["backbone"](x)
        if COUNT_FLOPS:
            print("FLOPS analysis: Camera neck")
            # print(flop_count_table(FlopCountAnalysis(self.encoders["camera"]["neck"], x)))

        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        #tic = time.time()
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        if COUNT_FLOPS:
            print("FLOPS analysis: Lidar backbone")
            # print(flop_count_table(FlopCountAnalysis(self.encoders["lidar"]["backbone"], (feats, coords, batch_size, sizes))))
        # self.times['extract_lidar_features'].append(time.time() - tic)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                #tic = time.time()
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                # self.times['extract_camera_features'].append(time.time() - tic)
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
        noise_loss = torch.tensor(0.0, device=features[0].device)
        #tic = time.time()
        intermediate_feats = []
        if self.fuser is not None:
            if "Diffuser" in self.fuser.__class__.__name__:
                if COUNT_FLOPS:
                    print("FLops analysis: Diffuser")
                    # print(flop_count_table(FlopCountAnalysis(self.fuser, features)))
                    x = self.fuser(features)
                    x = self.fuser_bn_act(x)

                else:
                    x, noise_loss, intermediate_feats = self.fuser(features)
                    x = self.fuser_bn_act(x)
            else:
                x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        # self.times['fuser'].append(time.time() - tic)


        batch_size = x.shape[0]

        #tic = time.time()
        if COUNT_FLOPS:
            print("FLOPS analysis: decoder backbone")
            # print(flop_count_table(FlopCountAnalysis(self.decoder["backbone"], x.clone())))
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        # self.times['decoder'].append(time.time() - tic)
        if len(intermediate_feats) > 0:
            for i, feat in enumerate(intermediate_feats):
                feat = self.decoder["backbone"](feat)
                feat = self.decoder["neck"](feat)
                intermediate_feats[i] = feat

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            losses["noise_loss"] = noise_loss
            outputs[f"loss/noise_loss"] = losses['noise_loss'] * self.loss_scale["noise_loss"]
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    #tic = time.time()
                    pred_dict = head(x, metas)
                    # self.times['head'].append(time.time() - tic)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    bboxes_intermediates = []

                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                                "boxes_3d_intermediate": [],
                                "scores_3d_intermediate": [],
                                "labels_3d_intermediate": [],
                            }
                        )
                    if len(intermediate_feats)>0:
                        for feat in intermediate_feats:
                            pred_dict_intermediate = head(feat, metas)
                            bboxes_intermediate = head.get_bboxes(pred_dict_intermediate, metas)
                            bboxes_intermediates.append(bboxes_intermediate)

                        for bboxes_intermediate in bboxes_intermediates:
                            for k, (boxes, scores, labels,) in enumerate(bboxes_intermediate):
                                outputs[k]["boxes_3d_intermediate"].append(boxes.to("cpu"))
                                outputs[k]["scores_3d_intermediate"].append(scores.cpu())
                                outputs[k]["labels_3d_intermediate"].append(labels.cpu()
                                )

                elif type == "map":
                    #tic = time.time()
                    logits = head(x)
                    if COUNT_FLOPS:
                        print("FLOPS analysis: map head")
                        # print(flop_count_table(FlopCountAnalysis(head, x)))
                    #self.times['head'].append(time.time() - tic)
                    logits_intermediate = []
                    if len(intermediate_feats) > 0:
                        for feat in intermediate_feats:
                            logits_intermediate.append(head(feat))
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                                "masks_bev_intermediate": [logit[k].cpu() for logit in logits_intermediate] if len(logits_intermediate) > 0 else None,
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")

            # for k, v in self.times.items():
            #     print(f"{k}: {torch.mean(torch.tensor(v))}")
            return outputs
