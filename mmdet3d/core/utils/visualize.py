import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


# OBJECT_PALETTE = {
#     "car": (255, 158, 0), # Orange
#     "truck": (255, 99, 71), # Tomato
#     "construction_vehicle": (233, 150, 70), # Dark Orange
#     "bus": (255, 69, 0), # Red-Orange
#     "trailer": (255, 140, 0), # Dark Orange
#     "barrier": (112, 128, 144), # Slate Gray
#     "motorcycle": (255, 61, 99), # Bright Pink
#     "bicycle": (220, 20, 60), # Crimson (Dark Red)
#     "pedestrian": (0, 0, 230), # Blue
#     "traffic_cone": (47, 79, 79), # Dark Slate Gray
# }
OBJECT_PALETTE = {
    "car": (255, 158, 0), # Orange
    "truck": (128, 0, 0),  # Maroon
    "construction_vehicle": (0, 128, 0),  # Green
    "bus": (0, 0, 128),  # Navy
    "trailer": (128, 0, 128),  # Purple
    "barrier": (128, 128, 0),  # Olive
    "motorcycle": (0, 0, 255),  # Blue
    "bicycle": (255, 0, 255),  # Magenta
    "pedestrian": (255, 0, 0),  # Cyan
    "traffic_cone": (0, 255, 255),  # Red
}
MAP_PALETTE = {
    "drivable_area": (166, 206, 227), # Light Sky Blue
    "road_segment": (31, 120, 180), # Medium Blue
    "road_block": (178, 223, 138), # Light Green
    "lane": (51, 160, 44), # Medium Green
    "ped_crossing": (251, 154, 153), # Light Pink
    "walkway": (227, 26, 28), # Firebrick (Dark Red)
    "stop_line": (253, 191, 111), # Peach-Orange
    "carpark_area": (255, 127, 0), # Dark Orange
    "road_divider": (202, 178, 214), # Light Lavender
    "lane_divider": (106, 61, 154), # Dark Purple
    "divider": (106, 61, 154), # Dark Purple
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    # save input image
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    # mmcv.imwrite(canvas, fpath)
    # return
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int32),
                    coords[index, end].astype(np.int32),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


# def visualize_lidar(
#     fpath: str,
#     lidar: Optional[np.ndarray] = None,
#     *,
#     bboxes: Optional[LiDARInstance3DBoxes] = None,
#     labels: Optional[np.ndarray] = None,
#     classes: Optional[List[str]] = None,
#     xlim: Tuple[float, float] = (-50, 50),
#     ylim: Tuple[float, float] = (-50, 50),
#     color: Optional[Tuple[int, int, int]] = None,
#     radius: float = 15,
#     thickness: float = 25,
# ) -> None:
#     fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
#
#     ax = plt.gca()
#     ax.set_xlim(*xlim)
#     ax.set_ylim(*ylim)
#     ax.set_aspect(1)
#     ax.set_axis_off()
#
#     if lidar is not None:
#         plt.scatter(
#             lidar[:, 0],
#             lidar[:, 1],
#             s=radius,
#             c="white",
#         )
#
#     if bboxes is not None and len(bboxes) > 0:
#         coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
#         for index in range(coords.shape[0]):
#             name = classes[labels[index]]
#             plt.plot(
#                 coords[index, :, 0],
#                 coords[index, :, 1],
#                 linewidth=thickness,
#                 color=np.array(color or OBJECT_PALETTE[name]) / 255,
#             )
#
#     mmcv.mkdir_or_exist(os.path.dirname(fpath))
#     fig.savefig(
#         fpath,
#         dpi=10,
#         facecolor="black",
#         format="png",
#         bbox_inches="tight",
#         pad_inches=0,
#     )
#     plt.close()
def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="black",  # Changed color to black
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor= (240/255, 240/255, 240/255), # Changed facecolor to white (240, 240, 240)
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    # rotate the image 90 degrees to the right to match the lidar view
    canvas = np.rot90(canvas, 1)
    mmcv.imwrite(canvas, fpath)
