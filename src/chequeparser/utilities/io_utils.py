import json
from pathlib import Path
from typing import List, Union

import h5py
from PIL import Image
from loguru import logger
from tqdm.autonotebook import tqdm

from chequeparser.utilities.misc import filter_list
from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.detection_results import DetectionResults


def convert_tif_to_jpg(path_tif: Path, path_jpeg: Path, ext=".jpg"):
    path_jpeg.mkdir(parents=True, exist_ok=True)
    for item in tqdm(list(path_tif.glob("*.tif"))):
        img = Image.open(item).convert("RGB")
        img.save(path_jpeg / Path(item.name).with_suffix(ext))


def get_files(
    source: Union[str, Path],
    exts: list = [".png", ".jpeg", ".jpg", ".tif"],
    ignore_hidden_dirs=True,
    ignore_hidden_files=True,
) -> list:
    """Gets all types of files from the directory.
    Filter for ignoring hidden directories by default
    Filter for ignoring hidden files by default
    """
    p_source = Path(source).resolve()
    l_files = []
    l_all_files = list(p_source.iterdir())
    s_suffixes = set(exts)
    for file in tqdm(l_all_files):
        parent_fname = file.parent.name
        if ignore_hidden_dirs and parent_fname.startswith("."):
            continue
        if ignore_hidden_files and file.name.startswith("."):
            continue
        if file.is_file():
            if file.suffix in s_suffixes:
                l_files.append(str(file))
    logger.info("Found {} files.".format(len(l_files)))
    return l_files


def change_suffixes(
    l_files: list, new_suffix: str, ref_dir: Union[str, Path, None] = None
) -> list:
    """Change the suffixes of a list of files
    If ref_dir is not None, the renamed files are checked
    to exist in ref_dir
    """
    l_new_files = []
    p_ref_dir = Path(ref_dir).resolve() if ref_dir else None
    l_new_files = [Path(file).with_suffix(new_suffix) for file in l_files]
    if p_ref_dir is None:
        return l_new_files
    func_cond = lambda f: p_ref_dir.joinpath(f.name).is_file()
    func_tgt = lambda f: p_ref_dir.joinpath(f.name).resolve()
    l_filtered, l_nonexistent = filter_list(
        l_new_files, func_cond, func_tgt, num_samples=3
    )
    if len(l_nonexistent) != 0:
        logger.warning(f"Found {len(l_nonexistent)} non-existent files")
        logger.warning(f"Few samples: {l_nonexistent}")
    return l_filtered


def save_dets(l_dets: List["DetectionResults"], path: str):
    with h5py.File(path, "w") as f:
        group = f.create_group("dets")
        for idx, dets in enumerate(l_dets):
            npy_bboxes = dets.to_numpy()
            dset = group.create_dataset(f"dets_{idx}", data=npy_bboxes)
            dset.attrs["width"] = dets.width
            dset.attrs["height"] = dets.height
            dset.attrs["img_name"] = dets.img_name
        logger.info(f"Detections saved to {path}")


def save_dets_as_label_studio(
    l_dets: List["DetectionResults"], path: str, subdir_images="images"
):
    """Save detections as Label Studio json format"""
    base_dir = "/data/local-files/?d={subdir_images}"
    l_json_data = [
        {
            "data": {
                "image": base_dir.format(
                    subdir_images=Path(subdir_images)
                    .joinpath(detection.img_name)
                    .as_posix()
                ),
            },
            "predictions": [
                {
                    "model_version": "one",
                    "score": 0.5,
                    "result": [
                        {
                            "id": f"bbox{i+1}",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": detection.width,
                            "original_height": detection.height,
                            "image_rotation": 0,
                            "value": {
                                "rotation": 0,
                                "x": bbox.x1 * 100,
                                "y": bbox.y1 * 100,
                                "width": bbox.w * 100,
                                "height": bbox.h * 100,
                                "rectanglelabels": [bbox.label],
                            },
                        }
                        for i, bbox in enumerate(detection.normalize().bboxes)
                    ],
                }
            ],
        }
        for detection in l_dets
    ]
    with open(path, "w") as f:
        json.dump(l_json_data, f, indent=2)


def load_dets(path: str) -> List["DetectionResults"]:
    with h5py.File(path, "r") as f:
        l_dets = []
        group = f["dets"]
        dets_keys = sorted(group.keys(), key=lambda x: int(x.split("_")[-1]))
        for key in dets_keys:
            dets_width = int(group[key].attrs["width"])
            dets_height = int(group[key].attrs["height"])
            dets_img_name = str(group[key].attrs["img_name"])
            dets_data = group[key][()]
            l_bboxes = [BBox.from_numpy(bbox) for bbox in dets_data]
            l_dets.append(
                DetectionResults(l_bboxes, dets_width, dets_height, dets_img_name)
            )
        return l_dets
