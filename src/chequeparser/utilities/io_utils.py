import tarfile
from pathlib import Path
from typing import Union

from loguru import logger
from PIL import Image
from tqdm.autonotebook import tqdm

from chequeparser.utilities.misc_utils import filter_list


def extract_files(tar_path: str, tar_file_name_list: list, output_dir: str):
    """Extracts files from a tar file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            for tar_file_name in tar_file_name_list:
                if member.name.endswith(tar_file_name):
                    filename = "inference" + tar_file_name
                    with Path(output_dir).joinpath(filename).open("wb") as f:
                        f.write(tar.extractfile(member).read())
                    logger.info(f"Extracted {member.name} ...")
                    break


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

    def func_cond(f):
        p_ref_dir.joinpath(f.name).is_file()

    def func_tgt(f):
        p_ref_dir.joinpath(f.name).resolve()

    l_filtered, l_nonexistent = filter_list(
        l_new_files, func_cond, func_tgt, num_samples=3
    )
    if len(l_nonexistent) != 0:
        logger.warning(f"Found {len(l_nonexistent)} non-existent files")
        logger.warning(f"Few samples: {l_nonexistent}")
    return l_filtered
