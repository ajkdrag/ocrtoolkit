from pathlib import Path

from chequeparser.utilities.core import filter_list
from loguru import logger
from tqdm.autonotebook import tqdm


def get_img_files(
    img_dir: Union[str, Path], ignore_hidden_dirs=True, ignore_hidden_files=True
) -> list:
    """Gets all types of image files from the directory.
    Supported filetypes: png, jpeg, jpg, tif
    Filter for ignoring hidden directories by default
    Filter for ignoring hidden files by default
    """
    p_img_dir = Path(img_dir).resolve()
    l_img_files = []
    l_all_files = list(p_img_dir.iterdir())
    s_suffixes = set([".png", ".jpeg", ".jpg", ".tif"])
    for file in tqdm(l_all_files):
        parent_fname = file.parent.name
        if ignore_hidden_dirs and parent_fname.startswith("."):
            continue
        if ignore_hidden_files and file.name.startswith("."):
            continue
        if file.is_file():
            if file.suffix in s_suffixes:
                img_files.append(file)
    logger.info("Found {} image files".format(len(l_img_files)))
    return l_img_files


def change_suffixes(
    l_files: list, new_suffix: str, ref_dir: Union[str, Path, None] = None
) -> list:
    """Change the suffixes of a list of files
    If ref_dir is not None, the renamed files are checked
    to exist in ref_dir
    """
    l_new_files = []
    p_ref_dir = Path(ref_dir).resolve() if ref_dir else None
    l_new_files = [f"{file}.{new_suffix}" for file in l_files]
    if p_ref_dir is not None:
        return l_new_files
    func_cond = lambda f: p_ref_dir.joinpath(f).is_file()
    func_tgt = lambda f: p_ref_dir.joinpath(f).resolve()
    l_filtered, l_nonexistent = filter_list(
        l_new_files, func_tgt, func_cond, num_samples=3
    )
    logger.warn("Found {} non-existent files".format(len(l_nonexistent)))
    logger.warn("Few samples: {}".format(l_nonexistent))
    return l_filtered
