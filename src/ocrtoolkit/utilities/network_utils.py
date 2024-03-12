import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm.autonotebook import tqdm

from ocrtoolkit.utilities.io_utils import extract_files


def retrieve_file(url: str, filename: Path, chunk_size: int = 1024) -> None:
    """
    Helper function to retrieve a file from a URL.

    Args:
        url (str): URL of the file to download.
        filename (Path): Destination filename as a Path object.
        chunk_size (int): Size of the chunks for downloading the file.
    """
    with filename.open("wb") as fh, urllib.request.urlopen(url) as response:
        progress_bar = tqdm(total=response.length)
        for chunk in iter(lambda: response.read(chunk_size), ""):
            if not chunk:
                break
            progress_bar.update(chunk_size)
            fh.write(chunk)
        progress_bar.close()


def verify_file_integrity(file_path: Path, hash_prefix: str) -> bool:
    """
    Verifies the integrity of a file using its SHA256 hash.

    Args:
        file_path (Path): Location of the file to verify.
        hash_prefix (str): Expected SHA256 hash prefix of the file.

    Returns:
        bool: True if the file integrity is verified, False otherwise.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest().startswith(hash_prefix)


def download_file(
    url: str,
    file_name: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_subdir: Optional[str] = None,
    untar: bool = False,
) -> Path:
    """
    Download a file from a URL to a local directory, optionally verifying its hash.

    Args:
        url: URL of the file to download.
        file_name: Optional custom file name for the downloaded file.
        hash_prefix: Optional expected SHA256 hash prefix to verify the file.
        cache_dir: Root directory for caching the file.
        cache_subdir: Optional subdirectory to use for caching.
        untar: Whether to untar the file.

    Returns:
        Path: The path to the downloaded (and verified) file.
        If untar is True, the file is untarred and folder is returned.
    """
    if file_name is None:
        file_name = Path(url).name.split("&")[0]

    cache_dir_path = Path(cache_dir or Path.home() / ".cache")
    if cache_subdir:
        cache_dir_path /= cache_subdir
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir_path / file_name

    if file_path.is_file() and (
        hash_prefix is None or verify_file_integrity(file_path, hash_prefix)
    ):
        logger.info(f"Found {file_path}. Skipping download.")
    else:
        try:
            logger.info(f"Downloading to {file_path}")
            retrieve_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                alternative_url = url.replace("https:", "http:")
                retrieve_file(alternative_url, file_path)
            else:
                raise e

        if hash_prefix and not verify_file_integrity(file_path, hash_prefix):
            file_path.unlink()
            raise ValueError(
                f"Corrupted download, hash of {url} does not match its expected value"
            )

    if untar and file_path.suffix == ".tar" and file_path.is_file():
        output_dir = file_path.parent.joinpath(file_path.stem)
        if output_dir.is_dir():
            logger.info(f"Found {output_dir}. Skipping untar.")
            return output_dir.as_posix()
        logger.info(f"Extracting to {output_dir}")
        suffixes = [".pdiparams", ".pdiparams.info", ".pdmodel"]
        extract_files(file_path.as_posix(), suffixes, output_dir)
        return output_dir.as_posix()

    return file_path.as_posix()
