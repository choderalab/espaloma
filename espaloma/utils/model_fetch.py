from pathlib import Path
from typing import Any

import requests
import torch.utils.model_zoo
from tqdm import tqdm


def _get_model_url(version: str) -> str:
    """
    Get the URL of the espaloma model from GitHub releases.

    Parameters:
        version (str): Version of the model. If set to "latest", the URL for the latest version will be returned.

    Returns:
        str: The URL of the espaloma model.

    Note:
        - If version is set to "latest", the URL for the latest version of the model will be returned.
        - The URL is obtained from the GitHub releases of the espaloma repository.

    Example:
        >>> url = _get_model_url(version="0.3.0")
    """

    if version == "latest":
        url = "https://github.com/choderalab/espaloma/releases/latest/download/espaloma-latest.pt"
    else:
        # TODO: This scheme requires the version string of the model to match the
        # release version
        url = f"https://github.com/choderalab/espaloma/releases/download/{version}/espaloma-{version}.pt"

    return url


def get_model_path(
    model_dir: str | Path = ".espaloma/",
    version: str = "latest",
    disable_progress_bar: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Download a model for espaloma.

    Parameters:
        model_dir (str or Path): Directory path where the model will be saved. Default is ``.espaloma/``.
        version (str): Version of the model to download. Default is "latest".
        disable_progress_bar (bool): Whether to disable the progress bar during the download. Default is False.
        overwrite (bool): Whether to overwrite the existing model file if it exists. Default is False.

    Returns:
        Path: The path to the downloaded model file.

    Raises:
        FileExistsError: If the model file already exists and overwrite is set to False.

    Note:
        - If version is set to "latest", the latest version of the model will be downloaded.
        - The model will be downloaded from GitHub releases.
        - The model file will be saved in the specified model directory.

    Example:
        >>> model_path = get_model(model_dir=".espaloma/", version="0.3.0", disable_progress_bar=True)
    """

    url = _get_model_url(version)

    # This will work as long as we never have a "/" in the version string
    file_name = Path(url.split("/")[-1])
    model_dir = Path(model_dir)
    model_path = Path(model_dir / file_name)

    if not overwrite and model_path.exists():
        raise FileExistsError(
            f"File '{model_path}' exiits, use overwrite=True to overwrite file"
        )
    model_dir.mkdir(parents=True, exist_ok=True)

    request = requests.get(url, stream=True)
    request_lenght = int(request.headers.get("content-length", 0))
    with open(model_path, "wb") as file, tqdm(
        total=request_lenght,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        disable=disable_progress_bar,
    ) as progress:
        for data in request.iter_content(chunk_size=1024):
            size = file.write(data)
            progress.update(size)

    return model_path


def get_model(version: str = "latest") -> dict[str, Any]:
    """
        Load an espaloma model from GitHub releases.

    Parameters:
        version (str): Version of the model to load. Default is "latest".

    Returns:
        dict[str, Any]: The loaded espaloma model.

    Note:
        - If version is set to "latest", the latest version of the model will be loaded.
        - The model will be loaded from GitHub releases.
        - The model will be loaded onto the CPU.

    Example:
        >>> model = get_model(version="0.3.0")
    """

    url = _get_model_url(version)
    model = torch.utils.model_zoo.load_url(url, map_location="cpu")
    model.eval()  # type: ignore

    return model
