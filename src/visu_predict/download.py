"""
Download the METR-LA / PEMS-BAY input files.

The files are the DCRNN releases of both datasets converted to CSV (one row
per 5-minute timestamp, one column per sensor) plus the DCRNN adjacency
pickles, hosted in the maintainer's public Google Drive folder:
https://drive.google.com/drive/folders/1eNGQpeHlxa7SWnpzIjeHFif4Ae15gjgs
"""

import os
from collections.abc import Iterable

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1eNGQpeHlxa7SWnpzIjeHFif4Ae15gjgs"

# file name -> (Google Drive file id, size in bytes)
FILES: dict[str, dict[str, tuple]] = {
    "METR-LA": {
        "METR-LA.csv": ("1q4-P4e_5o5mMlIr4DUf-BQFrmRgShgGH", 72_802_450),
        "adj_METR-LA.pkl": ("1z5LZC89dYZvQtWBeU-AxhQlwJt95Km05", 680_459),
    },
    "PEMS-BAY": {
        "PEMS-BAY.csv": ("14dfRl_m3nSQWhIuZSM_5uUkBA87d5-id", 85_771_183),
        "adj_PEMS-BAY.pkl": ("1Rzg5tSew7lXVDTj89wdKFiGZvgEnB1DR", 1_681_480),
    },
}


def download(datasets: Iterable[str] = ("METR-LA", "PEMS-BAY"), dest: str = "data",
             force: bool = False, quiet: bool = False) -> list[str]:
    """Download the traffic CSV and adjacency pickle of each dataset into ``dest``.

    Files already present with the expected size are skipped unless ``force``.
    Returns the paths of the files in ``dest``.
    """
    import gdown

    os.makedirs(dest, exist_ok=True)
    paths = []
    for ds in datasets:
        if ds not in FILES:
            raise ValueError(f"unknown dataset {ds!r}; available: {', '.join(FILES)}")
        for name, (file_id, size) in FILES[ds].items():
            path = os.path.join(dest, name)
            if not force and os.path.exists(path) and os.path.getsize(path) == size:
                print(f"{name}: already downloaded", flush=True)
                paths.append(path)
                continue
            print(f"{name}: downloading {size / 1e6:.1f} MB ...", flush=True)
            out = gdown.download(id=file_id, output=path, quiet=quiet)
            if out is None or not os.path.exists(path):
                raise RuntimeError(f"could not download {name}; get it manually from {DRIVE_FOLDER_URL}")
            if os.path.getsize(path) != size:
                raise RuntimeError(f"{name} has {os.path.getsize(path):,} bytes, expected {size:,}; "
                                   f"delete it and retry, or download it from {DRIVE_FOLDER_URL}")
            paths.append(path)
    return paths
