"""Minimal reader for binary PCD files, enough for the INFRA-3DRC radar scans."""
from pathlib import Path

import numpy as np

_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4",
          ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4"}


def read_pcd(path):
    """Read a binary PCD file into a structured numpy array (one field per PCD field)."""
    header = {}
    with open(Path(path), "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line in header")
            parts = line.decode("ascii").split()
            if not parts or parts[0].startswith("#"):
                continue
            key, *values = parts
            header[key] = values
            if key == "DATA":
                break
        payload = f.read()

    if header["DATA"][0] != "binary":
        raise ValueError(f"{path}: only binary PCD is supported, got {header['DATA'][0]}")

    fields = header["FIELDS"]
    sizes = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    if any(c != 1 for c in counts):
        raise ValueError(f"{path}: fields with COUNT > 1 are not supported")

    dtype = np.dtype([(name, _TYPES[(t, s)]) for name, t, s in zip(fields, types, sizes)])
    n_points = int(header.get("POINTS", [len(payload) // dtype.itemsize])[0])
    return np.frombuffer(payload, dtype=dtype, count=n_points)
