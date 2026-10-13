#!/usr/bin/env python3
# PEBR-Net -- MIT License, see LICENSE.
"""Unpack and check the paired dataset and, once one is published, the trained checkpoint.

usage: python scripts/download_data.py --from PATH     # the downloaded archive (.zip) or folder: unpack and check
       python scripts/download_data.py --verify        # only check data/paired
       python scripts/download_data.py                 # download from the address in data/sources.json, if one is set
       python scripts/download_data.py --checkpoint    # also fetch the trained checkpoint listed in data/sources.json
       python scripts/download_data.py --describe      # write data/paired_content.json from data/paired

The six files are read with the package's own reader and compared with data/paired_content.json: rows, profiles,
gate times, depths and value sums (relative tolerance 1e-6), so a copy written with another number format, line ending
or encoding passes. A byte-identical copy (data/paired.sha256) is reported as such. Exit status 1: a file is missing,
unreadable or holds different content; 2: no download address is configured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"
FILES = [s + "_" + k + ".csv" for s in ("train", "val", "test") for k in ("clean", "noisy")]
RTOL = 1e-6


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> Dict[str, str]:
    sums: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                sums[name.strip().lstrip("*")] = digest.lower()
    return sums


def describe_file(path: Path) -> Dict[str, Any]:
    """Content of one paired file as the package reads it."""
    sys.path.insert(0, str(ROOT))
    import pebrnet
    d = pebrnet.read_paired_csv(path, logging.getLogger("data"))
    mat = np.asarray(d["matrix"], dtype=np.float64)
    sid = np.asarray(d["sample_id"])
    dep = np.asarray(d["depth"], dtype=np.float64)
    times = [float(str(g)[2:-2]) if str(g).startswith("t_") and str(g).endswith("_s") else str(g)
             for g in d["gate_names"]]
    return {"rows": int(mat.shape[0]), "gates": int(mat.shape[1]), "profiles": int(np.unique(sid).size),
            "gate_times_s": times, "depth_min": float(dep.min()), "depth_max": float(dep.max()),
            "depth_sum": float(dep.sum()), "value_sum": float(mat.sum()), "value_abs_sum": float(np.abs(mat).sum()),
            "value_sq_sum": float(np.square(mat).sum()),
            "gate_abs_sum": [float(v) for v in np.abs(mat).sum(axis=0)]}


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= RTOL * max(abs(float(a)), abs(float(b)), 1e-300)
    return a == b


def check_folder(folder: Path, content: Dict[str, Any], sums: Dict[str, str]) -> List[str]:
    """Problems found in `folder`; an empty list means every file is present and holds the described content."""
    problems = []
    for name in FILES:
        f = folder / name
        if not f.is_file():
            problems.append("missing: %s" % f)
            continue
        if sums.get(name) and sha256_of(f) == sums[name]:
            print("ok  %s (byte-identical)" % name)
            continue
        ref = content.get(name)
        if ref is None:
            problems.append("no content record for %s in data/paired_content.json" % name)
            continue
        try:
            got = describe_file(f)
        except Exception as exc:   # noqa: BLE001 - report and continue with the other files
            problems.append("unreadable: %s (%s)" % (f, exc))
            continue
        bad = [k for k in ref if not _close(ref[k], got.get(k))]
        if bad:
            problems.append("%s differs in: %s" % (name, ", ".join(bad)))
        else:
            print("ok  %s (same content)" % name)
    return problems


def download(urls: List[str], dest: Path, expect_sha256: Optional[str]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expect_sha256 and sha256_of(dest) == expect_sha256:
        print("already downloaded: %s" % dest)
        return dest
    last: Optional[Exception] = None
    for url in urls:
        part = dest.with_name(dest.name + ".part")
        try:
            print("downloading %s" % url)
            with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    out.write(chunk)
            if expect_sha256 and sha256_of(part) != expect_sha256:
                raise ValueError("the SHA-256 of the download differs from data/sources.json")
            part.replace(dest)
            return dest
        except Exception as exc:   # noqa: BLE001 - try the next address
            last = exc
            print("  failed: %s" % exc)
    raise RuntimeError("no address delivered %s (%s)" % (dest.name, last))


def collect(source: Path, folder: Path) -> None:
    """Copy the six files from anywhere inside an archive or a folder into `folder` (the shortest path wins)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if source.is_dir():
            found = sorted(source.rglob(name), key=lambda p: (len(p.parts), str(p)))
            if found:
                print("copying %s" % name)
                shutil.copyfile(found[0], folder / name)
            continue
        with zipfile.ZipFile(source) as zf:
            found = sorted((m for m in zf.infolist() if not m.is_dir() and Path(m.filename).name == name),
                           key=lambda m: (m.filename.count("/"), m.filename))
            if found:
                print("unpacking %s" % name)
                with zf.open(found[0]) as src, open(folder / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="source", default="", help="the downloaded archive (.zip) or folder of the dataset")
    ap.add_argument("--verify", action="store_true", help="only check data/paired")
    ap.add_argument("--checkpoint", action="store_true", help="also fetch the trained checkpoint")
    ap.add_argument("--describe", action="store_true", help="write data/paired_content.json from data/paired")
    ap.add_argument("--downloads", default=str(ROOT / "data" / "downloads"), help="where archives are kept")
    a = ap.parse_args(argv)
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    spec = src["dataset"]
    folder = ROOT / spec["folder"]
    content_path = ROOT / spec.get("content", "data/paired_content.json")
    if a.describe:
        content = {name: describe_file(folder / name) for name in FILES}
        content_path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        print("written: %s" % content_path)
        return 0
    sums = read_sums(ROOT / spec["file_checksums"]) if spec.get("file_checksums") else {}
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if not a.verify:
        if a.source:
            collect(Path(a.source), folder)
        elif not all((folder / name).is_file() for name in FILES):
            if not spec.get("urls"):
                where = spec.get("doi") or "the address in data/README.md"
                print("Download the dataset from %s, then run\n"
                      "  python scripts/download_data.py --from <downloaded archive or folder>" % where)
                return 2
            archive = download(spec["urls"], Path(a.downloads) / spec["archive"], spec.get("archive_sha256"))
            collect(archive, folder)
    problems = check_folder(folder, content, sums)
    if problems:
        print("\n".join(problems))
        return 1
    print("dataset checked: %d files in %s" % (len(FILES), folder))
    if a.checkpoint:
        ck = src.get("checkpoint", {})
        if not ck.get("urls"):
            print("No trained checkpoint is listed in data/sources.json yet.")
            return 2
        path = download(ck["urls"], ROOT / ck["folder"] / ck["archive"], ck.get("archive_sha256"))
        print("checkpoint: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
