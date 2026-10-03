"""PartNet samples and MPS initialization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from marching_primitives import FastMPSConfig, MPSParams, load_sdf_csv, mps_fast
from marching_primitives.mesh import merged_superquadrics_mesh

from .geometry import SQPrimitive, load_global_sdf_if_needed, mps_x_to_primitive
from .supervision import SQCoverageLabelResult


@dataclass
class PartNetSample:
    sample_id: str
    pts_path: str
    label_path: Optional[str]
    ply_path: Optional[str]
    sdf_csv_path: Optional[str]
    mesh_path: Optional[str] = None
    bootstrap_source: str = "mps"
    initial_sq_path: Optional[str] = None
    initial_sq_metadata_path: Optional[str] = None
    initial_sq_provenance: Optional[Dict[str, Any]] = None
    bootstrap_x: Optional[np.ndarray] = None
    bootstrap_info_cache: Optional[Dict[str, Any]] = None
    cached_sdf_csv_path: Optional[str] = None
    cached_sdf_global: Optional[np.ndarray] = None
    cached_sdf_grid: Any = None
    cached_sq_label_key: Optional[str] = None
    cached_sq_label_result: Optional[SQCoverageLabelResult] = None
    trust_sdf_csv: bool = False


def read_pts_file(path: str) -> np.ndarray:
    """Read .pts / .txt point file with xyz columns."""
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            vals = []
            for t in toks:
                try:
                    vals.append(float(t))
                except Exception:
                    pass
            if len(vals) >= 3:
                pts.append(vals[:3])
    if not pts:
        raise ValueError(f"No points found in {path}")
    return np.asarray(pts, dtype=np.float32)


def read_pts_normals_file(path: str) -> Optional[np.ndarray]:
    """Read optional normals from point files with xyz nx ny nz columns."""
    normals = []
    saw_normal = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = []
            for t in line.split():
                try:
                    vals.append(float(t))
                except Exception:
                    pass
            if len(vals) >= 6:
                normals.append(vals[3:6])
                saw_normal = True
            elif len(vals) >= 3:
                normals.append([0.0, 0.0, 0.0])
    if not saw_normal or not normals:
        return None
    arr = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    if not np.any(valid):
        return None
    median_norm = float(np.median(norms[valid, 0]))
    if not (0.25 <= median_norm <= 2.5):
        return None
    arr[valid] = arr[valid] / np.maximum(norms[valid], 1e-6)
    arr[~valid] = 0.0
    return arr


def read_label_file(path: str) -> np.ndarray:
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                labels.append(int(float(line.split()[0])))
            except Exception:
                continue
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return np.asarray(labels, dtype=np.int64)


def find_matching_file(
    parent: Path, prefix: str, sample_id: str, suffixes: Sequence[str]
) -> Optional[str]:
    for suf in suffixes:
        p = parent / f"{prefix}-{sample_id}{suf}"
        if p.exists():
            return str(p)
    return None


def resolve_sdf_csv(
    sample_id: str, sample_dir: Path, sdf_root: Optional[str]
) -> Optional[str]:
    candidates: List[Path] = []
    if sdf_root is not None:
        root = Path(sdf_root)
        if root.is_file() and root.suffix.lower() == ".csv":
            return str(root)
        candidates.extend(
            [
                root / f"{sample_id}.csv",
                root / f"sdf-{sample_id}.csv",
                root / f"{sample_id}_sdf.csv",
                root / sample_id / f"{sample_id}.csv",
            ]
        )
    candidates.extend(
        [
            sample_dir / f"{sample_id}.csv",
            sample_dir / f"sdf-{sample_id}.csv",
            sample_dir / f"{sample_id}_sdf.csv",
        ]
    )
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def resolve_uid_point_sample(
    uid: str, point_sample_root: str
) -> Optional[PartNetSample]:
    sample_dir = Path(point_sample_root) / uid / "point_sample"
    if not sample_dir.exists():
        return None
    label_path = sample_dir / "label-10000.txt"
    pts_candidates = [sample_dir / "pts-10000.txt", sample_dir / "pts-10000.pts"]
    ply_path = sample_dir / "ply-10000.ply"
    pts_path = next((str(p) for p in pts_candidates if p.exists()), None)
    if pts_path is None or not ply_path.exists():
        return None
    return PartNetSample(
        sample_id=uid,
        pts_path=pts_path,
        label_path=str(label_path) if label_path.exists() else None,
        ply_path=str(ply_path),
        sdf_csv_path=None,
        mesh_path=None,
    )


def discover_uid_mesh_samples(
    mesh_root: str, point_sample_root: str, sdf_root: Optional[str] = None
) -> List[PartNetSample]:
    mesh_root_path = Path(mesh_root)
    mesh_suffixes = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}
    mesh_files = sorted(
        p
        for p in mesh_root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in mesh_suffixes
    )
    samples: List[PartNetSample] = []
    for mesh_path in mesh_files:
        uid = mesh_path.stem
        sample = resolve_uid_point_sample(uid, point_sample_root)
        if sample is None:
            continue
        sample.mesh_path = str(mesh_path)
        sample.sdf_csv_path = resolve_sdf_csv(
            uid, mesh_path.parent, sdf_root
        ) or resolve_sdf_csv(uid, Path(sample.pts_path).parent, sdf_root)
        samples.append(sample)
    if not samples:
        raise FileNotFoundError(
            f"No UID-paired train-shape samples found under {mesh_root} with point samples in {point_sample_root}"
        )
    return samples


def discover_partnet_samples(
    root: str, sdf_root: Optional[str] = None
) -> List[PartNetSample]:
    root_path = Path(root)
    label_files = sorted(root_path.rglob("label-*.txt"))
    samples: List[PartNetSample] = []
    for label_path in label_files:
        sample_dir = label_path.parent
        sample_id = label_path.stem.split("label-")[-1]
        pts_path = find_matching_file(sample_dir, "pts", sample_id, [".pts", ".txt"])
        ply_path = find_matching_file(sample_dir, "ply", sample_id, [".ply"])
        sdf_csv_path = resolve_sdf_csv(sample_id, sample_dir, sdf_root)
        if pts_path is None:
            continue
        samples.append(
            PartNetSample(
                sample_id=sample_id,
                pts_path=pts_path,
                label_path=str(label_path),
                ply_path=ply_path,
                sdf_csv_path=sdf_csv_path,
            )
        )
    if not samples:
        raise FileNotFoundError(f"No PartNet-style samples found under {root}")
    return samples


def discover_samples(
    data_root: str, point_sample_root: Optional[str], sdf_root: Optional[str] = None
) -> List[PartNetSample]:
    root_path = Path(data_root)
    if sorted(root_path.rglob("label-*.txt")):
        return discover_partnet_samples(data_root, sdf_root=sdf_root)
    if point_sample_root is None:
        raise FileNotFoundError(
            f"{data_root} does not look like a PartNet point-sample root and --point-sample-root was not provided"
        )
    return discover_uid_mesh_samples(
        data_root, point_sample_root=point_sample_root, sdf_root=sdf_root
    )


def _sdf_csv_read_error(path: Path) -> Optional[str]:
    try:
        sdf, grid = load_sdf_csv(str(path))
        expected = int(grid.size[0]) * int(grid.size[1]) * int(grid.size[2])
        if int(np.asarray(sdf).size) != expected:
            return (
                f"unexpected sdf value count {int(np.asarray(sdf).size)} != {expected}"
            )
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _sdf_csv_quick_error(path: Path) -> Optional[str]:
    try:
        stat = path.stat()
        if stat.st_size <= 0:
            return "empty file"
        with open(path, "rb") as f:
            head = f.read(4096)
            if stat.st_size > 4096:
                f.seek(max(0, stat.st_size - 4096))
                tail = f.read(4096)
            else:
                tail = b""
        probe = head + tail
        if b"\x00" in probe:
            return "contains NUL bytes near file boundary"
        first = head.splitlines()[0].strip() if head else b""
        try:
            resolution = int(float(first.decode("ascii")))
        except Exception as exc:
            return f"invalid resolution header {first!r}: {exc}"
        if resolution <= 0:
            return f"non-positive resolution {resolution}"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _wait_for_sdf_csv(path: Path, lock_path: Path, timeout_sec: float = 3600.0) -> bool:
    start = time.time()
    while lock_path.exists():
        if path.exists() and _sdf_csv_read_error(path) is None:
            return True
        if time.time() - start > float(timeout_sec):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return False
        time.sleep(2.0)
    return path.exists() and _sdf_csv_read_error(path) is None


def _acquire_sdf_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} time={time.time():.3f}\n")
    return True


def maybe_generate_sdf_csv(
    sample: PartNetSample, mesh2sdf_script: Optional[str], sdf_out_dir: Optional[str]
) -> Optional[str]:
    if sample.sdf_csv_path and os.path.exists(sample.sdf_csv_path):
        if bool(getattr(sample, "trust_sdf_csv", False)):
            return sample.sdf_csv_path
        existing = Path(sample.sdf_csv_path)
        error = _sdf_csv_read_error(existing)
        if error is None:
            return sample.sdf_csv_path
        print(
            f"[sdf] WARNING: invalid SDF csv for {sample.sample_id}: {existing} ({error})"
        )
        sample.sdf_csv_path = None
    mesh_path = sample.mesh_path or sample.ply_path
    if mesh2sdf_script is None or mesh_path is None:
        return None
    out_dir = Path(sdf_out_dir or (Path(mesh_path).parent / "sdf_csv"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{sample.sample_id}.csv"
    lock_path = out_csv.with_suffix(out_csv.suffix + ".lock")
    if out_csv.exists():
        error = _sdf_csv_read_error(out_csv)
        if error is None:
            sample.sdf_csv_path = str(out_csv)
            return str(out_csv)
        if lock_path.exists() and _wait_for_sdf_csv(out_csv, lock_path):
            sample.sdf_csv_path = str(out_csv)
            return str(out_csv)
        print(
            f"[sdf] WARNING: removing invalid generated SDF csv for {sample.sample_id}: {out_csv} ({error})"
        )
        try:
            out_csv.unlink()
        except FileNotFoundError:
            pass

    while not _acquire_sdf_lock(lock_path):
        if _wait_for_sdf_csv(out_csv, lock_path):
            sample.sdf_csv_path = str(out_csv)
            return str(out_csv)

    tmp_csv = out_csv.with_name(f".{out_csv.stem}.{os.getpid()}.tmp.csv")
    cmd = [sys.executable, mesh2sdf_script, mesh_path, "--output", str(tmp_csv)]
    try:
        if tmp_csv.exists():
            tmp_csv.unlink()
        subprocess.run(cmd, check=True)
        error = _sdf_csv_read_error(tmp_csv)
        if error is not None:
            raise RuntimeError(f"generated invalid SDF csv {tmp_csv}: {error}")
        os.replace(tmp_csv, out_csv)
        sample.sdf_csv_path = str(out_csv)
        return str(out_csv)
    except Exception as exc:
        print(
            f"[sdf] WARNING: failed to generate SDF csv for {sample.sample_id}: {exc}"
        )
        return None
    finally:
        try:
            tmp_csv.unlink()
        except FileNotFoundError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _bootstrap_x_from_mps(
    sample: PartNetSample,
    sdf_csv_path: Optional[str],
    global_cfg: Dict[str, Any],
    bootstrap_info: Dict[str, Any],
) -> np.ndarray:
    x = np.zeros((0, 11), dtype=np.float64)
    if sdf_csv_path is not None:
        sdf, grid = load_sdf_csv(sdf_csv_path)
        params = MPSParams(
            max_division=int(global_cfg.get("max_division", 50)),
            verbose=bool(global_cfg.get("verbose", False)),
            region_timeout_sec=global_cfg.get("region_timeout_sec", None),
            max_region_refits=global_cfg.get("max_region_refits", None),
            heartbeat_sec=float(global_cfg.get("heartbeat_sec", 30.0)),
        )
        fast_cfg = FastMPSConfig(
            num_workers=global_cfg.get("num_workers", None),
            chunk_size=max(1, int(global_cfg.get("chunk_size", 1))),
            parallel_min_regions=max(1, int(global_cfg.get("parallel_min_regions", 3))),
        )
        x = np.asarray(mps_fast(sdf, grid, params, fast_cfg), dtype=np.float64)
        bootstrap_info["global_optimizer"] = "mps_fast"
    return x


def bootstrap_primitives(
    sample: PartNetSample,
    out_dir: str,
    global_cfg: Dict[str, Any],
    write_outputs: bool = True,
) -> Tuple[List[SQPrimitive], Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    if sample.bootstrap_x is None:
        source = "mps"
        sdf_csv_path = maybe_generate_sdf_csv(
            sample,
            mesh2sdf_script=global_cfg.get("mesh2sdf_script"),
            sdf_out_dir=global_cfg.get("sdf_out_dir"),
        )
        if sdf_csv_path is None:
            raise RuntimeError(
                f"No SDF csv for sample {sample.sample_id}. Provide --sdf-root "
                f"or a working --mesh2sdf-script."
            )

        x = np.zeros((0, 11), dtype=np.float64)
        bootstrap_info: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "sdf_csv_path": sdf_csv_path or "",
            "bootstrap_source": source,
            "fallback_single_bbox": False,
        }

        x = _bootstrap_x_from_mps(sample, sdf_csv_path, global_cfg, bootstrap_info)

        if x.size == 0:
            raise RuntimeError(f"MPS returned no SQs for sample {sample.sample_id}")

        sample.bootstrap_x = np.array(x, copy=True)
        sample.bootstrap_info_cache = dict(bootstrap_info)
    else:
        x = np.array(sample.bootstrap_x, copy=True)
        bootstrap_info = dict(sample.bootstrap_info_cache or {})
        bootstrap_info.setdefault("sample_id", sample.sample_id)
        bootstrap_info.setdefault("sdf_csv_path", sample.sdf_csv_path or "")
        if bootstrap_info.get("bootstrap_source", "mps") != "mps" or bootstrap_info.get(
            "fallback_single_bbox"
        ):
            raise ValueError(
                f"Reused SQs for {sample.sample_id} were not initialized by MPS"
            )

    if write_outputs:
        np.save(os.path.join(out_dir, f"{sample.sample_id}_initial_sq.npy"), x)
        try:
            v, f = merged_superquadrics_mesh(x)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            mesh.export(os.path.join(out_dir, f"{sample.sample_id}_initial_sq.ply"))
        except Exception:
            pass
        provenance_payload = {
            k: v for k, v in bootstrap_info.items() if k != "sample_id"
        }
        provenance_payload["sample_id"] = sample.sample_id
        try:
            with open(
                os.path.join(out_dir, f"{sample.sample_id}_initial_sq_provenance.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(provenance_payload, f, ensure_ascii=False, indent=2)
        except TypeError:
            pass

    prims = [mps_x_to_primitive(row) for row in x]
    bootstrap_info["bootstrap_num_prims"] = float(len(prims))
    bootstrap_info["initial_sq_path"] = (
        os.path.join(out_dir, f"{sample.sample_id}_initial_sq.npy")
        if write_outputs
        else ""
    )
    return prims, bootstrap_info


def load_cached_global_sdf(
    sample: PartNetSample, sdf_csv_path: Optional[str]
) -> Tuple[Optional[np.ndarray], Any]:
    if sdf_csv_path is None:
        return None, None
    if (
        sample.cached_sdf_csv_path == sdf_csv_path
        and sample.cached_sdf_global is not None
        and sample.cached_sdf_grid is not None
    ):
        return sample.cached_sdf_global, sample.cached_sdf_grid
    sdf_global, sdf_grid = load_global_sdf_if_needed(sdf_csv_path)
    sample.cached_sdf_csv_path = sdf_csv_path
    sample.cached_sdf_global = sdf_global
    sample.cached_sdf_grid = sdf_grid
    return sdf_global, sdf_grid
