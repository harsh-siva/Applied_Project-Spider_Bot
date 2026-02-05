#!/usr/bin/env python3
"""
Deterministically convert URDF mesh visuals into link-local meshes:

- Input: an already-expanded URDF (no xacro), e.g. /tmp/spyderr.urdf
- For every <link><visual><geometry><mesh filename=...> with a non-identity <origin>:
    * copy the referenced STL into an output dir (meshes/link_local/)
    * apply the visual origin transform to vertices (so origin becomes identity)
    * rewrite URDF to point to the new mesh and set origin xyz/rpy to 0
- Write a CSV report of all applied transforms and file hashes.
- Validation: compare each link's mesh centroid and AABB (axis-aligned bounding box)
  in the base frame at q=0 between:
    (A) original URDF and (B) rewritten URDF
  Must match within tolerance (default 1e-6 m).

Notes:
- This script assumes mesh URIs resolve on the local filesystem (file:// or package://).
- It handles STL only (your repo uses STL). If other formats appear, it will error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from stl import mesh as stl_mesh

# -------------------------
# Math helpers (RPY + SE3)
# -------------------------

def rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF uses fixed-axis roll(X), pitch(Y), yaw(Z), applied in that order."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=float)
    return (Rz @ Ry @ Rx)

def make_T(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T

def apply_T_to_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,3)."""
    ones = np.ones((pts.shape[0], 1), dtype=float)
    ph = np.hstack([pts, ones])  # (N,4)
    out = (T @ ph.T).T
    return out[:, :3]

# -------------------------
# URDF parsing (minimal)
# -------------------------

def parse_xyz_rpy(origin_el: Optional[ET.Element]) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    if origin_el is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(x) for x in origin_el.get("xyz", "0 0 0").split())
    rpy = tuple(float(r) for r in origin_el.get("rpy", "0 0 0").split())
    return xyz, rpy

@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin_T: np.ndarray  # parent->child at q=0
    axis: np.ndarray      # in child frame per URDF convention? For q=0 only, axis unused here.

def load_joints(root: ET.Element) -> Dict[str, Joint]:
    joints: Dict[str, Joint] = {}
    for j in root.findall("joint"):
        jname = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        xyz, rpy = parse_xyz_rpy(j.find("origin"))
        T = make_T(xyz, rpy)
        axis_el = j.find("axis")
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if axis_el is not None and axis_el.get("xyz") is not None:
            axis = np.array([float(v) for v in axis_el.get("xyz").split()], dtype=float)
        joints[jname] = Joint(jname, parent, child, T, axis)
    return joints

def build_parent_map(joints: Dict[str, Joint]) -> Dict[str, Tuple[str, np.ndarray]]:
    """child_link -> (parent_link, T_parent_child)."""
    pm: Dict[str, Tuple[str, np.ndarray]] = {}
    for j in joints.values():
        pm[j.child] = (j.parent, j.origin_T)
    return pm

def link_T_base(link: str, parent_map: Dict[str, Tuple[str, np.ndarray]], base_link: str) -> np.ndarray:
    """Return T_base_link (base->link) by chaining joint origins at q=0."""
    if link == base_link:
        return np.eye(4, dtype=float)
    chain: List[np.ndarray] = []
    cur = link
    visited = set()
    while cur != base_link:
        if cur in visited:
            raise RuntimeError(f"Cycle detected in kinematic tree at link '{cur}'.")
        visited.add(cur)
        if cur not in parent_map:
            raise RuntimeError(f"Link '{cur}' has no parent in joint tree (base_link='{base_link}').")
        parent, T_parent_child = parent_map[cur]
        chain.append((parent, T_parent_child))
        cur = parent
    # chain currently: (parent_of_leaf, T_parent_leaf), ..., up to base
    # build forward from base:
    T = np.eye(4, dtype=float)
    for parent, T_parent_child in reversed(chain):
        T = T @ T_parent_child
    return T

# -------------------------
# Mesh IO + hashing
# -------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def resolve_mesh_uri(uri: str, workspace_root: Path) -> Path:
    """
    Support:
      - file:///abs/path
      - package://<pkg>/<relpath>  (prefer install if the file exists there, else src)
      - absolute /path
    """
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    if uri.startswith("/"):
        return Path(uri)
    if uri.startswith("package://"):
        # package://<pkg>/<relpath>
        rest = uri[len("package://"):]
        pkg, rel = rest.split("/", 1)

        install_share = workspace_root / "install" / pkg / "share" / pkg
        src_share = workspace_root / "src" / pkg

        # Prefer install only if the target file exists there; otherwise fall back to src.
        install_path = install_share / rel
        if install_path.exists():
            return install_path

        src_path = src_share / rel
        if src_path.exists():
            return src_path

        # Neither exists: return install_path so the eventual FileNotFoundError is stable/predictable.
        return install_path

    raise ValueError(f"Unsupported mesh URI scheme: {uri}")

def read_stl_vertices(stl_path: Path) -> np.ndarray:
    m = stl_mesh.Mesh.from_file(str(stl_path))
    # m.vectors: (N,3,3) triangles
    return m.vectors.reshape((-1, 3)).astype(float)

def write_stl_from_template(template_path: Path, new_vertices: np.ndarray, out_path: Path) -> None:
    m = stl_mesh.Mesh.from_file(str(template_path))
    if new_vertices.shape[0] != m.vectors.reshape((-1, 3)).shape[0]:
        raise RuntimeError("Vertex count mismatch writing STL.")
    m.vectors = new_vertices.reshape(m.vectors.shape)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))

# -------------------------
# Metrics for validation
# -------------------------

@dataclass
class MeshMetrics:
    centroid: np.ndarray  # (3,)
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)

def metrics_from_points(pts: np.ndarray) -> MeshMetrics:
    c = pts.mean(axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return MeshMetrics(c, mn, mx)

def metrics_close(a: MeshMetrics, b: MeshMetrics, tol: float) -> bool:
    return (np.allclose(a.centroid, b.centroid, atol=tol, rtol=0) and
            np.allclose(a.aabb_min, b.aabb_min, atol=tol, rtol=0) and
            np.allclose(a.aabb_max, b.aabb_max, atol=tol, rtol=0))

# -------------------------
# Main conversion
# -------------------------

def convert(
    urdf_in: Path,
    urdf_out: Path,
    workspace_root: Path,
    out_mesh_dir: Path,
    report_csv: Path,
    tol: float,
) -> None:
    rootA = ET.parse(urdf_in).getroot()

    # Determine base link (root link of tree).
    jointsA = load_joints(rootA)
    parent_map = build_parent_map(jointsA)

    all_links = [l.get("name") for l in rootA.findall("link")]
    children = set(pm_child for pm_child in parent_map.keys())
    parents = set(p for (p, _) in parent_map.values())
    # root link is a parent but never a child (typical URDF tree)
    candidates = [l for l in all_links if l not in children]
    if len(candidates) != 1:
        raise RuntimeError(f"Could not uniquely determine base link. Candidates: {candidates}")
    base_link = candidates[0]

    # Precompute base->link transforms at q=0 for validation use.
    T_base_link: Dict[str, np.ndarray] = {base_link: np.eye(4)}
    for l in all_links:
        if l == base_link:
            continue
        T_base_link[l] = link_T_base(l, parent_map, base_link)

    # Collect original metrics per visual mesh (in base frame at q=0)
    original_metrics: Dict[Tuple[str,int], MeshMetrics] = {}

    def compute_visual_points_in_base(root: ET.Element) -> Dict[Tuple[str,int], np.ndarray]:
        pts_map: Dict[Tuple[str,int], np.ndarray] = {}
        for link in root.findall("link"):
            ln = link.get("name")
            T_bl = T_base_link[ln]
            for i, vis in enumerate(link.findall("visual")):
                mesh_el = vis.find("./geometry/mesh")
                if mesh_el is None:
                    continue
                uri = mesh_el.get("filename", "")
                scale_str = mesh_el.get("scale", "1 1 1")
                sx, sy, sz = (float(s) for s in scale_str.split())
                xyz, rpy = parse_xyz_rpy(vis.find("origin"))
                T_lv = make_T(xyz, rpy)  # link->visual
                mesh_path = resolve_mesh_uri(uri, workspace_root)
                if mesh_path.suffix.lower() != ".stl":
                    raise RuntimeError(f"Non-STL mesh not supported: {mesh_path}")
                v = read_stl_vertices(mesh_path)
                v = v * np.array([sx, sy, sz], dtype=float)  # apply scale
                vL = apply_T_to_points(T_lv, v)              # into link
                vB = apply_T_to_points(T_bl, vL)             # into base
                pts_map[(ln, i)] = vB
        return pts_map

    ptsA = compute_visual_points_in_base(rootA)
    for k, pts in ptsA.items():
        original_metrics[k] = metrics_from_points(pts)

    # Now rewrite URDF and generate link-local meshes
    rootB = ET.fromstring(ET.tostring(rootA))  # deep copy

    report_rows = []
    for link in rootB.findall("link"):
        ln = link.get("name")
        for i, vis in enumerate(link.findall("visual")):
            mesh_el = vis.find("./geometry/mesh")
            if mesh_el is None:
                continue

            uri = mesh_el.get("filename", "")
            scale_str = mesh_el.get("scale", "1 1 1")
            sx, sy, sz = (float(s) for s in scale_str.split())
            origin_el = vis.find("origin")
            xyz, rpy = parse_xyz_rpy(origin_el)
            if np.allclose(np.array(xyz), 0.0) and np.allclose(np.array(rpy), 0.0):
                continue  # already identity; leave as-is

            mesh_path = resolve_mesh_uri(uri, workspace_root)
            if mesh_path.suffix.lower() != ".stl":
                raise RuntimeError(f"Non-STL mesh not supported: {mesh_path}")

            # Read, apply scale, then bake T_lv into vertices.
            verts = read_stl_vertices(mesh_path) * np.array([sx, sy, sz], dtype=float)
            T_lv = make_T(xyz, rpy)
            baked = apply_T_to_points(T_lv, verts)

            # Output mesh name: <link>__visual<idx>.stl to avoid collisions
            out_name = f"{ln}__visual{i}.stl"
            out_path = out_mesh_dir / out_name

            # Write mesh using original STL as template (preserves triangle topology)
            write_stl_from_template(mesh_path, baked, out_path)

            # Update URDF: point to new mesh and set origin to identity
            # Keep scale at 1 1 1 now that baked already includes it
            mesh_el.set("filename", f"package://spyderr_description/meshes/link_local/{out_name}")
            mesh_el.set("scale", "1 1 1")
            if origin_el is None:
                origin_el = ET.SubElement(vis, "origin")
            origin_el.set("xyz", "0 0 0")
            origin_el.set("rpy", "0 0 0")

            report_rows.append({
                "link": ln,
                "visual_idx": i,
                "src_uri": uri,
                "src_path": str(mesh_path),
                "src_sha256": sha256_file(mesh_path),
                "dst_rel": f"meshes/link_local/{out_name}",
                "dst_path": str(out_path),
                "dst_sha256": sha256_file(out_path),
                "applied_xyz": f"{xyz[0]} {xyz[1]} {xyz[2]}",
                "applied_rpy": f"{rpy[0]} {rpy[1]} {rpy[2]}",
                "scale_baked": f"{sx} {sy} {sz}",
            })

    # Write output URDF
    urdf_out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(rootB).write(urdf_out, encoding="utf-8", xml_declaration=True)

    # Write report
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "link","visual_idx","src_uri","src_path","src_sha256",
        "dst_rel","dst_path","dst_sha256",
        "applied_xyz","applied_rpy","scale_baked"
    ]
    with report_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    # Validate metrics equivalence in base frame at q=0
    rootB_loaded = ET.parse(urdf_out).getroot()
    ptsB = compute_visual_points_in_base(rootB_loaded)
    failures = []
    for k, mA in original_metrics.items():
        if k not in ptsB:
            failures.append((k, "missing in B"))
            continue
        mB = metrics_from_points(ptsB[k])
        if not metrics_close(mA, mB, tol):
            failures.append((k, "metrics differ"))

    if failures:
        msg = "\n".join([f"{k}: {reason}" for k, reason in failures[:50]])
        raise RuntimeError(
            f"VALIDATION FAILED for {len(failures)} visuals (tol={tol}). "
            f"First failures:\n{msg}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-in", required=True, type=Path)
    ap.add_argument("--urdf-out", required=True, type=Path)
    ap.add_argument("--workspace-root", required=True, type=Path)
    ap.add_argument("--out-mesh-dir", required=True, type=Path)
    ap.add_argument("--report-csv", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    convert(
        urdf_in=args.urdf_in,
        urdf_out=args.urdf_out,
        workspace_root=args.workspace_root,
        out_mesh_dir=args.out_mesh_dir,
        report_csv=args.report_csv,
        tol=args.tol,
    )
    print("OK: conversion + validation passed")

if __name__ == "__main__":
    main()
