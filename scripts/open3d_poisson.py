#!/usr/bin/env python3
"""Screened Poisson reconstruction backend (Open3D / Kazhdan & Hoppe 2013).

Takes the oriented dense cloud written by sfm2mesh (`*_oriented.ply`) and
produces a watertight, smooth mesh — the industry-standard counterpart to the
in-house TSDF + marching-tetrahedra path.

Usage:
  python open3d_poisson.py <oriented.ply> <out_mesh.ply> [--depth 9]
                           [--target-faces 20000] [--reference our_mesh.ply]

Prints a JSON summary (vertices, triangles, watertight, and — when a reference
mesh is given — symmetric nearest-neighbour distances).
"""

import argparse
import json

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud")
    parser.add_argument("output")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--target-faces", type=int, default=20000)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--density-quantile", type=float, default=0.02)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.cloud)
    if not pcd.has_points():
        raise SystemExit(f"no points in {args.cloud}")
    if not pcd.has_normals():
        pcd.estimate_normals()
    pcd.normalize_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth, width=0, scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities)
    if args.density_quantile > 0.0:
        cutoff = np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(densities < cutoff)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if args.target_faces > 0 and len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
    mesh.compute_vertex_normals()

    # Transfer colours from the cloud (nearest neighbour).
    colours = np.asarray(pcd.colors)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    if len(colours) > 0:
        for i, v in enumerate(np.asarray(mesh.vertices)):
            _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
            vertex_colors[i] = colours[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.io.write_triangle_mesh(
        args.output, mesh, write_vertex_normals=True, write_vertex_colors=True
    )

    summary = {
        "backend": "open3d-screened-poisson",
        "depth": args.depth,
        "points_in": len(pcd.points),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        # Open3D defines watertightness per edge/vertex manifoldness.
        "watertight": bool(mesh.is_watertight()),
        "output": args.output,
    }

    if args.reference:
        reference = o3d.io.read_triangle_mesh(args.reference)
        reference.compute_vertex_normals()
        a = np.asarray(mesh.vertices)
        b = np.asarray(reference.vertices)
        if len(a) and len(b):
            tree_a = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(a)))
            tree_b = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(b)))
            d_ab = np.array([tree_b.search_knn_vector_3d(p, 1)[2][0] for p in a])
            d_ba = np.array([tree_a.search_knn_vector_3d(p, 1)[2][0] for p in b])
            summary["reference"] = {
                "vertices": len(b),
                "triangles": len(reference.triangles),
                "watertight": bool(reference.is_watertight()),
                "mean_nn_distance": float(np.sqrt(d_ab).mean()),
                "hausdorff_nn_distance": float(
                    max(np.sqrt(d_ab).max(), np.sqrt(d_ba).max())
                ),
            }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
