"""POTS-GS: Part-ID-Guided Organ Topological Scaffold Gaussian Splatting.

中文方法名：Part-ID引导的器官拓扑支架高斯重建。
The implemented geometry core is POTS; full Gaussian joint optimization is not
implemented. Midpoint subdivision refines the scaffold. Height and reference
ground-plane handling belong to the coordinate module. The pots_gs import name
and existing public API remain unchanged.
"""
from .geometry import (LEAF_FACES, LEAF_LABELS, Face, Mesh, Provenance, Vertex,
                       combine_meshes, create_leaf_mesh, mesh_area, mesh_from_dict)
from .subdivision import subdivide, subdivision_levels

__version__ = "0.1.0"
__all__ = ["LEAF_FACES", "LEAF_LABELS", "Face", "Mesh", "Provenance", "Vertex",
           "combine_meshes", "create_leaf_mesh", "mesh_area", "mesh_from_dict", "subdivide", "subdivision_levels"]
