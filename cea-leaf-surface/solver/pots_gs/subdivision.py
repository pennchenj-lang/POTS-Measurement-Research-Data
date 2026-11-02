"""Uniform midpoint refinement; geometry remains the same piecewise planar set."""
from .geometry import Face, Mesh, Provenance, Vertex


def subdivide(mesh: Mesh) -> Mesh:
    mesh.validate()
    level = mesh.level + 1
    vertices = list(mesh.vertices)
    faces = []
    lineage = dict(mesh.lineage)
    midpoint_cache: dict[tuple[str, int, int], int] = {}

    def midpoint(a: int, b: int, organ_id: str) -> int:
        key = (organ_id, min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        va, vb = mesh.vertices[a], mesh.vertices[b]
        index = len(vertices)
        views = tuple(sorted(set(va.provenance.source_views) | set(vb.provenance.source_views)))
        origin = Provenance("interpolated", views, (va.vertex_id, vb.vertex_id),
                            "3D edge midpoint; derived variable, not a new observation; source_views are inherited dependencies")
        vertices.append(Vertex(f"{organ_id}:L{level}:e{min(a,b)}_{max(a,b)}", organ_id,
                               tuple((x + y) * .5 for x, y in zip(va.xyz, vb.xyz)),
                               tuple((x + y) * .5 for x, y in zip(va.uv, vb.uv)),
                               level, min(va.confidence, vb.confidence), origin))
        midpoint_cache[key] = index
        return index

    for parent in mesh.faces:
        a, b, c = parent.vertices
        ab, bc, ca = midpoint(a, b, parent.organ_id), midpoint(b, c, parent.organ_id), midpoint(c, a, parent.organ_id)
        child_ids = tuple(f"{parent.face_id}.{i}" for i in range(4))
        lineage[parent.face_id] = child_ids
        faces.extend(Face(child_ids[i], parent.organ_id, child, level, parent.face_id)
                     for i, child in enumerate(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))))
    result = Mesh(vertices, faces, level, mesh.data_type, mesh.coordinate_system, lineage,
                  list(mesh.petiole_controls), list(mesh.petiole_edges),
                  {**mesh.metadata, "operation": "pure_3d_midpoint_subdivision", "geometry_optimized": False,
                   "new_vertices_are_observations": False, "new_midpoints": len(midpoint_cache)})
    result.validate()
    return result


def subdivision_levels(mesh: Mesh, maximum_level: int = 3) -> list[Mesh]:
    if not isinstance(maximum_level, int) or maximum_level < mesh.level:
        raise ValueError("maximum_level must be an integer at least as large as the input level")
    result = [mesh]
    while result[-1].level < maximum_level:
        result.append(subdivide(result[-1]))
    return result
