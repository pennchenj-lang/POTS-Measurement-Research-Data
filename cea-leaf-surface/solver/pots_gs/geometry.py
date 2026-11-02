"""Identity-preserving triangular organ scaffolds; no Gaussian backend is implied."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]
LEAF_LABELS = ("B", "L1", "C1", "R1", "L2", "C2", "R2", "L3", "C3", "R3", "T")
# Canonical oriented triangles, indexed into LEAF_LABELS above.
LEAF_FACES = ((0, 2, 1), (0, 3, 2), (1, 2, 5), (1, 5, 4),
              (2, 3, 6), (2, 6, 5), (4, 5, 8), (4, 8, 7),
              (5, 6, 9), (5, 9, 8), (7, 8, 10), (8, 9, 10))
LEAF_UV = ((0.0, 0.0), (.25, -1.0), (.25, 0.0), (.25, 1.0),
           (.5, -1.0), (.5, 0.0), (.5, 1.0),
           (.75, -1.0), (.75, 0.0), (.75, 1.0), (1.0, 0.0))
SOURCE_KINDS = {"observed", "estimated", "synthetic_control", "interpolated", "surface_sample"}


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))


def scale(a: Vec3, t: float) -> Vec3:
    return tuple(t * x for x in a)


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


@dataclass(frozen=True)
class Provenance:
    kind: str
    source_views: tuple[str, ...] = ()
    contributors: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Vertex:
    vertex_id: str
    organ_id: str
    xyz: Vec3
    uv: Vec2
    level: int
    confidence: float
    provenance: Provenance
    label: str = ""


@dataclass(frozen=True)
class Face:
    face_id: str
    organ_id: str
    vertices: tuple[int, int, int]
    level: int
    parent_face_id: str | None = None


@dataclass
class Mesh:
    vertices: list[Vertex]
    faces: list[Face]
    level: int = 0
    data_type: str = "synthetic"
    coordinate_system: str = "right_handed; x=lateral, y=base_to_tip, z=normal; units=metre"
    lineage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    petiole_controls: list[Vertex] = field(default_factory=list)
    petiole_edges: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def edge_incidence(self) -> dict[tuple[str, int, int], list[tuple[int, int, int]]]:
        edges: dict[tuple[str, int, int], list[tuple[int, int, int]]] = {}
        for fi, face in enumerate(self.faces):
            a, b, c = face.vertices
            for start, end in ((a, b), (b, c), (c, a)):
                key = (face.organ_id, min(start, end), max(start, end))
                edges.setdefault(key, []).append((fi, start, end))
        return edges

    def validate(self) -> None:
        if self.data_type not in {"synthetic", "real"}:
            raise ValueError("data_type must explicitly be synthetic or real")
        if not isinstance(self.level, int) or self.level < 0:
            raise ValueError("level must be a nonnegative integer")
        if not self.vertices or not self.faces:
            raise ValueError("a surface mesh must contain vertices and faces")
        all_vertices = self.vertices + self.petiole_controls
        ids = [v.vertex_id for v in all_vertices]
        if len(ids) != len(set(ids)):
            raise ValueError("vertex IDs must be unique, including petiole controls")
        for v in all_vertices:
            if not v.organ_id or not v.vertex_id:
                raise ValueError("organ_id and vertex_id must be nonempty")
            if len(v.xyz) != 3 or len(v.uv) != 2 or not all(math.isfinite(x) for x in (*v.xyz, *v.uv)):
                raise ValueError("coordinates and UV must be finite and have correct dimensions")
            if not math.isfinite(v.confidence) or not 0 <= v.confidence <= 1:
                raise ValueError("confidence must be finite and in [0,1]")
            if not isinstance(v.level, int) or not 0 <= v.level <= self.level:
                raise ValueError("invalid vertex creation level")
            if v.provenance.kind not in SOURCE_KINDS:
                raise ValueError("unknown provenance kind")
            if v.provenance.kind == "observed" and not v.provenance.source_views:
                raise ValueError("observed vertices require source_views")
            if self.data_type == "real" and v.provenance.kind == "synthetic_control":
                raise ValueError("synthetic controls cannot be relabelled as real")
        fids = [f.face_id for f in self.faces]
        if len(fids) != len(set(fids)):
            raise ValueError("face IDs must be unique")
        for f in self.faces:
            if len(f.vertices) != 3 or len(set(f.vertices)) != 3:
                raise ValueError("faces must have three distinct indices")
            if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self.vertices) for i in f.vertices):
                raise ValueError("face index out of range")
            if any(self.vertices[i].organ_id != f.organ_id for i in f.vertices):
                raise ValueError("a face must not join different organ identities")
            if f.level != self.level:
                raise ValueError("only active leaf-level faces belong in a mesh")
            if norm(face_cross(self, f)) <= 1e-15:
                raise ValueError("degenerate triangle")
        for incident in self.edge_incidence().values():
            if len(incident) > 2:
                raise ValueError("non-manifold edge")
            if len(incident) == 2 and incident[0][1:] != incident[1][1:][::-1]:
                raise ValueError("inconsistent face orientation across shared edge")
        by_id = {v.vertex_id: v for v in all_vertices}
        for a, b in self.petiole_edges:
            if a not in by_id or b not in by_id or a == b:
                raise ValueError("invalid petiole edge")
            if by_id[a].organ_id != by_id[b].organ_id:
                raise ValueError("petiole edges must preserve organ identity")

    def to_dict(self) -> dict:
        self.validate()
        return {"schema_version": "pots-gs-mesh-1.0", **asdict(self)}

    def write_json(self, path: str | Path) -> None:
        """Exclusive creation protects prior experimental artifacts."""
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def write_obj(self, path: str | Path) -> None:
        """OBJ carries geometry and organ groups; use JSON for full provenance."""
        self.validate()
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(f"# POTS-GS data_type={self.data_type}; level={self.level}\n")
            handle.write("# Metadata, UV semantics and hierarchy are authoritative in companion JSON.\n")
            for v in self.vertices + self.petiole_controls:
                handle.write("v " + " ".join(format(x, ".17g") for x in v.xyz) + "\n")
            for v in self.vertices:
                handle.write("vt " + " ".join(format(x, ".17g") for x in v.uv) + "\n")
            organ = None
            for f in self.faces:
                if organ != f.organ_id:
                    organ = f.organ_id
                    # Retain the exact ID in JSON and encode whitespace in OBJ group names.
                    handle.write("g " + "".join(c if c.isalnum() or c in "_-" else "_" for c in organ) + "\n")
                handle.write("f " + " ".join(f"{i+1}/{i+1}" for i in f.vertices) + "\n")
            indices = {v.vertex_id: i + 1 for i, v in enumerate(self.vertices + self.petiole_controls)}
            for a, b in self.petiole_edges:
                handle.write(f"l {indices[a]} {indices[b]}\n")


def mesh_from_dict(data: dict) -> Mesh:
    if data.get("schema_version") != "pots-gs-mesh-1.0":
        raise ValueError("unsupported mesh schema")
    def vertex(item: dict) -> Vertex:
        p = item["provenance"]
        provenance = Provenance(p["kind"], tuple(p["source_views"]), tuple(p["contributors"]), p["note"])
        return Vertex(item["vertex_id"], item["organ_id"], tuple(item["xyz"]), tuple(item["uv"]),
                      item["level"], item["confidence"], provenance, item.get("label", ""))
    mesh = Mesh([vertex(v) for v in data["vertices"]],
                [Face(f["face_id"], f["organ_id"], tuple(f["vertices"]), f["level"], f["parent_face_id"]) for f in data["faces"]],
                data["level"], data["data_type"], data["coordinate_system"],
                {k: tuple(v) for k, v in data["lineage"].items()},
                [vertex(v) for v in data["petiole_controls"]], [tuple(e) for e in data["petiole_edges"]], data["metadata"])
    mesh.validate()
    return mesh


def create_leaf_mesh(controls: Sequence[Sequence[float]], organ_id: str, *,
                     data_type: str, provenance: Sequence[Provenance],
                     confidence: Sequence[float], petiole: Sequence[Sequence[float]] | None = None,
                     petiole_provenance: Sequence[Provenance] | None = None,
                     petiole_confidence: Sequence[float] | None = None) -> Mesh:
    """Build the taskbook topology. Every control's origin must be declared."""
    if len(controls) != 11 or len(provenance) != 11 or len(confidence) != 11:
        raise ValueError("exactly 11 surface controls, provenance records and confidences required")
    vertices = [Vertex(f"{organ_id}:v{i}", organ_id, tuple(map(float, xyz)), LEAF_UV[i], 0,
                       float(confidence[i]), provenance[i], LEAF_LABELS[i]) for i, xyz in enumerate(controls)]
    faces = [Face(f"{organ_id}:L0:f{i}", organ_id, face, 0) for i, face in enumerate(LEAF_FACES)]
    mesh = Mesh(vertices, faces, data_type=data_type,
                metadata={"topology_source": "taskbook 2026-09-05 section 7; fixed connection table verified",
                          "surface_controls": 11, "geometry_optimized": False,
                          "uv_semantics": "u in [0,1] longitudinal; v in [-1,1] lateral; endpoints v=0"})
    if petiole is not None:
        if len(petiole) != 2 or petiole_provenance is None or len(petiole_provenance) != 2 or petiole_confidence is None or len(petiole_confidence) != 2:
            raise ValueError("petiole requires separate P,J coordinates, provenance and confidence")
        mesh.petiole_controls = [Vertex(f"{organ_id}:{label}", organ_id, tuple(map(float, xyz)), (0., 0.), 0,
                                       float(conf), origin, label)
                                 for xyz, label, origin, conf in zip(petiole, ("P", "J"), petiole_provenance, petiole_confidence)]
        mesh.petiole_edges = [(vertices[0].vertex_id, mesh.petiole_controls[0].vertex_id),
                              (mesh.petiole_controls[0].vertex_id, mesh.petiole_controls[1].vertex_id)]
    elif petiole_provenance is not None or petiole_confidence is not None:
        raise ValueError("petiole metadata supplied without petiole coordinates")
    mesh.validate()
    return mesh


def combine_meshes(meshes: Iterable[Mesh]) -> Mesh:
    meshes = list(meshes)
    if not meshes:
        raise ValueError("at least one mesh is required")
    first = meshes[0]
    result = Mesh([], [], level=first.level, data_type=first.data_type,
                  coordinate_system=first.coordinate_system, metadata={"operation": "identity_preserving_concatenation"})
    seen_organs: set[str] = set()
    for mesh in meshes:
        mesh.validate()
        if (mesh.level, mesh.data_type, mesh.coordinate_system) != (result.level, result.data_type, result.coordinate_system):
            raise ValueError("combined meshes must share level, data type and coordinate system")
        organs = {v.organ_id for v in mesh.vertices}
        if seen_organs & organs:
            raise ValueError("duplicate organ IDs across input meshes")
        seen_organs.update(organs)
        offset = len(result.vertices)
        result.vertices.extend(mesh.vertices)
        result.faces.extend(Face(f.face_id, f.organ_id, tuple(i + offset for i in f.vertices), f.level, f.parent_face_id) for f in mesh.faces)
        result.lineage.update(mesh.lineage)
        result.petiole_controls.extend(mesh.petiole_controls)
        result.petiole_edges.extend(mesh.petiole_edges)
    result.validate()
    return result


def face_cross(mesh: Mesh, face: Face) -> Vec3:
    a, b, c = (mesh.vertices[i].xyz for i in face.vertices)
    return cross(sub(b, a), sub(c, a))


def face_normal(mesh: Mesh, face: Face) -> Vec3:
    value = face_cross(mesh, face)
    return scale(value, 1.0 / norm(value))


def mesh_area(mesh: Mesh) -> float:
    return math.fsum(norm(face_cross(mesh, f)) * .5 for f in mesh.faces)


def topology_summary(mesh: Mesh) -> dict:
    edges = mesh.edge_incidence()
    boundary = [k for k, inc in edges.items() if len(inc) == 1]
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(mesh.vertices))}
    boundary_adjacency: dict[int, set[int]] = {}
    for _, a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    for _, a, b in boundary:
        boundary_adjacency.setdefault(a, set()).add(b)
        boundary_adjacency.setdefault(b, set()).add(a)
    def components(graph: dict[int, set[int]]) -> int:
        pending = set(graph)
        count = 0
        while pending:
            count += 1
            stack = [pending.pop()]
            while stack:
                for nxt in graph[stack.pop()]:
                    if nxt in pending:
                        pending.remove(nxt)
                        stack.append(nxt)
        return count
    return {"vertices": len(mesh.vertices), "edges": len(edges), "faces": len(mesh.faces),
            "euler_characteristic": len(mesh.vertices) - len(edges) + len(mesh.faces),
            "connected_components": components(adjacency), "boundary_edges": len(boundary),
            "boundary_components": components(boundary_adjacency),
            "boundary_is_cycles": all(len(neighbours) == 2 for neighbours in boundary_adjacency.values())}


def point_triangle_distance(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    """Exact Euclidean distance to a nondegenerate closed triangle (region tests)."""
    ab, ac, ap = sub(b, a), sub(c, a), sub(point, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return norm(ap)
    bp = sub(point, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return norm(sub(point, add(a, scale(ab, d1 / (d1 - d3)))))
    cp = sub(point, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return norm(sub(point, add(a, scale(ac, d2 / (d2 - d6)))))
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return norm(sub(point, add(b, scale(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))))))
    return abs(dot(ap, cross(ab, ac))) / norm(cross(ab, ac))


def point_mesh_distance(point: Vec3, mesh: Mesh) -> float:
    return min(point_triangle_distance(point, *(mesh.vertices[i].xyz for i in f.vertices)) for f in mesh.faces)
