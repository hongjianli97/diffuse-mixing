"""Limited ordinary central-resistance FV rows under a material Cartesian prior.

This is an exact-arithmetic reference adapter, not a general point-cloud
backend. Points must already be centers of static Cartesian material cells,
and each cell must have constant density. These are model assumptions; a
tensor point pattern alone does not identify material geometry.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from periodic_density_path import _fraction, _identity, _pair


@dataclass(frozen=True)
class SharedFVEdge:
    left_id: int
    right_id: int
    axis: int
    conductance: F


@dataclass(frozen=True)
class MaterialFVRow:
    target_id: int
    mass: F
    neighbors: tuple  # (direction, neighbor ID, shared conductance)


@dataclass(frozen=True)
class CartesianMaterialRows:
    point_count: int
    shape: tuple
    periods: tuple
    spacing: tuple
    thickness: F
    diffusivity: F
    edges: tuple
    rows: tuple
    model_prior: str = 'static Cartesian material cells, supplied points at centers, constant density per cell'


def build_cartesian_material_rows(positions, ids, mass, rho, periods, D, target_ids, thickness=1):
    """Validate the complete supplied grid, construct only requested FV rows.

    All numeric values are interpreted exactly, including binary floats.
    Exact mass consistency is required; inputs are never snapped or repaired.
    At least three cells per axis avoid two distinct faces to one neighbor.
    """
    box = _pair(periods, 'periods')
    diffusion = _fraction(D, 'D')
    b = _fraction(thickness, 'thickness')
    if any(value <= 0 for value in box) or b <= 0 or diffusion < 0:
        raise ValueError('positive periods/thickness and nonnegative diffusivity required')
    points = tuple(_pair(point, 'position') for point in positions)
    identities = tuple(_identity(value) for value in ids)
    masses = tuple(_fraction(value, 'mass') for value in mass)
    densities = tuple(_fraction(value, 'density') for value in rho)
    targets = tuple(_identity(value) for value in target_ids)
    if not points or any(len(values) != len(points) for values in [identities, masses, densities]):
        raise ValueError('nonempty matching positions, IDs, masses and densities required')
    if len(set(identities)) != len(identities) or len(set(points)) != len(points):
        raise ValueError('unique IDs and point positions required')
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(identities):
        raise ValueError('distinct requested target IDs must be present')
    if any(value <= 0 for value in masses + densities):
        raise ValueError('positive masses and densities required')
    if any(value < 0 or value >= length for point in points for value, length in zip(point, box)):
        raise ValueError('point coordinates must already be wrapped')
    axes = tuple(tuple(sorted({point[k] for point in points})) for k in range(2))
    shape = tuple(len(axis) for axis in axes)
    if min(shape) < 3 or shape[0] * shape[1] != len(points):
        raise ValueError('complete tensor grid with at least three cells per axis required')
    spacing = tuple(box[k] / shape[k] for k in range(2))
    for axis, length, step in zip(axes, box, spacing):
        gaps = tuple(axis[i + 1] - axis[i] for i in range(len(axis) - 1)) + (axis[0] + length - axis[-1],)
        if any(value != step for value in gaps):
            raise ValueError('uniform cyclic spacing required; no coordinate snapping')
    area = spacing[0] * spacing[1]
    if any(m != r * b * area for m, r in zip(masses, densities)):
        raise ValueError('original m must equal rho*b*dx*dy under this declared material prior')
    axis_indices = tuple({value: index for index, value in enumerate(axis)} for axis in axes)
    grid = {(axis_indices[0][point[0]], axis_indices[1][point[1]]): identity
            for point, identity in zip(points, identities)}
    lookup = {identity: index for index, identity in enumerate(identities)}
    edge_values, rows = {}, []
    for target in targets:
        index = lookup[target]
        cell = tuple(axis_indices[k][points[index][k]] for k in range(2))
        neighbors = []
        for axis, sign, direction in [(0, -1, 'x_minus'), (0, 1, 'x_plus'),
                                       (1, -1, 'y_minus'), (1, 1, 'y_plus')]:
            other_cell = list(cell)
            other_cell[axis] = (other_cell[axis] + sign) % shape[axis]
            other = grid[tuple(other_cell)]
            other_index = lookup[other]
            if diffusion == 0:
                g = F(0)
            else:
                resistance = spacing[axis] / (2 * densities[index] * diffusion)
                resistance += spacing[axis] / (2 * densities[other_index] * diffusion)
                g = b * spacing[1 - axis] / resistance
            key = (*sorted((target, other)), axis)
            if key in edge_values and edge_values[key] != g:
                raise RuntimeError('shared FV edge has inconsistent conductance')
            edge_values[key] = g
            neighbors.append((direction, other, g))
        rows.append(MaterialFVRow(target, masses[index], tuple(neighbors)))
    edges = tuple(SharedFVEdge(left, right, axis, value)
                  for (left, right, axis), value in sorted(edge_values.items()))
    return CartesianMaterialRows(len(points), shape, box, spacing, b, diffusion, edges, tuple(rows))


def apply_material_rows(stencil, material_means_by_id):
    """Exact local edge-flux action on supplied mass means, not point samples."""
    required = {row.target_id for row in stencil.rows}
    required.update(other for row in stencil.rows for _, other, _ in row.neighbors)
    if not required <= set(material_means_by_id):
        raise ValueError('all requested target/neighbor material means are required')
    values = {identity: _fraction(material_means_by_id[identity], 'material mean') for identity in required}
    return {row.target_id: sum((g * (values[other] - values[row.target_id])
                               for _, other, g in row.neighbors), F(0)) / row.mass
            for row in stencil.rows}
