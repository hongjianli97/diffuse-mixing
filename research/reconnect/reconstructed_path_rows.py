"""Frozen target rows from the predeclared nearest-point density path rule.

Only point inputs enter construction. Fractions preserve the binary input and
the existing exact path integral; B_h is the unchanged floating kernel. Rows
are an open subdomain, not a closed two-node graph or a production backend.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import lcm, sqrt
import numpy as np

from periodic_density_path import _fraction, _identity, _pair
from endpoint_density_path import localized_inverse_density_path
from positive_kernel_2d import slope_over_radius


@dataclass(frozen=True)
class PathEdge:
    left_id: int
    right_id: int
    displacement: tuple
    distance_squared: F
    path: object
    retained_ids: tuple
    kernel_B: float
    conductance: F


@dataclass(frozen=True)
class PathRow:
    target_id: int
    mass: F
    neighbors: tuple  # (other ID, index in shared edge tuple)


@dataclass(frozen=True)
class PathRows:
    ids: tuple
    mass: tuple
    rho: tuple
    quadrature_area: tuple
    periods: tuple
    diffusivity: F
    h: F
    edges: tuple
    rows: tuple
    exact_integer_broad_phase: bool


class _Geometry:
    """Exact support and start-ball broad phase, with Fraction fallback.

    An integer lattice is only an arithmetic representation of input binary
    coordinates. It does not identify material geometry. Integer overflow is
    excluded before using NumPy; otherwise all comparisons use Fractions.
    """
    def __init__(self, points, box):
        self.points, self.box = points, box
        scale = lcm(*(value.denominator for point in points for value in point),
                    *(value.denominator for value in box))
        box_integer = tuple(int(value * scale) for value in box)
        self.integer = 2 * max(box_integer)**2 <= np.iinfo(np.int64).max
        self.scale = scale
        if self.integer:
            self.x = np.array([[int(v * scale) for v in p] for p in points], dtype=np.int64)
            self.box_integer = np.array(box_integer, dtype=np.int64)

    def displacement(self, i, j):
        delta = []
        for a, b, length in zip(self.points[i], self.points[j], self.box):
            d = b - a
            if d > length / 2:
                d -= length
            elif d < -length / 2:
                d += length
            delta.append(d)
        return tuple(delta)

    def inside_ball(self, center, radius_squared):
        if self.integer:
            difference = np.abs(self.x - self.x[center])
            difference = np.minimum(difference, self.box_integer - difference)
            squared = np.sum(difference * difference, axis=1, dtype=np.int64)
            # Comparison against the floor is exactly equivalent for integer distances.
            limit = radius_squared * self.scale**2
            return tuple(map(int, np.flatnonzero(squared <= limit.numerator // limit.denominator)))
        return tuple(i for i in range(len(self.points))
                     if sum((v*v for v in self.displacement(center, i)), F(0)) <= radius_squared)


def build_reconstructed_path_rows(positions, ids, mass, rho, D, h, periods, target_ids,
                                  thickness=1, progress=None):
    """No c0, response, true faces, true volumes, F/C or gradient argument.

    Validate every global input before exact start-ball preselection. Existing
    endpoint-ball localization then retains the intersection of the closed
    endpoint balls and calls the unchanged exact inverse-density primitive.
    """
    box = _pair(periods, 'periods')
    points = tuple(_pair(p, 'position') for p in positions)
    identities = tuple(_identity(v) for v in ids)
    masses = tuple(_fraction(v, 'mass') for v in mass)
    density = tuple(_fraction(v, 'density') for v in rho)
    diffusivity, width = _fraction(D, 'D'), _fraction(h, 'h')
    b = _fraction(thickness, 'thickness')
    targets = tuple(_identity(v) for v in target_ids)
    if any(v <= 0 for v in box) or b != 1:
        raise ValueError('positive periods and unit physical thickness required')
    if diffusivity < 0 or width <= 0 or 4*width >= min(box):
        raise ValueError('D>=0, h>0 and support below half each period required')
    if not points or not (len(points) == len(identities) == len(masses) == len(density)):
        raise ValueError('matching nonempty point inputs required')
    if len(set(identities)) != len(identities) or len(set(targets)) != len(targets) or not targets:
        raise ValueError('unique fixed input and target IDs required')
    if any(v <= 0 for v in masses + density):
        raise ValueError('all global masses and densities must be positive')
    coincident = {}
    for point, value in zip(points, density):
        if any(v < 0 or v >= length for v, length in zip(point, box)):
            raise ValueError('already wrapped coordinates required; no repair')
        if point in coincident and coincident[point] != value:
            raise ValueError('coincident conflicting densities')
        coincident[point] = value
    lookup = {v: i for i, v in enumerate(identities)}
    if any(v not in lookup for v in targets):
        raise ValueError('target IDs must be present')
    q = tuple(m/(r*b) for m, r in zip(masses, density))
    geometry = _Geometry(points, box)
    pairs = set()
    for target in targets:
        i = lookup[target]
        for j in geometry.inside_ball(i, (2*width)**2):
            if j != i:
                pairs.add(tuple(sorted((target, identities[j]))))
    edges = []
    for count, (left, right) in enumerate(sorted(pairs), 1):
        i, j = lookup[left], lookup[right]
        delta = geometry.displacement(i, j)
        radius2 = sum((v*v for v in delta), F(0))
        radius = sqrt(float(radius2))
        Bh = float(slope_over_radius(radius, float(width)))
        if diffusivity == 0:
            path, retained, g = None, (), F(0)
        else:
            # Every possible path owner lies in the start ball, by the already
            # proved endpoint-ball lemma. Broad phase has exact comparisons.
            candidates = geometry.inside_ball(i, radius2)
            localized = localized_inverse_density_path(
                [points[k] for k in candidates], [identities[k] for k in candidates],
                [density[k] for k in candidates], box, left, right)
            path = localized.path
            retained = localized.selection.retained_ids
            g = 2*b*q[i]*q[j]*diffusivity/path.mean_inverse_density*F(Bh)
        if Bh < 0 or g < 0 or not np.isfinite(Bh):
            raise ArithmeticError('invalid raw conductance; no clipping')
        edges.append(PathEdge(left, right, delta, radius2, path, retained, Bh, g))
        if progress is not None and (count % 50 == 0 or count == len(pairs)):
            progress(count, len(pairs))
    rows = []
    for target in targets:
        neighbors = tuple((edge.right_id if edge.left_id == target else edge.left_id, index)
                          for index, edge in enumerate(edges)
                          if target in (edge.left_id, edge.right_id))
        rows.append(PathRow(target, masses[lookup[target]], neighbors))
    return PathRows(identities, masses, density, q, box, diffusivity, width,
                    tuple(edges), tuple(rows), geometry.integer)


def apply_reconstructed_rows(stencil, cbar_by_id):
    """Apply already fixed coefficients to material mass averages, exactly."""
    states = {key: _fraction(value, 'cbar') for key, value in cbar_by_id.items()}
    return {row.target_id: sum((stencil.edges[index].conductance *
            (states[other] - states[row.target_id]) for other, index in row.neighbors), F(0))/row.mass
            for row in stencil.rows}
