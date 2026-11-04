"""Certified endpoint-ball localization of the immutable density-path primitive.

All distances and comparisons are exact Fractions. The closed-ball radius is
the path length, never a user parameter. No mass, scalar, material geometry,
diffusivity, graph or conductance is involved.
"""
from dataclasses import dataclass
from fractions import Fraction

from periodic_density_path import _fraction, _identity, _pair, inverse_density_path


@dataclass(frozen=True)
class EndpointBallSelection:
    points: tuple
    ids: tuple
    periods: tuple
    start_id: int
    end_id: int
    displacement: tuple
    radius_squared: Fraction
    retained_indices: tuple
    distance2_to_start: tuple
    distance2_to_end: tuple

    @property
    def retained_ids(self):
        return tuple(self.ids[index] for index in self.retained_indices)


@dataclass(frozen=True)
class LocalizedDensityPath:
    selection: EndpointBallSelection
    path: object


def _distance_squared(left, right, periods):
    total = Fraction(0)
    for a, b, length in zip(left, right, periods):
        difference = abs(a - b)
        distance = min(difference, length - difference)
        total += distance**2
    return total


def endpoint_ball_selection(positions, ids, periods, start_id, end_id):
    """Keep all sites in both endpoint closed balls of radius path length."""
    box = _pair(periods, 'periods')
    if any(value <= 0 for value in box):
        raise ValueError('periods must be positive')
    points = tuple(_pair(point, 'position') for point in positions)
    identities = tuple(_identity(value) for value in ids)
    if not points or len(points) != len(identities):
        raise ValueError('nonempty positions and IDs of equal length required')
    if len(set(identities)) != len(identities):
        raise ValueError('fixed IDs must be unique')
    if any(value < 0 or value >= length for point in points for value, length in zip(point, box)):
        raise ValueError('coordinates must already be wrapped into [0, period)')
    start_id, end_id = _identity(start_id), _identity(end_id)
    lookup = dict(zip(identities, points))
    if start_id not in lookup or end_id not in lookup:
        raise ValueError('both endpoint IDs must be present')
    start, end = lookup[start_id], lookup[end_id]
    delta = []
    for a, b, length in zip(start, end, box):
        value = b - a
        if abs(value) == length / 2:
            raise ValueError('half-period displacement has no unique minimum image')
        if value > length / 2:
            value -= length
        elif value < -length / 2:
            value += length
        delta.append(value)
    delta = tuple(delta)
    radius2 = sum((value**2 for value in delta), Fraction(0))
    if 4 * radius2 >= min(box)**2:
        raise ValueError('segment length must be strictly below half the shortest period')
    start_distances = tuple(_distance_squared(point, start, box) for point in points)
    end_distances = tuple(_distance_squared(point, end, box) for point in points)
    keep = tuple(index for index, (a, b) in enumerate(zip(start_distances, end_distances))
                 if a <= radius2 and b <= radius2)
    if start_id not in {identities[index] for index in keep} or end_id not in {identities[index] for index in keep}:
        raise RuntimeError('endpoint balls lost an endpoint')
    return EndpointBallSelection(points, identities, box, start_id, end_id, delta,
                                 radius2, keep, start_distances, end_distances)


def localized_inverse_density_path(positions, ids, rho, periods, start_id, end_id):
    """Validate the complete density field, then call the unchanged primitive."""
    selection = endpoint_ball_selection(positions, ids, periods, start_id, end_id)
    values = tuple(_fraction(value, 'density') for value in rho)
    if len(values) != len(selection.points) or any(value <= 0 for value in values):
        raise ValueError('matching positive point densities required for the full input')
    coincident = {}
    for point, value in zip(selection.points, values):
        if point in coincident and coincident[point] != value:
            raise ValueError('coincident points with conflicting densities are not a density field')
        coincident[point] = value
    indices = selection.retained_indices
    path = inverse_density_path(
        [selection.points[index] for index in indices],
        [selection.ids[index] for index in indices], [values[index] for index in indices],
        selection.periods, selection.start_id, selection.end_id)
    return LocalizedDensityPath(selection, path)
