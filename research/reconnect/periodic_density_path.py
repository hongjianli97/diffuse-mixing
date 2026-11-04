"""Exact reference primitive for a declared nearest-point density closure.

Input floats are interpreted as their exact binary values. Output intervals
describe density interpolation, never material cells or flux faces. This small
standard-library prototype constructs neither conductances nor a graph.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from numbers import Integral, Rational, Real


def _fraction(value, name):
    if isinstance(value, bool):
        raise TypeError(f'{name}: booleans are not physical numbers')
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, Real) and hasattr(value, 'as_integer_ratio'):
        try:
            numerator, denominator = value.as_integer_ratio()
        except (ValueError, OverflowError) as error:
            raise ValueError(f'{name}: finite number required') from error
        return Fraction(int(numerator), int(denominator))
    raise TypeError(f'{name}: rational or finite binary floating-point number required')


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError('fixed IDs must be integers, excluding booleans')
    return int(value)


def _pair(values, name):
    try:
        items = tuple(values)
    except TypeError as error:
        raise TypeError(f'{name}: two coordinates required') from error
    if len(items) != 2:
        raise ValueError(f'{name}: two coordinates required')
    return tuple(_fraction(value, name) for value in items)


@dataclass(frozen=True)
class DensityInterval:
    """An almost-everywhere owner on the interior of a positive-length interval."""
    start: Fraction
    stop: Fraction
    source_id: int
    density: Fraction


@dataclass(frozen=True)
class DensityPath:
    start_id: int
    end_id: int
    displacement: tuple
    intervals: tuple
    breakpoint_owners: tuple
    mean_inverse_density: Fraction


@dataclass(frozen=True)
class _Line:
    slope: Fraction
    intercept: Fraction
    source_id: int
    density: Fraction
    image: tuple


def _lower_envelope(lines):
    # Minimum-envelope slopes decrease as the path parameter increases.
    ordered = sorted(lines, key=lambda line: (
        -line.slope, line.intercept, line.source_id, line.image))
    distinct = []
    for line in ordered:
        if not distinct or line.slope != distinct[-1].slope:
            distinct.append(line)
    hull, starts = [], []
    for line in distinct:
        crossing = None
        while hull:
            crossing = (line.intercept - hull[-1].intercept) / (hull[-1].slope - line.slope)
            if len(hull) == 1 or crossing > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            crossing = None  # minus infinity, not a floating-point sentinel
        hull.append(line)
        starts.append(crossing)
    return hull, starts


def inverse_density_path(positions, ids, rho, periods, start_id, end_id):
    """Return integral_0^1 1/rho_hat(x_start+theta*d) dtheta exactly.

    positions: wrapped 2D coordinates; ids: unique integer material identities;
    rho: positive point densities; periods: two positive lengths. Endpoint IDs
    select a unique shortest segment of length strictly below min(periods)/2.
    A zero-length path is permitted. A half-period ambiguity is rejected.

    Intervals are positive-length, almost-everywhere ownership records. At all
    envelope breakpoints, separate point ownership follows the smallest fixed
    ID; even a node that wins only at a single parameter is recorded there.
    """
    box = _pair(periods, 'periods')
    if any(length <= 0 for length in box):
        raise ValueError('periods must be positive')
    points = tuple(_pair(point, 'position') for point in positions)
    identities = tuple(_identity(value) for value in ids)
    density = tuple(_fraction(value, 'density') for value in rho)
    if not points or len(points) != len(identities) or len(points) != len(density):
        raise ValueError('nonempty positions, IDs and densities of equal length required')
    if len(set(identities)) != len(identities):
        raise ValueError('fixed IDs must be unique')
    if any(value <= 0 for value in density):
        raise ValueError('point densities must be positive')
    coincident = {}
    for point, value in zip(points, density):
        if any(coordinate < 0 or coordinate >= length for coordinate, length in zip(point, box)):
            raise ValueError('coordinates must already be wrapped into [0, period)')
        if point in coincident and coincident[point] != value:
            raise ValueError('coincident points with conflicting densities are not a density field')
        coincident[point] = value
    lookup = dict(zip(identities, points))
    start_id, end_id = _identity(start_id), _identity(end_id)
    if start_id not in lookup or end_id not in lookup:
        raise ValueError('both endpoint IDs must be present')
    origin, end = lookup[start_id], lookup[end_id]
    delta = []
    for left, right, length in zip(origin, end, box):
        displacement = right - left
        if abs(displacement) == length / 2:
            raise ValueError('half-period displacement has no unique minimum image')
        if displacement > length / 2:
            displacement -= length
        elif displacement < -length / 2:
            displacement += length
        delta.append(displacement)
    delta = tuple(delta)
    if 4 * sum(value**2 for value in delta) >= min(box)**2:
        raise ValueError('segment length must be strictly below half the shortest period')

    # The lifted short segment lies inside (-L/2,3L/2) in each coordinate.
    # For wrapped samples, the nearest image is among these nine images.
    lines = []
    for point, identity, value in zip(points, identities, density):
        for image in product((-1, 0, 1), repeat=2):
            offset = tuple(origin[k] - point[k] - image[k] * box[k] for k in range(2))
            lines.append(_Line(
                2 * sum(delta[k] * offset[k] for k in range(2)),
                sum(component**2 for component in offset), identity, value, image))
    hull, starts = _lower_envelope(lines)
    intervals = []
    events = {Fraction(0), Fraction(1)}
    for index, line in enumerate(hull):
        left = starts[index]
        right = starts[index + 1] if index + 1 < len(starts) else None
        if left is not None and left >= 1 or right is not None and right <= 0:
            continue
        # Exact intersection with the integration domain, no tolerance repair.
        left = Fraction(0) if left is None or left < 0 else left
        right = Fraction(1) if right is None or right > 1 else right
        if right <= left:
            continue
        events.update((left, right))
        if intervals and intervals[-1].source_id == line.source_id:
            previous = intervals[-1]
            if previous.stop != left:
                raise RuntimeError('noncontiguous envelope ownership')
            intervals[-1] = DensityInterval(previous.start, right, line.source_id, line.density)
        else:
            intervals.append(DensityInterval(left, right, line.source_id, line.density))
    if not intervals or intervals[0].start != 0 or intervals[-1].stop != 1:
        raise RuntimeError('envelope does not cover the integration domain')
    if any(left.stop != right.start for left, right in zip(intervals, intervals[1:])):
        raise RuntimeError('envelope has a gap or overlap')
    if sum((part.stop - part.start for part in intervals), Fraction(0)) != 1:
        raise RuntimeError('envelope lengths do not sum to one')
    mean = sum(((part.stop - part.start) / part.density for part in intervals), Fraction(0))
    if not 1 / max(density) <= mean <= 1 / min(density):
        raise RuntimeError('inverse-density mean violates the input range')
    owners = []
    for parameter in sorted(events):
        winner = min(lines, key=lambda line: (
            line.slope * parameter + line.intercept, line.source_id, line.image))
        owners.append((parameter, winner.source_id))
    return DensityPath(start_id, end_id, delta, tuple(intervals), tuple(owners), mean)
