"""Stream ordinary FV shared faces under an explicitly supplied material prior.

Exact arithmetic, no adjacency/transport matrix. Uniform point spacing alone
does not establish this material geometry. Means are fixed-cell mass means.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from periodic_density_path import _fraction, _identity, _pair


MATERIAL_PRIOR = 'static Cartesian material cells, supplied points at centers, constant density per cell'


@dataclass(frozen=True)
class StreamAction:
    ids: tuple
    shape: tuple
    spacing: tuple
    mass_rates: tuple
    rates: tuple
    row_absolute_sum_bounds: tuple
    face_count: int
    face_counts_by_axis: tuple
    minimum_g: F
    maximum_g: F
    dissipation: F
    exact_mass_balance: F
    exact_energy_residual: F
    selected_faces: tuple


def stream_action(positions, ids, mass, rho, periods, D, material_means, *, model_prior,
                  thickness=1, selected_edges=()):
    if model_prior != MATERIAL_PRIOR:
        raise ValueError('the explicit constant-density Cartesian material-cell prior is required')
    box, b, diffusion = _pair(periods, 'periods'), _fraction(thickness, 'thickness'), _fraction(D, 'D')
    if any(x <= 0 for x in box) or b <= 0 or diffusion < 0:
        raise ValueError('positive geometry and nonnegative D required')
    points = tuple(_pair(p, 'point') for p in positions)
    identities = tuple(_identity(x) for x in ids)
    masses = tuple(_fraction(x, 'mass') for x in mass)
    densities = tuple(_fraction(x, 'rho') for x in rho)
    means = tuple(_fraction(x, 'material mean') for x in material_means)
    n = len(points)
    if not n or any(len(x) != n for x in [identities, masses, densities, means]):
        raise ValueError('nonempty matching inputs required')
    if len(set(points)) != n or len(set(identities)) != n:
        raise ValueError('unique positions and IDs required')
    if any(x <= 0 for x in masses + densities):
        raise ValueError('positive original masses and densities required')
    if any(x < 0 or x >= length for p in points for x, length in zip(p, box)):
        raise ValueError('coordinates must already be wrapped')
    axes = tuple(tuple(sorted({p[k] for p in points})) for k in range(2))
    shape = tuple(map(len, axes))
    if min(shape) < 3 or shape[0]*shape[1] != n:
        raise ValueError('complete tensor grid with at least three cells per axis required')
    spacing = tuple(box[k]/shape[k] for k in range(2))
    for axis, length, step in zip(axes, box, spacing):
        gaps = tuple(axis[i+1]-axis[i] for i in range(len(axis)-1)) + (axis[0]+length-axis[-1],)
        if any(gap != step for gap in gaps):
            raise ValueError('uniform cyclic spacing required; no snapping')
    if any(m != r*b*spacing[0]*spacing[1] for m,r in zip(masses,densities)):
        raise ValueError('mass must exactly match declared material cells; no repair')
    index = {p:i for i,p in enumerate(points)}
    grid = {(i,j):index[(x,y)] for j,y in enumerate(axes[1]) for i,x in enumerate(axes[0])}
    selected = {(*sorted((int(a),int(c))),int(axis)) for a,c,axis in selected_edges}
    found = {}
    mass_rate, gsum = [F(0)]*n, [F(0)]*n
    dissipation, counts, gmin, gmax = F(0), [0,0], None, F(0)
    for j in range(shape[1]):
        for i in range(shape[0]):
            left = grid[(i,j)]
            for axis, neighbor in [(0,((i+1)%shape[0],j)), (1,(i,(j+1)%shape[1]))]:
                right = grid[neighbor]
                g = F(0) if diffusion == 0 else b*spacing[1-axis] / (
                    spacing[axis]/(2*densities[left]*diffusion) + spacing[axis]/(2*densities[right]*diffusion))
                difference = means[right]-means[left]
                flux = g*difference
                mass_rate[left] += flux
                mass_rate[right] -= flux
                gsum[left] += g
                gsum[right] += g
                dissipation += flux*difference
                counts[axis] += 1
                gmin, gmax = g if gmin is None else min(gmin,g), max(gmax,g)
                key = (*sorted((identities[left],identities[right])),axis)
                if key in selected:
                    if key in found:raise RuntimeError('duplicate selected shared face')
                    found[key] = (identities[left],identities[right],axis,g,flux)
    if set(found) != selected:
        raise ValueError('selected edge absent from this periodic material grid')
    mass_balance = sum(mass_rate,F(0))
    energy = sum((c*q for c,q in zip(means,mass_rate)),F(0)) + dissipation
    assert mass_balance == energy == 0
    assert counts == [n,n]
    return StreamAction(identities,shape,spacing,tuple(mass_rate),
        tuple(q/m for q,m in zip(mass_rate,masses)),tuple(2*g/m for g,m in zip(gsum,masses)),
        sum(counts),tuple(counts),gmin,gmax,dissipation,mass_balance,energy,
        tuple(found[key] for key in sorted(found)))
