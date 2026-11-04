"""Sparse positive 1D exchange with one global kernel width.

The kernel, physical coefficient and Neumann reflection are unchanged from
positive_kernel_1d. Only constant-width pair enumeration/storage are new.
No moment correction, clipping or normalization is performed.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from positive_kernel_1d import slope_over_radius


def build_constant_exchange(positions, mass, volumes, rho, D, h, bounds):
    x, m, V, rho = [np.asarray(v, dtype=float) for v in (positions, mass, volumes, rho)]
    if x.ndim != 1 or len(x) < 2 or any(v.shape != x.shape for v in (m, V, rho)):
        raise ValueError('compatible node vectors required')
    if any(not np.isfinite(v).all() for v in (x, m, V, rho)):
        raise ValueError('finite node data required')
    if np.any(np.diff(x) <= 0) or any(np.any(v <= 0) for v in (m, V, rho)):
        raise ValueError('sorted distinct nodes and positive physical measures required')
    if not np.isscalar(h) or not np.isscalar(D):
        raise ValueError('this backend requires scalar h and D')
    a, b = map(float, bounds)
    if not np.isfinite([a, b, D, h]).all() or D < 0 or h <= 0 or b <= a:
        raise ValueError('valid domain, nonnegative D and positive h required')
    if x[0] <= a or x[-1] >= b or 2*h >= b-a:
        raise ValueError('interior nodes and support shorter than domain required')
    if not np.allclose(m, rho*V, rtol=1e-10, atol=0):
        raise ValueError('fixed mass disagrees with oracle rho and true volume')

    # A reflected distance is >= the direct distance for interior nodes, so
    # enumerating all direct-support pairs includes every image-support pair.
    ii, jj, gg = [], [], []
    first = np.zeros(len(x)); second = np.zeros(len(x))
    for i in range(len(x)-1):
        end = np.searchsorted(x, x[i]+2*h, side='left')
        j = np.arange(i+1, end)
        if not len(j):
            continue
        r = x[j]-x[i]
        H = 2*D*rho[i]*rho[j]/(rho[i]+rho[j])
        direct = 2*V[i]*V[j]*H*slope_over_radius(r, h)
        g = direct + 2*V[i]*V[j]*H*(
            slope_over_radius(x[i]+x[j]-2*a, h)
            + slope_over_radius(2*b-x[i]-x[j], h))
        if not np.isfinite(g).all() or np.any(g < 0):
            raise RuntimeError('invalid conductance, no repair allowed')
        # Omit exact zero entries only; negative values have already raised.
        present = g > 0
        ii.append(np.full(np.count_nonzero(present), i, dtype=int))
        jj.append(j[present]); gg.append(g[present])
        first[i] += np.sum(direct*r)
        first[j] -= direct*r
        second[i] += np.sum(direct*r*r)/2
        second[j] += direct*r*r/2
    I = np.concatenate(ii) if ii else np.array([], dtype=int)
    J = np.concatenate(jj) if jj else np.array([], dtype=int)
    g = np.concatenate(gg) if gg else np.array([], dtype=float)
    G = coo_matrix((np.r_[g, g], (np.r_[I, J], np.r_[J, I])), shape=(len(x), len(x))).tocsr()
    L = diags(np.asarray(G.sum(axis=1)).ravel())-G
    first /= m; second /= m
    pure = (x-a > 2*h) & (b-x > 2*h) & (np.abs(x) > 2*h)
    diag = {'nodes': len(x), 'edges': len(g),
            'components': int(connected_components(G, directed=False, return_labels=False)),
            'symmetry_residual': float(abs(G-G.T).max()),
            'cross_contact_edges': int(np.count_nonzero((x[I] < 0) & (x[J] > 0))),
            'moment_diagnostic_nodes': int(pure.sum()),
            'interior_first_moment_max_abs': float(np.max(abs(first[pure]))) if pure.any() else None,
            'interior_diffusivity_ratio_min': float(np.min(second[pure]/D)) if pure.any() and D > 0 else None,
            'interior_diffusivity_ratio_max': float(np.max(second[pure]/D)) if pure.any() and D > 0 else None,
            'max_relative_m_vs_rho_volume': float(np.max(abs(m-rho*V)/m)),
            'clipping_or_normalization': False}
    arrays = {'pair_i': I, 'pair_j': J, 'conductance': g,
              'moment_first': first, 'moment_diffusivity': second,
              'moment_interior_mask': pure}
    return L.tocsr(), arrays, diag
