"""Continuous unit-flux transmission layer, piecewise-constant Galerkin solve.

The affine background and its residual are integrated as continuous fields.
Pair-cell weights use positive polynomial-exact Gaussian quadrature. This is
an analysis cell problem, not a replacement of the saved physical backend.
"""
import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from positive_kernel_1d import slope_over_radius


def gauss_integral(function, a, b, order=5):
    if b <= a:
        return 0.
    z, w = np.polynomial.legendre.leggauss(order)
    r = (a+b)/2+(b-a)*z/2
    return float((b-a)/2*np.dot(w, function(r)))


def integrated_pair_weight(k, spacing):
    """Integral of B_1(y-x) over two distinct uniform cells k apart."""
    d = spacing; center = k*d
    left = gauss_integral(lambda r: (r-(center-d))*slope_over_radius(r, 1.),
                          max(0., center-d), min(2., center), order=4)
    right = gauss_integral(lambda r: ((center+d)-r)*slope_over_radius(r, 1.),
                           center, min(2., center+d), order=4)
    return left+right


def affine_weighted_residual(u, contrast):
    """Unit-flux weighted affine residual on a half-line, at distance u."""
    u = np.asarray(u, dtype=float)
    result = np.zeros_like(u)
    for index in np.ndindex(u.shape):
        z = float(u[index])
        if z < 2:
            result[index] = -2*contrast*gauss_integral(
                lambda r: slope_over_radius(r, 1.)*(r-2*z), z, 2., order=4)
    return result


def solve_inner(rho_left=2., rho_right=1., half_window=8., spacing=1/32):
    if not np.isfinite([rho_left, rho_right, half_window, spacing]).all():
        raise ValueError('finite cell-problem parameters required')
    if min(rho_left, rho_right, spacing) <= 0 or half_window < 4:
        raise ValueError('positive densities/spacing and window >=4 required')
    if any(abs(v/spacing-round(v/spacing)) > 1e-10 for v in (2., half_window)):
        raise ValueError('support and matching window must align to cell faces')
    extent = half_window+2
    n = int(round(2*extent/spacing)); faces = np.linspace(-extent, extent, n+1)
    x = (faces[:-1]+faces[1:])/2
    rho = np.where(x < 0, rho_left, rho_right)
    interior = abs(x) < half_window; ids = np.flatnonzero(interior)
    left_collar = x <= -half_window; right_collar = x >= half_window
    I, J, weights = [], [], []
    for k in range(1, int(round(2/spacing))+1):
        B = integrated_pair_weight(k, spacing)
        i = np.arange(n-k); j = i+k
        H = 2*rho[i]*rho[j]/(rho[i]+rho[j])
        g = 2*H*B
        if not np.isfinite(g).all() or np.any(g < 0):
            raise RuntimeError('invalid positive cell-pair quadrature')
        I.append(i); J.append(j); weights.append(g)
    I = np.concatenate(I); J = np.concatenate(J); g = np.concatenate(weights)
    G = coo_matrix((np.r_[g, g], (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    L = diags(np.asarray(G.sum(axis=1)).ravel())-G
    beta = (rho_left-rho_right)/(rho_left+rho_right)
    source = np.zeros(n)
    # The residual is a polynomial of degree 8 on each half of its support;
    # 5-point outer and 4-point inner Gauss rules integrate it exactly.
    for i in ids:
        lo = min(abs(faces[i]), abs(faces[i+1])); hi = max(abs(faces[i]), abs(faces[i+1]))
        if lo < 2:
            source[i] = gauss_integral(lambda u: affine_weighted_residual(u, beta), lo, min(hi, 2.), order=5)
    C = np.column_stack([np.asarray(L[interior][:, left_collar].sum(axis=1)).ravel(),
                         np.asarray(L[interior][:, right_collar].sum(axis=1)).ravel()])
    cross = (x[I] < 0) & (x[J] > 0)
    f = np.bincount(I[cross], weights=g[cross], minlength=n)-np.bincount(J[cross], weights=g[cross], minlength=n)
    # Unknowns: interior corrector and two far constants. The affine
    # background has exactly unit continuous flux; its correction has zero
    # integrated cross flux. b_L+b_R=0 only fixes the constant nullspace.
    A = bmat([[L[interior][:, interior], csr_matrix(C)],
              [csr_matrix((1, len(ids))), csr_matrix([[1., 1.]])],
              [csr_matrix(f[interior][None, :]), csr_matrix((1, 2))]], format='csr')
    rhs = np.r_[source[interior], 0., 0.]
    sol = spsolve(A, rhs)
    u = np.where(x < 0, sol[-2], sol[-1]); u[interior] = sol[:-2]
    b_left, b_right = map(float, sol[-2:]); kappa = b_right-b_left
    far = np.where(x < 0, b_left, b_right)
    psi = u-far
    residual = L@u-source
    diag = {'rho_left': rho_left, 'rho_right': rho_right, 'half_window': half_window,
            'spacing': spacing, 'interior_cells': int(interior.sum()),
            'kappa': kappa, 'b_left_gauge': b_left, 'b_right_gauge': b_right,
            'gauge_residual': abs(b_left+b_right),
            'max_stationary_integrated_residual': float(np.max(abs(residual[interior]))),
            'zero_correction_flux_residual': abs(float(f@u)),
            'max_augmented_linear_residual': float(np.max(abs(A@sol-rhs))),
            'affine_residual_half_mass': float(source[(x > 0)&interior].sum()),
            'minimum_pair_weight': float(g.min()),
            'local_tail_max_abs_last_two_units': float(np.max(abs(psi[interior & (abs(x)>half_window-2)]))),
            'constant_shift_only_gauge': True, 'no_weight_or_moment_normalization': True}
    arrays = {'faces': faces, 'positions': x, 'interior_mask': interior, 'rho': rho,
              'corrector': u, 'localized_corrector': psi, 'affine_weighted_residual_integral': source,
              'far_constant': far, 'pair_i': I, 'pair_j': J, 'conductance': g}
    return arrays, diag
