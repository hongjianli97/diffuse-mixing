"""Raw mass-similar exponential action and independent positive uniformization."""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import expm_multiply


@dataclass
class Actions:
    mass: np.ndarray
    edges: np.ndarray
    conductance: np.ndarray
    symmetric_generator: csr_matrix
    uniformization_matrix: csr_matrix
    Lambda: int
    max_degree: int
    trace: float
    diagnostics: dict


def operators(mass, edges, conductances):
    m = np.asarray(mass, dtype=float)
    ij = np.asarray(edges)
    gg = tuple(F(v) for v in conductances)
    if m.ndim != 1 or len(m) < 2 or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError('finite positive original masses required')
    if ij.ndim != 2 or ij.shape[1] != 2 or ij.dtype.kind not in 'iu' or len(ij) != len(gg):
        raise ValueError('matching integer shared-face edges required')
    n = len(m)
    if np.any(ij < 0) or np.any(ij >= n) or np.any(ij[:, 0] == ij[:, 1]) or any(g < 0 for g in gg):
        raise ValueError('valid shared edges and nonnegative exact conductances required')
    exact_m = tuple(F(float(v)) for v in m)
    exact_degree = [F(0)] * n
    counts = np.zeros(n, dtype=int)
    for (i, j), g in zip(ij, gg):
        exact_degree[i] += g
        exact_degree[j] += g
        counts[i] += 1
        counts[j] += 1
    Lambda = math.ceil(max(d / a for d, a in zip(exact_degree, exact_m)))
    gf = np.array([float(g) for g in gg])
    if not np.isfinite(gf).all() or any(g > 0 and x == 0 for g, x in zip(gg, gf)):
        raise ArithmeticError('nonrepresentable conductance; no clipping')
    i, j = ij.T
    degree = np.bincount(np.r_[i, j], weights=np.r_[gf, gf], minlength=n)
    root = np.sqrt(m)
    off = gf / (root[i] * root[j])
    diagonal = -degree / m
    A = coo_matrix((np.r_[off, off, diagonal],
                    (np.r_[i, j, np.arange(n)], np.r_[j, i, np.arange(n)])), shape=(n, n)).tocsr()
    if Lambda:
        pd = tuple(1 - d / (Lambda * a) for d, a in zip(exact_degree, exact_m))
        if min(pd) < 0:
            raise ArithmeticError('negative exact uniformization diagonal')
        pi = np.array([float(g / (Lambda * exact_m[a])) for (a, _), g in zip(ij, gg)])
        pj = np.array([float(g / (Lambda * exact_m[b])) for (_, b), g in zip(ij, gg)])
        P = coo_matrix((np.r_[pi, pj, list(map(float, pd))],
                       (np.r_[i, j, np.arange(n)], np.r_[j, i, np.arange(n)])), shape=(n, n)).tocsr()
    else:
        pd = tuple(F(1) for _ in m)
        P = eye(n, format='csr')
    trace = float(A.diagonal().sum())
    diagnostics = {
        'nodes': n, 'shared_faces': len(gg), 'maximum_degree': int(counts.max()),
        'Lambda_exact_integer': Lambda, 'minimum_exact_uniformization_diagonal': str(min(pd)),
        'symmetric_generator_trace_raw': trace,
        'symmetric_constant_residual_Linf': float(np.max(abs(A @ root))),
        'uniformization_row_sum_residual_Linf': float(np.max(abs(P @ np.ones(n) - 1))),
        'uniformization_invariant_mass_residual_Linf': float(np.max(abs(P.T @ m - m))),
        'uniformization_minimum_stored_coefficient': float(P.data.min()),
        'primary_diagonal_is_raw_sum_of_float_face_coefficients': True,
        'uniformization_diagonal_from_exact_local_rates': True,
        'rows_weights_or_masses_normalized': False, 'zero_eigenvalues_snapped': False,
        'dense_full_generator_or_propagator_constructed': False,
    }
    return Actions(m.copy(), ij.copy(), gf, A, P, Lambda, int(counts.max()), trace, diagnostics)


def poisson_plan(z, budget=F(1, 10**16)):
    z, budget = F(z), F(budget)
    if z < 0 or not 0 < budget < 1:
        raise ValueError('nonnegative exact duration and tail budget in(0,1) required')
    if not z:
        return 0, F(0)
    M, term = 0, z
    while M < 100000:
        ratio = z / (M + 2)
        if ratio < 1:
            bound = term / (1 - ratio)  # exp(-z)<=1; no fitted tail or weight scaling
            if bound <= budget:
                return M, bound
        M += 1
        term *= z / (M + 1)
    raise ArithmeticError('unsupported uniformization degree; no budget relaxation')


def checked_state(action, state, dt):
    c = np.asarray(state, dtype=float)
    duration = F(dt)
    if c.shape != action.mass.shape or not np.isfinite(c).all() or duration <= 0:
        raise ValueError('matching finite state and positive duration required')
    return c, duration


def exponential(action, state, dt):
    c, duration = checked_state(action, state, dt)
    if not action.Lambda:
        return c.copy(), c.copy()
    root = np.sqrt(action.mass)
    path = expm_multiply(action.symmetric_generator, root * c,
                         start=0., stop=float(duration), num=3, endpoint=True, traceA=action.trace)
    return path[-1] / root, path[1] / root


def uniformization(action, state, dt, budget=F(1, 10**16)):
    c, duration = checked_state(action, state, dt)
    z = action.Lambda * duration
    M, bound = poisson_plan(z, budget)
    if not z:
        return c.copy(), np.array([1.]), {'M': 0, 'z_exact': '0', 'tail_upper_exact': '0',
            'raw_weight_sum': 1., 'weights_or_state_normalized': False}
    weight = math.exp(-float(z))
    if not 0 < weight <= 1:
        raise ArithmeticError('unsupported raw Poisson seed; no rescaling')
    v = c.copy()
    result = weight * v
    weights = [weight]
    for k in range(1, M + 1):
        v = action.uniformization_matrix @ v
        weight *= float(z) / k
        weights.append(weight)
        result += weight * v
    info = {'M': M, 'z_exact': str(z), 'tail_upper_exact': str(bound),
            'raw_weight_sum': math.fsum(weights), 'weights_or_state_normalized': False}
    return result, np.array(weights), info


def arithmetic_allowance(action, initial, dt, degree):
    """A predeclared machine-scale indicator, not a certified exponential error bound."""
    c, duration = checked_state(action, initial, dt)
    return float(64 * np.finfo(float).eps * (action.max_degree + 1) * (degree + 1)
                 * (1 + float(action.Lambda * duration))
                 * max(1., math.sqrt(float(action.mass.sum()))) * max(1., float(np.max(abs(c)))))


def metrics(m, c, c0):
    # Same definition and evaluation order as the saved original shear experiment.
    mu = float(m @ c / m.sum())
    mu0 = float(m @ c0 / m.sum())
    E = float(m @ ((c - mu)**2))
    E0 = float(m @ ((c0 - mu0)**2))
    if E0 <= 0:
        raise ValueError('nonconstant original initial mass average required for I/H')
    return {'mean': mu, 'variance': E, 'I': E / E0, 'H': 1 - E / E0}


def dissipation(action, state):
    i, j = action.edges.T
    return float(action.conductance @ ((state[j] - state[i])**2))
