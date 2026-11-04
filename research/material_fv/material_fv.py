"""Experimental 1D material-cell FV; not recovered ltnmix 0.8.0.

Unknowns are material-cell mass averages. Caller supplies fixed masses and
true moving faces; trajectory points are not treated as geometric centers.
No-flux endpoints, unit cross section, scalar spatially constant D.
"""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded


def vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite vector')
    return a


def center_resistance(faces, rho, D, t):
    """g = 1 / integral_center_to_center dx/(rho D).

    Centers are geometric midpoints of the supplied material faces. Split
    integration at the shared face. Smooth positive oracle density required;
    discontinuous coefficients must be handled by a separately specified rule.
    Returned quadrature errors are errors in resistance, not PDE error bounds.
    """
    faces = vector(faces, 'faces')
    if len(faces) < 2 or np.any(np.diff(faces) <= 0):
        raise ValueError('faces must strictly increase')
    if not np.isfinite(D) or D < 0 or not np.isfinite(t):
        raise ValueError('finite t and nonnegative finite D required')
    centers = (faces[:-1] + faces[1:]) / 2
    g, errors = np.zeros(len(centers)-1), np.zeros(len(centers)-1)
    if D == 0:
        return g, errors

    def resistance(x):
        density = float(rho(x, t))
        if not np.isfinite(density) or density <= 0:
            raise ValueError('oracle density must be finite positive')
        return 1 / (density * D)

    for i in range(len(g)):
        left, el = quad(resistance, centers[i], faces[i+1], epsabs=1e-12, epsrel=1e-11)
        right, er = quad(resistance, faces[i+1], centers[i+1], epsabs=1e-12, epsrel=1e-11)
        g[i], errors[i] = 1 / (left + right), el + er
    return g, errors


def step(c, mass, g, dt):
    """Backward Euler (M+dt L)c_next=M c, without clipping or rescaling."""
    c, mass, g = vector(c, 'c'), vector(mass, 'mass'), vector(g, 'g')
    if len(c) == 0 or mass.shape != c.shape or len(g) != len(c)-1:
        raise ValueError('inconsistent cell/face dimensions')
    if np.any(mass <= 0) or np.any(g < 0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive masses/dt and nonnegative conductances required')
    a = np.zeros((3, len(c)))
    a[1] = mass
    a[1, :-1] += dt*g
    a[1, 1:] += dt*g
    a[0, 1:] = -dt*g
    a[2, :-1] = -dt*g
    return solve_banded((1, 1), a, mass*c)


def propagate(c0, mass, times, faces_at, rho, D):
    """Left-end geometry; fixed masses throughout; first-order time method."""
    times = vector(times, 'times')
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('times must strictly increase')
    c0, mass = vector(c0, 'c0'), vector(mass, 'mass')
    if mass.shape != c0.shape or np.any(mass <= 0):
        raise ValueError('fixed positive masses required')
    history = [c0.copy()]
    for t, dt in zip(times[:-1], np.diff(times)):
        faces = vector(faces_at(t), 'faces')
        if len(faces) != len(mass)+1:
            raise ValueError('moving faces must retain cell identity/count')
        g, _ = center_resistance(faces, rho, D, t)
        history.append(step(history[-1], mass, g, dt))
    return np.asarray(history)
