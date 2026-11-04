"""Stable finite-interval heat-step averages and derivatives in tau=sqrt(t).

chi solves constant-D Neumann heat flow on [-L,L] with initial sign(x).
The physical density-contact reference is mean-a_phase*chi. No kernel
coefficients enter this independent image reference.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import erfcx


@dataclass(frozen=True)
class HeatStepImages:
    D: float = .02
    half_length: float = 1.
    images: int = 2

    def __post_init__(self):
        if not np.isfinite([self.D, self.half_length]).all() or min(self.D, self.half_length) <= 0:
            raise ValueError('positive finite D and half length required')
        if not isinstance(self.images, int) or self.images < 1:
            raise ValueError('positive integer image count required')

    def average_and_tau_derivative(self, faces, tau):
        faces = np.asarray(faces, dtype=float)
        L = self.half_length
        if faces.ndim != 1 or len(faces) < 3 or not np.isfinite(faces).all() or np.any(np.diff(faces) <= 0):
            raise ValueError('finite increasing faces required')
        if not np.isfinite(tau) or tau < 0 or faces[0] < -L or faces[-1] > L:
            raise ValueError('valid time and physical domain required')
        if np.any((faces[:-1] < 0) & (faces[1:] > 0)):
            raise ValueError('this reference helper requires interface-aligned cells')
        widths = np.diff(faces); mids = (faces[:-1]+faces[1:])/2
        if tau == 0:
            chi = np.sign(mids)
            gaussian = (faces == 0).astype(float)
        else:
            w = 2*np.sqrt(self.D)*tau
            def primitive(z):
                q = z/w
                return w*np.exp(-q*q)*(q*erfcx(q)-1/np.sqrt(np.pi))
            lo = np.minimum(abs(faces[:-1]), abs(faces[1:]))
            hi = np.maximum(abs(faces[:-1]), abs(faces[1:]))
            integral = np.zeros_like(widths)
            z = abs(faces); gaussian = np.zeros_like(faces)
            for n in range(self.images):
                integral += (-1.)**n*(primitive(2*n*L+hi)-primitive(2*n*L+lo)
                           + primitive(2*(n+1)*L-lo)-primitive(2*(n+1)*L-hi))
                gaussian += (-1.)**n*(np.exp(-((2*n*L+z)/w)**2)
                              - np.exp(-((2*(n+1)*L-z)/w)**2))
            chi = np.sign(mids)*(1-integral/widths)
        # This is 2*tau*(-D chi_x), finite at tau=0, including both endpoints.
        scaled_flux = -2*np.sqrt(self.D/np.pi)*gaussian
        chi_tau = (scaled_flux[:-1]-scaled_flux[1:])/widths
        return chi, chi_tau

    def log_concentration_tail_bound(self, t):
        if not np.isfinite(t) or t <= 0:
            raise ValueError('positive finite time required')
        a = self.half_length**2/(self.D*t); M = self.images
        return float(np.log(2.)-a*M*M-np.log(-np.expm1(-2*a*M)))
