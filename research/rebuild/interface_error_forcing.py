"""Exact density-interface versus homogeneous heat-profile forcing split.

The prescribed coefficients are derived from the physical transmission
reference, not fitted. This module diagnoses the unchanged kernel operator.
"""
import numpy as np


def scaled_forcing(Q, Q0, mass, chi, chi_tau, tau, positions,
                   pair_i, pair_j, conductance, rho_left, rho_right, left=1., right=0.):
    x = np.asarray(positions); m = np.asarray(mass)
    contrast = (rho_left-rho_right)/(rho_left+rho_right)
    jump = left-right
    a = np.where(x < 0, rho_right, rho_left)/(rho_left+rho_right)*jump
    mean = (rho_left*left+rho_right*right)/(rho_left+rho_right)
    reference = mean-a*chi
    # Exact per-row homogeneous continuation of each phase's reference.
    bulk = -a*(2*tau*(Q0@chi)-chi_tau)
    I = np.asarray(pair_i); J = np.asarray(pair_j); g = np.asarray(conductance)
    cross = (x[I] < 0) & (x[J] > 0)
    i, j, weights = I[cross], J[cross], g[cross]
    value = tau*jump*contrast*weights*(chi[i]+chi[j])
    interface = (np.bincount(j, weights=value, minlength=len(x))
                - np.bincount(i, weights=value, minlength=len(x)))/m
    direct = 2*tau*(Q@reference)+a*chi_tau
    diagnostic = {'algebra_closure_max_abs': float(np.max(abs(interface+bulk-direct))),
                  'interface_mass_forcing': float(m@interface),
                  'bulk_mass_forcing': float(m@bulk)}
    return np.column_stack([interface, bulk]), reference, diagnostic
