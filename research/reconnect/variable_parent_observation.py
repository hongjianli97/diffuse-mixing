"""Analysis-only projection onto saved parent masses, preserving roundoff.

Variable-density analytic integrals can differ by floating rounding between
parent and child formulas. Do not force bitwise equality or rescale children.
The existing 1e-10 algebra tolerance only rejects incompatible input records;
it is not a physical accuracy threshold.
"""
import numpy as np
from trajectory_chain import ALGEBRA_TOL


def parent_mass_audit(fine_mass,parent_mass):
    mf=np.asarray(fine_mass,dtype=float);mp=np.asarray(parent_mass,dtype=float);N=int(np.sqrt(len(mp)));nf=2*N
    if N*N!=len(mp) or mf.shape!=(nf*nf,) or any(not np.isfinite(a).all() or np.any(a<=0) for a in [mf,mp]):raise ValueError('finite positive nested 2x2 masses required')
    child_sum=mf.reshape(N,2,N,2).sum(axis=(1,3)).ravel();relative=(child_sum-mp)/mp
    maximum=float(np.max(abs(relative)))
    if maximum>ALGEBRA_TOL:raise ValueError('child masses incompatible with saved parent masses at existing algebra tolerance; no repair')
    audit={'maximum_relative_child_sum_discrepancy':maximum,'maximum_absolute_child_sum_discrepancy_kg':float(np.max(abs(child_sum-mp))),
        'bitwise_equal_child_sums_and_parent_mass':bool(np.array_equal(child_sum,mp)),
        'non_bitwise_equal_parent_count':int(np.count_nonzero(child_sum!=mp)),
        'existing_input_compatibility_algebra_tolerance':ALGEBRA_TOL,'child_mass_renormalized':False,'saved_parent_mass_modified':False,
        'constant_projection_max_difference_from_one':float(np.max(abs(child_sum/mp-1)))}
    return child_sum,relative,audit


def observe_parent(values,fine_mass,parent_mass):
    child_sum,relative,audit=parent_mass_audit(fine_mass,parent_mass)
    v=np.asarray(values,dtype=float);mf=np.asarray(fine_mass);mp=np.asarray(parent_mass);N=int(np.sqrt(len(mp)))
    if v.shape!=mf.shape or not np.isfinite(v).all():raise ValueError('matching finite fine observations required')
    return (mf*v).reshape(N,2,N,2).sum(axis=(1,3)).ravel()/mp
