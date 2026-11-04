"""Row-level mass-average moment cone certificates, never a diffusion backend.

Linear programming is used only to find a small feasibility basis. Independent
Fraction arithmetic certifies that basis or an explicit separating functional
for the stored floating-point data. No state/mass/rate clipping, fitted F/C,
threshold changes, globally conservative operator or propagation is involved.
"""
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog


def _fraction(x):
    return Fraction.from_float(float(x))


def _record(x):
    return {'numerator':str(x.numerator),'denominator':str(x.denominator),'float':float(x)}


def require_bulk_context(touches_boundary,context):
    if context not in ('actual_interior','explicit_virtual_transverse_extension'):
        raise ValueError('bulk moments require actual interior or an explicit virtual extension; no silent boundary substitution')
    if touches_boundary and context=='actual_interior':
        raise ValueError('boundary cell cannot use bare bulk moment equations without boundary action')


def moment_features(displacements,covariance_difference):
    d=np.asarray(displacements,dtype=float);s=np.asarray(covariance_difference,dtype=float)
    if d.ndim!=2 or d.shape[1]!=2 or s.shape!=(len(d),2,2) or len(d)==0:
        raise ValueError('nonempty 2D displacements and matching covariance differences required')
    if not np.isfinite(d).all() or not np.isfinite(s).all() or not np.array_equal(s,s.swapaxes(1,2)):
        raise ValueError('finite symmetric covariance differences required')
    h=d[:,:,None]*d[:,None,:]+s
    return np.stack((d[:,0],d[:,1],h[:,0,0],h[:,0,1],h[:,1,1]))


def exact_basis_witness(A,b,active):
    """Exact rational solve on the unmodified LP basis; does not alter LP output."""
    A=np.asarray(A);b=np.asarray(b);active=np.asarray(active,dtype=int)
    if len(active)>len(b):return {'verified':False,'reason':'more active columns than equations'}
    aug=[[_fraction(A[i,j]) for j in active]+[_fraction(b[i])] for i in range(len(b))]
    pivots=[];row=0
    for col in range(len(active)):
        p=next((i for i in range(row,len(b)) if aug[i][col]!=0),None)
        if p is None:return {'verified':False,'reason':'rank-deficient proposed basis'}
        aug[row],aug[p]=aug[p],aug[row];scale=aug[row][col];aug[row]=[v/scale for v in aug[row]]
        for i in range(len(b)):
            if i==row:continue
            factor=aug[i][col]
            aug[i]=[a-factor*z for a,z in zip(aug[i],aug[row])]
        pivots.append(row);row+=1
    if any(any(v!=0 for v in r[:-1]) or r[-1]!=0 for r in aug[row:]):
        return {'verified':False,'reason':'remaining exact equations inconsistent'}
    values=[aug[r][-1] for r in pivots]
    if any(v<0 for v in values):return {'verified':False,'reason':'exact basis has a negative coefficient'}
    equalities=all(sum((_fraction(A[i,j])*v for j,v in zip(active,values)),Fraction(0))==_fraction(b[i]) for i in range(len(b)))
    return {'verified':equalities,'active_columns':active.tolist(),'exact_nonnegative_rates':[_record(v) for v in values],
            'meaning':'exact cone witness for stored binary floating-point inputs only; not a global operator or continuum interval proof'}


def cross_moment_separator(A,b):
    """Exact lower or upper cross/transverse cone separator when available."""
    A=np.asarray(A);b=np.asarray(b);transverse=[j for j in range(A.shape[1]) if A[4,j]>0]
    if not transverse or any(A[4,j]<0 for j in range(A.shape[1])):
        return {'verified':False,'reason':'cross-ratio separator not available for this covariance support'}
    if any(A[4,j]==0 and A[3,j]!=0 for j in range(A.shape[1])):
        return {'verified':False,'reason':'zero-transverse column carries cross moment'}
    ratios=[_fraction(A[3,j])/_fraction(A[4,j]) for j in transverse];lower=min(ratios);upper=max(ratios)
    if b[4]<=0:return {'verified':False,'reason':'positive transverse target required'}
    target=_fraction(b[3])/_fraction(b[4]);y=None
    if target<lower:y=[Fraction(0),Fraction(0),Fraction(0),Fraction(1),-lower]
    elif target>upper:y=[Fraction(0),Fraction(0),Fraction(0),Fraction(-1),upper]
    if y is None:return {'verified':False,'reason':'cross-ratio necessary condition passes; no separating claim',
        'cross_ratio_interval':[float(lower),float(upper)],'target_cross_ratio':float(target)}
    slacks=[sum((y[i]*_fraction(A[i,j]) for i in range(5)),Fraction(0)) for j in range(A.shape[1])]
    gap=sum((y[i]*_fraction(b[i]) for i in range(5)),Fraction(0))
    return {'verified':bool(min(slacks)>=0 and gap<0),'functional':[_record(v) for v in y],
        'minimum_exact_column_slack':_record(min(slacks)),'exact_target_dot_functional':_record(gap),
        'cross_ratio_interval':[float(lower),float(upper)],'target_cross_ratio':float(target),
        'meaning':'A^T y >= 0, b^T y < 0 in exact rational arithmetic for stored input: infeasible nonnegative cone'}


def transverse_drift_separator(A,b):
    """Necessary drift/second-moment compatibility, without solving any rates."""
    A=np.asarray(A);b=np.asarray(b);nonzero=[j for j in range(A.shape[1]) if A[1,j]!=0]
    if not nonzero or any(A[4,j]<0 for j in range(A.shape[1])) or b[4]<=0:
        return {'verified':False,'reason':'nonnegative transverse moment support and positive target required'}
    h=min(_fraction(A[4,j])/abs(_fraction(A[1,j])) for j in nonzero)
    if h<=0:return {'verified':False,'reason':'no positive drift/second-moment lower bound on this support'}
    sign=1 if b[1]>=0 else -1;y=[Fraction(0),-sign*h,Fraction(0),Fraction(0),Fraction(1)]
    slacks=[sum((y[i]*_fraction(A[i,j]) for i in range(5)),Fraction(0)) for j in range(A.shape[1])]
    gap=sum((y[i]*_fraction(b[i]) for i in range(5)),Fraction(0))
    result={'verified':bool(min(slacks)>=0 and gap<0),'functional':[_record(v) for v in y],
        'minimum_exact_column_slack':_record(min(slacks)),'exact_target_dot_functional':_record(gap),
        'minimum_yy_moment_per_abs_y_displacement_m':_record(h),
        'necessary_condition':'Gamma_yy >= h*abs(beta_y); failure rules out this nonnegative moment cone',
        'meaning':'exact rational certificate for stored data, not a boundary or nonlinear-method impossibility claim'}
    if b[1]!=0:result['upper_admissible_h_from_target_m']=_record(_fraction(b[4])/abs(_fraction(b[1])))
    return result


def solve_moment_cone(A,b,length_unit=.5):
    A=np.asarray(A,dtype=float);b=np.asarray(b,dtype=float)
    if A.ndim!=2 or A.shape[0]!=5 or A.shape[1]==0 or b.shape!=(5,) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('finite five-moment system required')
    if not np.isfinite(length_unit) or length_unit<=0:raise ValueError('positive length unit required')
    units=np.array([length_unit,length_unit,length_unit**2,length_unit**2,length_unit**2])
    # Only change the units of equations, not physical data, rates or tolerances.
    result=linprog(np.zeros(A.shape[1]),A_eq=A/units[:,None],b_eq=b/units,bounds=(0,None),method='highs')
    report={'solver':'scipy.optimize.linprog(highs), default tolerances, zero objective','status':int(result.status),
      'message':result.message,'solver_reports_feasible':bool(result.success),'row_length_unit_m':length_unit,
      'physical_acceptance':False,'globally_conservative_operator_constructed':False,'rates_used_for_evolution':False,
      'cross_ratio_separator':cross_moment_separator(A,b)}
    rates=None
    if result.success:
        rates=result.x.copy();active=np.flatnonzero(rates!=0)
        report.update({'minimum_raw_rate':float(rates.min()),'active_columns':active.tolist(),
            'maximum_raw_residual_by_unit_m_per_s_or_m2_per_s':(A@rates-b).tolist(),
            'maximum_equation_residual_after_unit_conversion':float(np.max(abs((A@rates-b)/units))),
            'exact_basis_certificate':exact_basis_witness(A,b,active)})
    return report,rates
