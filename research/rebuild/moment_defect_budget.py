"""Fixed-grid diagnostic minimax second-moment defect; physical targets fixed.

First moments remain exact constraints. The objective is the largest absolute
xx/xy/yy defect divided by 2D in the already-fixed physical coordinate frame.
This norm is not rotation invariant and is never an acceptance threshold.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack


def budget_problem(A,b,mass,interior_cells,D,length_unit=.5):
    A=csr_matrix(A);b=np.asarray(b,dtype=float);m=np.asarray(mass,dtype=float);inside=np.asarray(interior_cells,dtype=int)
    if b.ndim!=1 or len(b)!=len(inside)*5 or A.shape[0]!=len(b) or not np.isfinite(A.data).all() or not np.isfinite(b).all():
        raise ValueError('finite five-moment system matching the interior cells required')
    if m.ndim!=1 or np.any(m<=0) or not np.isfinite(m).all() or np.any(inside<0) or np.any(inside>=len(m)) or len(inside)==0:
        raise ValueError('positive fixed mass vector and valid interior cells required')
    if not np.isfinite(D) or D<=0 or not np.isfinite(length_unit) or length_unit<=0:
        raise ValueError('positive physical D and length unit required for this relative diagnostic')
    rows=np.arange(len(b)).reshape(-1,5);erows=rows[:,:2].ravel();srows=rows[:,2:].ravel()
    E=A[erows];e=b[erows];S=A[srows];s=b[srows];scale=np.repeat(2*D*m[inside],3)
    eq_units=np.repeat(length_unit*m[inside],2)
    return {'E':E,'e':e,'S':S,'s':s,'scale':scale,'eq_units':eq_units,
        'first_rows':erows,'second_rows':srows,'mass':m,'interior_cells':inside,'D':D,'length_unit':length_unit}


def solve_budget(problem):
    E,e,S,s,c,u=[problem[k] for k in ('E','e','S','s','scale','eq_units')];n=E.shape[1]
    E_lp=hstack((E.multiply((1/u)[:,None]),csr_matrix((len(e),1))),format='csr')
    H=vstack((hstack((S.multiply((1/c)[:,None]),-np.ones((len(s),1)))),
              hstack((-S.multiply((1/c)[:,None]),-np.ones((len(s),1))))),format='csr')
    rhs=np.r_[s/c,-s/c];objective=np.r_[np.zeros(n),1.]
    result=linprog(objective,A_eq=E_lp,b_eq=e/u,A_ub=H,b_ub=rhs,bounds=(0,None),method='highs')
    report={'solver':'HiGHS default tolerances, one fixed minimax problem','status':int(result.status),'message':result.message,
        'solver_reports_feasible':bool(result.success),'conductance_variables':n,'budget_variables':1,
        'exact_first_moment_equations':len(e),'second_moment_bounds':len(rhs),
        'objective':'min max_i,k abs(Gamma_graph_ik-Gamma_physical_ik)/(2D), k=xx,xy,yy; fixed coordinate frame',
        'physical_targets_changed':False,'physical_acceptance':False,'fluid_orc_enabled':False}
    arrays={}
    if result.success:
        g=result.x[:-1].copy();defect=(S@g-s)/c;drift=(E@g-e)/u
        report.update({'epsilon_solver':float(result.x[-1]),'epsilon_from_unmodified_conductances':float(np.max(abs(defect))),
            'minimum_raw_conductance':float(g.min()),'positive_raw_conductances':int(np.count_nonzero(g>0)),
            'max_first_moment_residual_after_unit_conversion':float(np.max(abs(drift))),
            'rates_used_for_evolution':False,'accepted_backend':False})
        arrays={'raw_conductances':g,'raw_epsilon':result.x[-1],
            'first_residual_mass_scaled':E@g-e,'second_residual_mass_scaled':S@g-s,
            'second_component_defect_relative_to_2D':defect,
            'dual_equality_marginals':result.eqlin.marginals,'dual_inequality_marginals':result.ineqlin.marginals}
    return report,arrays
