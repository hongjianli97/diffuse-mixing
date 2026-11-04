"""Fixed-density Laguerre theory and exact symbolic witnesses; no mesh optimizer."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import shutil
import sys

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-orthogonal-mean-closure'
THEORY=ROOT/'research/reconnect/LAGUERRE_MASS_INTERIORITY_REVIEW.md'
ZERO_FIELDS=['new_software_checks','previous_checks_rerun','new_candidate_weights','new_kernel_or_path_evaluations',
    'new_operator_actions','new_quadrature_evaluations','new_Fourier_transforms_or_eigensolves',
    'new_candidate_graphs','new_numerical_or_flow_cases','new_candidate_transient_runs','new_PDE_reference_solves',
    'numerical_partition_optimizer_runs']
BLOCKED=['numpy','scipy','orthogonal_mean_closure','saved_exchange_time','point_fixed_time_study',
    'positive_kernel_2d','oracle_path_reference','cartesian_fv_modes','contact_spectral_reference']


class Polynomial:
    """Small exact commutative algebra for scalar formula checks only."""
    def __init__(self,x=0):
        if isinstance(x,Polynomial):self.terms=dict(x.terms)
        elif isinstance(x,dict):self.terms={k:F(v) for k,v in x.items() if v}
        else:self.terms={():F(x)} if x else {}
    def __add__(self,x):
        d=dict(self.terms)
        for k,v in Polynomial(x).terms.items():d[k]=d.get(k,F(0))+v
        return Polynomial(d)
    __radd__=__add__
    def __neg__(self):return Polynomial({k:-v for k,v in self.terms.items()})
    def __sub__(self,x):return self+-Polynomial(x)
    def __rsub__(self,x):return Polynomial(x)+-self
    def __mul__(self,x):
        d={}
        for a,v in self.terms.items():
            for b,w in Polynomial(x).terms.items():
                k=tuple(sorted(a+b));d[k]=d.get(k,F(0))+v*w
        return Polynomial(d)
    __rmul__=__mul__


def var(s):return Polynomial({(s,):1})
def zero(p):assert not Polynomial(p).terms
def square(p):return p*p


def certificates():
    groups=[]
    def done(name,scope):groups.append({'name':name,'exact_algebra_passed':True,'scope':scope})
    a,T,M,m0,cell0=map(var,['a','target_total','source_total','m0','cell0'])
    zero((a*T-a*M)-a*(T-M))
    zero((1-m0)-(1-cell0)-(cell0-m0))
    assert (a*(T-M)).terms
    done('dual_gauge_and_stationarity','gauge mismatch and two-label directional derivative; analytical coercivity/differentiability proof stays in report')
    d,wi,wj=map(var,['d','wi','wj'])
    ni=square(d)+wi-wj;nj=square(d)+wj-wi
    zero(ni+nj-2*square(d));zero(ni-square(d)-(wi-wj))
    done('oriented_power_face','two signed distance numerators and their sum; no mesh constructed')
    x=var('x')
    zero(square(x+F(3,4))-square(x)-(F(3,2)*x+F(9,16)))
    zero(square(x-F(1,4))-square(x)-(-F(1,2)*x+F(1,16)))
    assert F(3,2)*F(-1,2)+F(9,16)==-F(3,16)
    assert F(3,2)*F(-1,4)+F(9,16)==F(3,16)
    assert -F(1,2)*F(1,2)+F(1,16)==-F(3,16)
    done('periodic_squared_cost_branches','exact quadratic cancellation, turning values and branch slopes')
    delta=var('delta');left=F(2,3)*delta-F(3,8);right=F(1,8)-2*delta
    zero(F(3,2)*left+F(9,16)-delta);zero(-F(1,2)*right+F(1,16)-delta)
    zero(right-left-(F(1,2)-F(8,3)*delta))
    target=F(1,4);dw=F(3,32);L=F(2,3)*dw-F(3,8);R=F(1,8)-2*dw
    assert -F(3,16)<dw<F(3,16) and -F(1,2)<L<-F(1,4)<R<0
    assert (L,R)==(-F(5,16),-F(1,16)) and R-L==target
    seed_cost_other=F(1,16)-dw;ti=(F(1,16)-dw)/(2*F(1,4));tj=F(1,4)-ti
    assert seed_cost_other==-F(1,32)<0 and ti==-F(1,16)<0 and tj==F(5,16)>0
    done('exact_mass_seed_exclusion','two-site periodic strip algebra, masses1/4 and3/4, seed score and negative oriented face distance; not a physical flow case')
    mass_at_interior_boundary=F(1,2)-F(8,3)*F(1,16)
    assert mass_at_interior_boundary==F(1,3) and mass_at_interior_boundary-target==F(1,12)
    assert F(1,2)-F(8,3)*(-F(1,16))==F(2,3)
    assert -F(8,3)<0 and (1-target)-(1-mass_at_interior_boundary)==F(1,12)
    done('interiority_mass_residual','closed seed inclusion interval and minimum1/12 mass residual; no PDE error or universal positive-method bound')
    di,dj,Gi,Gj=map(var,['di','dj','Gi','Gj'])
    zero(di*Gj+dj*Gi-F(1,2)*(di+dj)*(Gi+Gj)-F(1,2)*(di-dj)*(Gj-Gi))
    assert F(1,3)/2+F(2,3)/5==F(3,10)
    assert F(1,2)/2+F(1,2)/5==F(7,20)
    assert F(3,10)-F(7,20)==-F(1,20)
    done('nonmidpoint_layer_resistance','cleared resistance difference and exact layered scalar substitution; no candidate face conductance evaluated')
    r,s,z,t,v,w=map(var,['r','s','z','t','v','w']);mass=r+s
    raw=mass*(r*(square(z)+v)+s*(square(t)+w))-square(r*z+s*t)
    zero(raw-mass*(r*v+s*w)-r*s*square(z-t))
    y,q,c1,c2=map(var,['y','q','c1','c2'])
    raw_cross=mass*(r*(z*y+c1)+s*(t*q+c2))-(r*z+s*t)*(r*y+s*q)
    zero(raw_cross-mass*(r*c1+s*c2)-r*s*(z-t)*(y-q))
    done('fixed_density_mixture_moments','mass-weighted within/between variance and mixed covariance for a cell cut by fixed density')
    thick,rhoi,rhoj,A,B=map(var,['b','rhoi','rhoj','A','B'])
    actual=thick*(rhoi*A+rhoj*B);relabel=thick*rhoi*(A+B)
    zero(actual-relabel-thick*(rhoj-rhoi)*B);assert (actual-relabel).terms
    done('fixed_field_not_cell_relabel','changing auxiliary labels does not change fixed density; cut-cell mass correction retained')
    assert len(groups)==8
    for name in BLOCKED:assert name not in sys.modules,name
    return {'verified_exact_algebra_groups':8,'groups':groups,'standard_library_only':True,
        'not_automatic_optimal_transport_or_PDE_proof':True,'no_numerical_partition_or_graph':True,
        'witness':{'cost':'periodic squared distance','density':'1','target_mass_0':str(target),
            'target_mass_1':str(1-target),'weight_difference':str(dw),'cell_0_left':str(L),'cell_0_right':str(R),
            'seed_0_competitor_score':str(seed_cost_other),'right_face_signed_distance_0':str(ti),
            'right_face_signed_distance_1':str(tj),'minimum_mass_0_with_closed_seed_inclusion':'1/3',
            'unavoidable_mass_0_residual':'1/12'}}


def contract():
    return {'state':'fixed_true_material_cell_mass_average','scope':'theory_only_fixed_density_periodic_Laguerre_feasibility',
        'density_model':'keep original nearest-point rho_hat fixed under all auxiliary weights',
        'mass_data':'unchanged original positive m_i; no normalization',
        'cost':'squared periodic distance minus geometric weight w_i',
        'existence_hypotheses':['finite distinct periodic seeds','fixed bounded positive density',
            'positive target masses','sum targets equals integral fixed density'],
        'existence_proved_in_report':True,'weight_uniqueness_in_general_claimed':False,
        'seed_inclusion_criterion':'w_j-w_i < squared periodic seed distance for every j != i',
        'all_seed_interiority_automatic':False,'all_positive_face_distances_automatic':False,
        'oriented_face_distance':'d/2+(w_i-w_j)/(2*d); opposite distance is d minus this',
        'cannot_force_both_exact_mass_and_seed_interiority_for_all_inputs':True,
        'mass_residual_bound_scope':'specified two-seed geometry only; not PDE accuracy or all positive exchange',
        'ordinary_positive_distance_TPFA_general_backend_selected':False,
        'absolute_value_of_negative_face_distance_forbidden':True,
        'old_half_distance_resistance_valid_for_new_cells':False,
        'generic_2D_cut_density_face_conductance_justified':False,
        'weighted_moments_use_fixed_density_integrals':True,'true_geometry_or_mean_identified':False,
        'scalar_or_reference_response_used_for_geometry_weights':False,
        'numerical_partition_algorithm_implemented':False,'new_numerical_gate_selected':False,
        'next_work':'theory-only explicit capacity-defect budget with original masses and fixed density, without forced mass-compatible cells',
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def replay(directory):
    out=Path(directory);receipt=read(out/'definition-receipt.json')
    assert digest(out/'theory-at-definition.md')==receipt['theory_sha256_before_algebra']
    assert contract()==read(out/'feasibility-contract.json')
    assert certificates()==read(out/'algebra-certificates.json')
    print('Eight exact algebra groups, mass/interiority witness and unchanged fixed-density contract replay; no partition optimizer, mesh, graph or PDE executed.',flush=True)


def main(output):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    parent=read(SOURCE/'stage3-status.json');assert parent['recorded_checks_total']==428
    scope=read(SOURCE/'next-theory-scope.json');assert not scope['numerical_gate_selected']
    assert 'Laguerre' in parent['next_work']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'mass-constrained fixed-density Laguerre feasibility and seed-interiority obstruction',
        'parent_manifest_entries_verified':len(manifest),'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
        'parent_theory_scope_sha256':digest(SOURCE/'next-theory-scope.json'),'old_science_programs_not_rerun':True})
    save(out/'feasibility-contract.json',contract());shutil.copyfile(THEORY,out/'theory-at-definition.md')
    save(out/'definition-receipt.json',{'theory_sha256_before_algebra':digest(THEORY),
        'contract_sha256_before_algebra':digest(out/'feasibility-contract.json'),
        'analytic_formulas_and_scope_saved_before_algebra':True,'no_numerical_partition_or_flow_case_selected':True})
    save(out/'algebra-certificates.json',certificates())
    save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC',**{k:0 for k in ZERO_FIELDS},
        'new_exact_algebra_certificate_groups':8,'recorded_checks_total':428,'mass_partition_existence_proved_in_report':True,
        'general_seed_interiority_and_positive_face_distance_guarantee':False,
        'general_Laguerre_TPFA_backend_selected':False,'new_numerical_gate_selected':False,
        'next_work':contract()['next_work'],'ordinary_central_resistance_FV_baseline_retained':True,
        'mass_average_resistance_remains_experimental':True,'finite_node_candidate_admitted':False,
        'contact_transient_physically_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'pdf_generated':False,'software_release_created':False,'published':False})
    replay(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--replay');p.add_argument('--certificates-only',action='store_true');args=p.parse_args()
    if args.certificates_only:print(json.dumps(certificates(),indent=2),flush=True)
    elif args.replay:replay(args.replay)
    elif args.output:main(args.output)
    else:p.error('--output, --replay or --certificates-only required')
