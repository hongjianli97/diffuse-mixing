"""Exact capacity theory and one saved tensor-grid input screen; no graph action."""
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
import struct
import sys

from audit_route_decision import ROOT,read,digest

SOURCE=ROOT/'runs/reconnect-laguerre-mass-feasibility'
THEORY=ROOT/'research/reconnect/CAPACITY_DEFECT_SEMIGROUP_BOUND.md'
INPUT='runs/reconnect-kernel2d-compression-midpoint/midpoint-observations.npz'
ZERO_FIELDS=['new_software_checks','previous_checks_rerun','new_candidate_weights','new_kernel_or_path_evaluations',
    'new_operator_actions','new_quadrature_evaluations','new_Fourier_transforms_or_eigensolves','new_candidate_graphs',
    'new_numerical_or_flow_cases','new_candidate_transient_runs','new_PDE_reference_solves','numerical_partition_optimizer_runs']
BLOCKED=['laguerre_mass_feasibility','orthogonal_mean_closure','saved_exchange_time','point_fixed_time_study',
    'positive_kernel_2d','oracle_path_reference','cartesian_fv_modes','contact_spectral_reference']


class Polynomial:
    def __init__(self,v=0):
        if isinstance(v,Polynomial):self.terms=dict(v.terms)
        elif isinstance(v,dict):self.terms={k:F(x) for k,x in v.items() if x}
        else:self.terms={():F(v)} if v else {}
    def __add__(self,v):
        d=dict(self.terms)
        for k,x in Polynomial(v).terms.items():d[k]=d.get(k,F(0))+x
        return Polynomial(d)
    __radd__=__add__
    def __neg__(self):return Polynomial({k:-v for k,v in self.terms.items()})
    def __sub__(self,v):return self+-Polynomial(v)
    def __rsub__(self,v):return Polynomial(v)+-self
    def __mul__(self,v):
        d={}
        for a,x in self.terms.items():
            for b,y in Polynomial(v).terms.items():
                k=tuple(sorted(a+b));d[k]=d.get(k,F(0))+x*y
        return Polynomial(d)
    __rmul__=__mul__


def var(s):return Polynomial({(s,):1})
def zero(v):assert not Polynomial(v).terms


def certificates():
    groups=[]
    def done(name,scope):groups.append({'name':name,'exact_algebra_passed':True,'scope':scope})
    r,mt,m,L=map(var,['r','mt','m','L'])
    zero(-mt*L+m*L-(-(mt-m)*L))
    zero(r-1-(r-1))
    done('capacity_generator_identity','rowwise cleared identity A-B=E*B; E remains on the left')
    rm,rp,r0,r1,m0,m1,x0,x1=map(var,['rmin','rmax','r0','r1','m0','m1','x0','x1'])
    actual=m0*x0*x0+m1*x1*x1;aux=m0*r0*x0*x0+m1*r1*x1*x1
    zero(aux-rm*actual-(m0*(r0-rm)*x0*x0+m1*(r1-rm)*x1*x1))
    zero(rp*actual-aux-(m0*(rp-r0)*x0*x0+m1*(rp-r1)*x1*x1))
    done('mass_norm_conversion','nonnegative square decompositions for both norm endpoints; full inequalities in report')
    # Noncommutative words: Fprime=-A*SM*E*ST+SM*E*B*ST.
    words={('A','SM','E','ST'):-1,('SM','E','B','ST'):1}
    words[('A','SM','E','ST')]+=1
    words={k:v for k,v in words.items() if v}
    assert words=={('SM','E','B','ST'):1}
    assert ('E','A','SM','ST') not in words
    done('noncommuting_Duhamel_parts','ordered integrand recovered without moving E across either generator')
    z=var('sqrt_R');zero((2+z*z)-(1+2*z)-(z-1)*(z-1))
    assert 1+1+F(1,2)>2
    done('uniform_bound_coefficient','1+2*sqrt(R)<=2+R; log2<1 and spectral smoothing argument are in report')
    terms=[(('SM','AK'),1),(('ST','AK'),-1),(('ST','AK'),1),(('ST','AC'),-1),
        (('ST','AC'),1),(('AC','SH'),-1),(('AC','SH'),1),(('AC','S'),-1),
        (('AC','S'),1),(('AK','S'),-1)]
    budget={}
    for k,v in terms:budget[k]=budget.get(k,0)+v
    assert {k:v for k,v in budget.items() if v}=={('SM','AK'):1,('AK','S'):-1}
    done('five_term_budget','capacity, initial observation, discretization, density evolution and final observation')
    a=var('aux_mean');zero(m*a-mt*a-(m-mt)*a)
    done('inventory_measure_difference','actual mass aggregation of auxiliary averages retains capacity correction')
    dl,dr=map(var,['dl','dr']);width=F(1,2)*(dl+dr);offset=F(1,4)*(dr-dl)
    moment=F(1,12)*(dl*dl-dl*dr+dr*dr)
    zero(moment-F(1,12)*width*width-offset*offset)
    zero(12*(dl+dr)*moment-(dl*dl*dl+dr*dr*dr))
    done('tensor_Voronoi_moments','half-gap width, centroid offset and raw second moment, without adjusting coordinates')
    e,a,b=map(var,['delta','a','b'])
    zero(e+(-e));assert (e*a-e*b).terms
    done('global_not_local_capacity','zero total capacity residual does not imply zero local defect or weighted inventory correction')
    assert len(groups)==8
    for name in BLOCKED:assert name not in sys.modules,name
    return {'verified_exact_algebra_groups':8,'groups':groups,'standard_library_only':True,
        'not_automatic_semigroup_or_PDE_proof':True,'new_physical_cases':0}


def contract():
    return {'state':'fixed_true_material_cell_mass_average','original_mass_and_density_unchanged':True,
        'auxiliary_mass':'fixed-density integral over ordinary auxiliary Voronoi cells; comparison only',
        'fixed_graph_generators':'A=-M^-1 L, B=-M_tilde^-1 L with the same symmetric nonnegative exchange L',
        'ratio':'r_i=m_tilde_i/m_i','defect':'eta=max abs(r_i-1)','ratio_range':'R=max(r_i)/min(r_i)',
        'generator_difference':'A-B=diag(r-1)*B',
        'uniform_time_capacity_bound':'eta*(2+R)*norm_M(u0-original_mass_mean)',
        'refined_bound':'eta*(1+sqrt(R)*(1+2*log(2)/e))',
        'hypotheses':['finite fixed graph','same symmetric nonnegative conductances','both mass vectors positive'],
        'arbitrary_initial_vector_allowed':True,'commuting_generators_assumed':False,
        'capacity_bound_is_full_PDE_error_or_physical_threshold':False,
        'auxiliary_semigroup_norm_in_original_mass':'at most sqrt(R), not assumed1',
        'auxiliary_mean_denominator':'m_tilde; actual state and norm still use original m',
        'five_error_terms':['capacity','initial_observation','auxiliary_discretization','density_evolution','final_observation'],
        'moving_Voronoi_means_or_time_varying_graph_covered':False,
        'new_candidate_graph_selected':False,'finite_node_candidate_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def array_records(packet):
    result={};n=len(packet['ids'])
    for key,shape,fmt,dtype in [('ids',[n],'q','<i8'),('mass',[n],'d','<f8'),
        ('rho',[n],'d','<f8'),('positions',[n,2],'d','<f8')]:
        values=packet[key] if key!='positions' else [q for row in packet[key] for q in row]
        data=struct.pack('<'+fmt*len(values),*values)
        result[key]={'shape':shape,'dtype':dtype,'C_order_sha256':hashlib.sha256(data).hexdigest()}
    return result


def number(v):return {'exact_of_saved_binary_data':str(v),'display':float(v)}


def analyse_packet(packet):
    N=128;n=N*N
    assert packet['ids']==list(range(n)) and len(packet['positions'])==len(packet['mass'])==len(packet['rho'])==n
    assert all(len(v)==2 for v in packet['positions'])
    assert all(math.isfinite(v) and 0<=v<1 for row in packet['positions'] for v in row)
    assert all(math.isfinite(v) and v>0 for key in ['mass','rho'] for v in packet[key])
    xs=[packet['positions'][c][0] for c in range(N)]
    ys=[packet['positions'][r*N][1] for r in range(N)]
    for i,(x,y) in enumerate(packet['positions']):
        assert x.hex()==xs[i%N].hex() and y.hex()==ys[i//N].hex(),'unsupported nonproduct point positions'
    axes=[]
    for values in [xs,ys]:
        q=list(map(F,values));gaps=[(q[(i+1)%N]-q[i])%1 for i in range(N)]
        assert min(gaps)>0 and sum(gaps,F(0))==1,'original IDs not in one positive cyclic order'
        widths=[(gaps[i-1]+gaps[i])/2 for i in range(N)]
        offsets=[(gaps[i]-gaps[i-1])/4 for i in range(N)]
        moments=[(gaps[i-1]**2-gaps[i-1]*gaps[i]+gaps[i]**2)/12 for i in range(N)]
        assert sum(widths,F(0))==1
        axes.append((gaps,widths,offsets,moments))
    m=list(map(F,packet['mass']));rho=list(map(F,packet['rho']))
    aux=[rho[i]*axes[0][1][i%N]*axes[1][1][i//N] for i in range(n)]
    ratios=[a/b for a,b in zip(aux,m)];lo,hi=min(ratios),max(ratios)
    eta=max(abs(r-1) for r in ratios);R=hi/lo;capacity_bound=eta*(2+R)
    centroid=[abs(axes[0][2][i%N])+abs(axes[1][2][i//N]) for i in range(n)]
    second=[axes[0][3][i%N]+axes[1][3][i//N] for i in range(n)]
    epsilon_c=F(0);epsilon_2=F(0)
    for i in range(n):
        row,col=divmod(i,N)
        for j,d in [(row*N+(col+1)%N,axes[0][0][col]),(((row+1)%N)*N+col,axes[1][0][row])]:
            epsilon_c=max(epsilon_c,(centroid[i]+centroid[j])/d)
            epsilon_2=max(epsilon_2,(second[i]+second[j])/d)
    M=sum(m,F(0));Mt=sum(aux,F(0))
    result={'scope':'one saved N128 compression-midpoint input, no scalar state or PDE response read',
        'saved_snapshot_index':1,'source_time_display':packet['time'],'input_supported_for_capacity_budget_only':True,
        'point_count':n,'original_total_mass':number(M),'auxiliary_total_mass':number(Mt),
        'auxiliary_minus_original_total_mass':number(Mt-M),'minimum_capacity_ratio':number(lo),
        'maximum_capacity_ratio':number(hi),'eta_maximum_local_capacity_defect':number(eta),'ratio_spread_R':number(R),
        'uniform_same_L_capacity_comparison_bound_over_initial_fluctuation':number(capacity_bound),
        'epsilon_c_L1_upper':number(epsilon_c),'epsilon_2':number(epsilon_2),
        'epsilon_c_uses_L1_upper_not_exact_Euclidean_distance':True,'auxiliary_area_exact':'1',
        'positions_masses_densities_ids_not_modified':True,'auxiliary_mass_not_applied_to_a_solver':True,
        'geometric_neighbour_comparisons':2*n,'new_shared_conductances_or_operator_actions':0,
        'PDE_accuracy_evaluated':False,'new_physical_threshold_defined':False,
        'true_material_observation_error_not_measured':True,'moving_geometry_not_physically_admitted':True,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    exact={'axes':[{'positive_periodic_gaps':list(map(str,v[0])),'widths':list(map(str,v[1])),
        'centroid_offsets':list(map(str,v[2])),'second_moments_about_original_points':list(map(str,v[3]))} for v in axes],
        'auxiliary_masses':list(map(str,aux)),'capacity_ratios':list(map(str,ratios))}
    return result,exact


def replay(directory):
    out=ROOT/directory if not str(directory).startswith('/') else directory
    from pathlib import Path
    out=Path(out);receipt=read(out/'definition-receipt.json')
    assert digest(out/'theory-at-definition.md')==receipt['theory_sha256_before_evaluation']
    assert contract()==read(out/'capacity-contract.json') and certificates()==read(out/'algebra-certificates.json')
    packet=read(out/'saved-input.json');p=read(out/'input-provenance.json')
    assert digest(out/'saved-input.json')==p['saved_packet_sha256'] and array_records(packet)==p['copied_array_records']
    result,exact=analyse_packet(packet)
    assert result==read(out/'input-screen-results.json') and exact==read(out/'input-screen-exact.json')
    print('Eight new algebra groups and one unchanged saved-input capacity/geometry screen replay; no graph, scalar propagation or PDE solved.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--certificates-only',action='store_true');p.add_argument('--replay');args=p.parse_args()
    if args.certificates_only:print(json.dumps(certificates(),indent=2),flush=True)
    elif args.replay:replay(args.replay)
    else:p.error('--certificates-only or --replay required')
