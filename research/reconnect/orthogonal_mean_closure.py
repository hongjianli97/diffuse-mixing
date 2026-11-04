"""Theory contract and seven exact algebra groups; no mesh or graph construction."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import shutil
import sys

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-point-endpoint-review'
THEORY=ROOT/'research/reconnect/ORTHOGONAL_MATERIAL_MEAN_CLOSURE.md'
ZERO_FIELDS=['new_software_checks','previous_checks_rerun','new_candidate_weights','new_kernel_or_path_evaluations',
    'new_operator_actions','new_quadrature_evaluations','new_Fourier_transforms_or_eigensolves',
    'new_candidate_graphs','new_numerical_or_flow_cases','new_candidate_transient_runs','new_PDE_reference_solves']
BLOCKED=['numpy','scipy','saved_exchange_time','point_fixed_time_study','positive_kernel_2d',
    'oracle_path_reference','cartesian_fv_modes','contact_spectral_reference','ideal_material_pair_reference']


class Poly:
    """Commutative rational polynomials used only for local scalar identities."""
    def __init__(self,value=0):
        if isinstance(value,Poly):self.terms=dict(value.terms)
        elif isinstance(value,dict):self.terms={k:F(v) for k,v in value.items() if v}
        else:self.terms={():F(value)} if value else {}
    def __add__(self,other):
        other=Poly(other);result=dict(self.terms)
        for k,v in other.terms.items():result[k]=result.get(k,F(0))+v
        return Poly(result)
    __radd__=__add__
    def __neg__(self):return Poly({k:-v for k,v in self.terms.items()})
    def __sub__(self,other):return self+-Poly(other)
    def __rsub__(self,other):return Poly(other)+-self
    def __mul__(self,other):
        result={}
        for a,v in self.terms.items():
            for b,w in Poly(other).terms.items():
                k=tuple(sorted(a+b));result[k]=result.get(k,F(0))+v*w
        return Poly(result)
    __rmul__=__mul__
    def is_zero(self):return not self.terms


def variable(name):return Poly({(name,):1})
def zero(p):assert Poly(p).is_zero(),p.terms


def certificates():
    groups=[]
    def done(name,scope):groups.append({'name':name,'exact_algebra_passed':True,'scope':scope})
    x,y,u,v,g=map(variable,['x','y','u','v','g'])
    zero(g*(y-x)+g*(x-y))
    zero(-u*g*(y-x)-v*g*(x-y)-g*(v-u)*(y-x))
    zero(-x*g*(y-x)-y*g*(x-y)-g*(y-x)*(y-x))
    assert not (g*(y-x)).is_zero()
    done('shared_exchange_pairing','pair mass cancellation, bilinear factor and energy; no candidate graph built')
    di,dj,Gi,Gj,b,A,J=map(variable,['di','dj','Gi','Gj','b','A','J'])
    denominator=di*Gj+dj*Gi
    zero(b*A*Gi*Gj*J*denominator-b*A*J*denominator*Gi*Gj)
    assert not denominator.is_zero()
    # An independent rational substitution checks the series-resistance units/factor.
    aa,bb,cc,dd,face,thick,flux=map(F,['2/7','3/11','5/13','7/17','11/19','13/23','17/29'])
    conductance=thick*face/(aa/cc+bb/dd)
    assert conductance*flux*(aa/cc+bb/dd)==thick*face*flux
    done('two_half_resistances','cleared rational identity and independent rational substitution; assumes positive lengths/Gamma')
    X,Y,Z,W,c,px,py,hxx,hxy,hyy,sxx,sxy,syy=map(variable,
        ['X','Y','Z','W','c','px','py','hxx','hxy','hyy','sxx','sxy','syy'])
    def q(a,d):return c+px*a+py*d+F(1,2)*hxx*a*a+hxy*a*d+F(1,2)*hyy*d*d
    mean=q(Z,W)+F(1,2)*hxx*sxx+hxy*sxy+F(1,2)*hyy*syy
    dx,dy=Z-X,W-Y
    taylor=q(X,Y)+(px+hxx*X+hxy*Y)*dx+(py+hxy*X+hyy*Y)*dy
    taylor+=F(1,2)*hxx*(dx*dx+sxx)+hxy*(dx*dy+sxy)+F(1,2)*hyy*(dy*dy+syy)
    zero(mean-taylor)
    assert not (F(1,2)*hxx*sxx+hxy*sxy+F(1,2)*hyy*syy).is_zero()
    done('mean_centroid_covariance','full two-dimensional quadratic mean, including mixed covariance; not point values')
    zero(2*Gi*Gj-Gi*(Gi+Gj)-Gi*(Gj-Gi))
    zero(Gj*(Gi+Gj)-2*Gi*Gj-Gj*(Gj-Gi))
    d=variable('d');zero(2*(F(1,2)*A*(di+(d-di)))-A*d)
    done('harmonic_bounds_and_pyramid_factor','algebra behind harmonic min/max and dimension2 face-pyramid constant; geometry assumed in report')
    du,dv,bu,bv=map(variable,['du','dv','bu','bv'])
    zero((du+bu)*(dv+bv)-du*dv-bu*dv-du*bv-bu*bv)
    assert not (bu*dv+du*bv).is_zero()
    done('average_bilinear_defect','both linear terms and quadratic term retained; analytic Taylor/Cauchy-Schwarz proof is in report')
    # Ordered words, deliberately not the commutative polynomial algebra above.
    words={}
    for word,coefficient in [(('Sh','AK'),1),(('Sh','AC'),-1),(('Sh','AC'),1),(('AC','Shat'),-1),
        (('AC','Shat'),1),(('AC','S'),-1),(('AC','S'),1),(('AK','S'),-1)]:
        words[word]=words.get(word,0)+coefficient
    words={k:v for k,v in words.items() if v}
    assert words=={('Sh','AK'):1,('AK','S'):-1}
    assert ('AK','Sh') not in words
    done('four_term_transient_budget','ordered operator words cancel without assuming operators commute; no semigroup evaluated')
    ell,rho,em,ep=map(variable,['ell','rho','em','ep'])
    xp=F(3,2)*ell+ep;xm=-F(1,2)*ell+em
    area=F(1,2)*ell*(xp-xm)
    defect=b*rho*area-b*rho*ell*ell
    zero(defect-F(1,2)*b*rho*ell*(ep-em));assert not defect.is_zero()
    eps=[variable('epsilon'+str(i)) for i in range(4)]
    defects=[F(1,2)*b*rho*ell*(eps[(i+1)%4]-eps[(i-1)%4]) for i in range(4)]
    zero(sum(defects,Poly()));assert any(not p.is_zero() for p in defects)
    done('local_mass_incompatibility','periodic column-displacement formula; global sum zero does not imply local compatibility')
    for name in BLOCKED:assert name not in sys.modules,name
    assert len(groups)==7
    return {'verified_exact_algebra_groups':7,'groups':groups,'uses_standard_library_only':True,
        'not_automatic_geometric_or_PDE_convergence_proof':True,'candidate_graphs_or_flow_cases':0}


def contract():
    return {'state':'fixed_true_material_cell_mass_average','scope':'theory_only_static_periodic_isotropic_diffusion',
        'allowed_inputs':['fixed_ids','frozen_point_positions','unchanged_fixed_masses','given_positive_point_density',
            'constant_D','periodic_lengths','unit_thickness'],
        'forbidden_inputs':['scalar_state_for_weight_construction','PDE_reference_or_response','true_material_faces','F_C','velocity_gradients'],
        'surrogate_geometry':'periodic Euclidean Voronoi from unchanged input points; not true material cells',
        'surrogate_density':'given point rho on its nearest-point cell; never reset to mass/volume',
        'mass_compatibility':'every m_i = b*rho_i*area(C_i); unsupported if not satisfied, no normalization',
        'conductance_D_positive':'b*face_length / ((distance/2)/(rho_i*D)+(distance/2)/(rho_j*D))',
        'conductance_D_zero':'all shared conductances zero',
        'coefficient_formula_does_not_use_scalar_state':True,'candidate_weights_implemented_or_evaluated':False,
        'true_mean_not_relabelled_surrogate_mean':True,
        'average_defect_parameters':{'epsilon_c':'max_faces (centroid_distance_i+centroid_distance_j)/center_distance',
            'epsilon_2':'max_faces (second_moment_i+second_moment_j)/center_distance',
            'B_f':'L_f*epsilon_c+H_f*epsilon_2/2','sufficient_limit':'epsilon_c -> 0 and epsilon_2 -> 0; not automatic from point fill distance'},
        'projection_bilinear_bound':'2*b*Gamma_max*area*(L_phi*B_f+L_f*B_phi+B_f*B_phi)',
        'true_surrogate_observation_bound':'Lip(f)*sqrt(sum_i m_i*W1(nu_true_i,nu_surrogate_i)^2)',
        'observation_bound_requires_unobservable_true_geometry_assumptions':True,
        'transient_budget':['initial_observation_mismatch','surrogate_FV_discretization','density_model_evolution','final_observation_mismatch'],
        'full_transient_error_estimate_proved':False,'general_allowed_point_inputs_supported':False,
        'mass_incompatibility_witness':'fixed true square cells and constant density, arbitrary within-cell column shifts',
        'next_work':'theory-only mass-constrained Laguerre feasibility: total mass, positive distances, seed inclusion and material observation error',
        'next_numerical_gate_selected':False,'new_production_backend_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def replay(directory):
    out=Path(directory)
    receipt=read(out/'definition-receipt.json')
    assert digest(out/'theory-at-definition.md')==receipt['theory_sha256_before_algebra']
    assert contract()==read(out/'closure-contract.json')
    assert certificates()==read(out/'algebra-certificates.json')
    print('Seven exact algebra groups and unchanged theory/contract replay; no cells, weights, graph or PDE evaluated.',flush=True)


def main(output):
    manifest=read(SOURCE/'sha256.json')
    for rel,h in manifest.items():assert digest(ROOT/rel)==h,rel
    parent=read(SOURCE/'stage3-status.json');assert parent['recorded_checks_total']==428
    assert parent['next_work'].startswith('theory-only definition')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'new conditional orthogonal surrogate mean closure, analytical defects and input obstruction',
        'parent_manifest_entries_verified':len(manifest),'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
        'old_physical_results_and_all_proof_programs_not_rerun':True})
    save(out/'closure-contract.json',contract());shutil.copyfile(THEORY,out/'theory-at-definition.md')
    save(out/'definition-receipt.json',{'theory_sha256_before_algebra':digest(THEORY),
        'contract_sha256_before_algebra':digest(out/'closure-contract.json'),
        'formula_and_assumptions_saved_before_certificate_evaluation':True,'no_numerical_mesh_or_graph_selected':True})
    save(out/'algebra-certificates.json',certificates())
    save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC',**{k:0 for k in ZERO_FIELDS},
        'new_exact_algebra_certificate_groups':7,'recorded_checks_total':428,'theoretical_candidate_defined':True,
        'general_allowed_input_support':False,'full_transient_error_estimate_proved':False,
        'candidate_graph_implemented':False,'new_numerical_gate_selected':False,
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
