"""One Becker FV case: average-density closure versus material geometry.

Four-step diagnostic chain, not a physical-parameter matrix. Reuse the oracle
baseline, masses, c0 and independent references. Three new positive FV models
separate cellwise averaging, dual-volume density and proxy face placement.
"""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import scipy
from scipy.integrate import quad
from becker import Becker
from experiment import step
from fv_density_geometry import density_from_mass_faces,midpoint_dual_faces,cellwise_conductance
from kernel_joint_study import verify
from spatial_study import normalized_error

MODES=('oracle_resistance','material_faces_mean_density','density_only_dual_proxy','dual_geometry_proxy')


def mass_coordinate_at_x(flow,x,t):
    return flow.B*(flow.z_at_xi(1.)+flow.j*np.asarray(t)/flow.B-flow.z_at_xi(flow.u1*np.asarray(t)-x))


def velocity_at_x(flow,x,t):
    return flow.u1-flow.u(flow.z_at_xi(flow.u1*np.asarray(t)-x))


def run_history(m,c0,g,T):
    steps=len(g);dt=T/steps;c=c0.copy();mean=float(m@c0/m.sum())
    mass=solve=energy=0.;lo=float(c.min());hi=float(c.max())
    for weights in g:
        nxt=step(c,m,weights,dt)
        residual=m*(nxt-c);flux=dt*weights*np.diff(nxt)
        residual[:-1]-=flux;residual[1:]+=flux
        solve=max(solve,float(np.linalg.norm(residual)/np.linalg.norm(m*c)))
        E=lambda v:.5*float(m@((v-mean)**2))
        balance=E(nxt)-E(c)+dt*np.dot(weights,np.diff(nxt)**2)+.5*m@((nxt-c)**2)
        energy=max(energy,abs(float(balance))/float(m@(c*c)))
        mass=max(mass,abs(float(m@(nxt-c0)))/abs(float(m@c0)))
        lo=min(lo,float(nxt.min()));hi=max(hi,float(nxt.max()));c=nxt
    if max(mass,solve,energy)>1e-10 or lo<c0.min()-1e-10 or hi>c0.max()+1e-10:
        raise RuntimeError('unchanged fixed-mass BE algebra checks failed')
    return c,{'max_relative_scalar_mass_drift':mass,'max_relative_linear_residual':solve,
              'max_relative_energy_identity_residual':energy,'minimum_concentration':lo,'maximum_concentration':hi}


def geometry_diagnostics(flow,labels,edges,m,times):
    true=flow.x_mu(edges[None,:],times[:,None])
    point_labels=(labels[:-1]+labels[1:])/2;point_mu=flow.mu(point_labels)
    points=flow.x_mu(point_mu[None,:],times[:,None])
    dual=midpoint_dual_faces(points,true[:,[0,-1]])
    rho_true=density_from_mass_faces(m,true);rho_dual=density_from_mass_faces(m,dual)
    return true,points,dual,rho_true,rho_dual,point_labels,point_mu


def point_trace_nonidentifiability(flow,labels,T):
    if len(labels)<3 or labels[0]!=-1 or labels[-1]!=1 or not np.allclose(np.diff(labels),labels[1]-labels[0],rtol=0,atol=1e-14):
        raise ValueError('this analytic counterexample uses uniform labels on [-1,1]')
    d=labels[1]-labels[0];nodes=(labels[:-1]+labels[1:])/2
    amplitude=.02*d
    phi=lambda a:(1-a*a)*np.cos(np.pi*(a+1)/d)
    base_faces=flow.x_mu(flow.mu(labels),T)
    alternative=base_faces+amplitude*phi(labels)
    base_points=flow.x_mu(flow.mu(nodes),T)
    other_points=base_points+amplitude*phi(nodes)
    min_flow_derivative=flow.u2/flow.u1
    perturbation_bound=amplitude*(2+np.pi/d)
    return {'scope':'kinematically admissible maps, not a second Becker momentum/energy solution',
            'amplitude_fraction_of_initial_width':.02,
            'perturbation':'Xtilde(a,t)=X(a,t)+.02*da*(t/T)*(1-a^2)*cos(pi*(a+1)/da)',
            'minimum_map_derivative_certified_lower_bound':min_flow_derivative-perturbation_bound,
            'same_initial_map':True,'same_node_and_endpoint_histories_analytically':True,
            'final_node_position_max_difference':float(np.max(abs(other_points-base_points))),
            'final_material_volume_max_relative_difference':float(np.max(abs(np.diff(alternative)-np.diff(base_faces))/np.diff(base_faces)))}, {
                'initial_faces':labels,'point_labels':nodes,'base_final_faces':base_faces,
                'alternative_final_faces':alternative,'base_final_points':base_points,'alternative_final_points':other_points}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());physical=root/'runs'/Path(old['physical_source']).name;verify(root,physical)
    p=json.loads((physical/'config.json').read_text())
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    D,T=p['D'],p['T'];n=32
    with np.load(source/f'inputs-cells{n}.npz') as z:
        labels=z['label_edges'];edges=z['mass_edges'];m=z['mass'];c0=z['c0'];reference=z['reference']
        refspace=z['reference_n4096-k4096']-z['reference_n8192-k4096']
        reftime=z['reference_n8192-k4096']-z['reference_n8192-k8192']
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'one fixed Becker case, density averaging and material-volume identifiability',
            'source':str(source),'physical_source':str(physical),'physical_parameters':p,
            'material_columns':n,'steps':[4096,8192],'methods':MODES,
            'state':'same true material-cell mass averages; fixed original m and exact c0',
            'new_candidate_density_input':'mass/face geometry arithmetic only; oracle baseline remains separately reused',
            'synthetic_trajectory_input':'Becker exact motion supplies true material-face tracers, one original midpoint tracer per cell and both endpoints',
            'material_face_method_is_extra_information':'all actual material faces are tracked; not inferred uniquely from interior point paths',
            'diagnostic_chain':['oracle continuous density resistance on true faces',
                                'true faces and mass/true-volume piecewise density',
                                'true faces but mass/dual-volume density (inconsistent hybrid, diagnostic only)',
                                'dual faces and mass/dual-volume density (nonmaterial negative control)'],
            'reference':'reuse saved independent n8192,k8192 mass-FV reference and n4096/k4096 refinement projections',
            'normalization':'unchanged initial mass-weighted fluctuation',
            'diagnostic_error_budget_limit':old['error_separation_diagnostic_limit'],
            'diagnostic_limit_scope':'existing 10% empirical numerical separation, not physical acceptance',
            'density_mass_audit_samples':129,'physical_acceptance':False,'fluid_orc_enabled':False,
            'legacy_ledger_or_kernel_restored':False,'no_F_C_fit_or_repairs':True}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    times=np.linspace(0,T,max(config['steps'])+1)
    true,points,dual,rhot,rhod,point_labels,point_mu=geometry_diagnostics(flow,labels,edges,m,times)
    g={'material_faces_mean_density':cellwise_conductance(true,rhot,D),
       'density_only_dual_proxy':cellwise_conductance(true,rhod,D),
       'dual_geometry_proxy':cellwise_conductance(dual,rhod,D)}
    sampled=np.arange(0,len(times),64);ts=times[sampled,None]
    actual_mu=mass_coordinate_at_x(flow,dual[sampled],ts);actual_mass=np.diff(actual_mu,axis=1)
    fluid_velocity=velocity_at_x(flow,dual[sampled],ts)
    tracer_velocity=velocity_at_x(flow,points[sampled],ts)
    boundary_velocity=velocity_at_x(flow,true[sampled][:,[0,-1]],ts)
    dual_velocity=np.column_stack((boundary_velocity[:,0],
                                  (tracer_velocity[:,:-1]+tracer_velocity[:,1:])/2,
                                  boundary_velocity[:,1]))
    density_on_dual=flow.j/flow.u(flow.z_at_xi(flow.u1*ts-dual[sampled]))
    mass_flux=density_on_dual*(fluid_velocity-dual_velocity)
    mass_rate=mass_flux[:,:-1]-mass_flux[:,1:]
    eps=1e-7
    true_velocity_numerical=(flow.x_mu(edges[None,:],ts+eps)-flow.x_mu(edges[None,:],ts-eps))/(2*eps)
    true_velocity_oracle=velocity_at_x(flow,true[sampled],ts)
    Vtrue=np.diff(true,axis=1);Vdual=np.diff(dual,axis=1)
    geometry={'max_volume_proxy_relative_error_all_8193_frames':float(np.max(abs(Vdual-Vtrue)/Vtrue)),
              'max_density_proxy_relative_error_vs_true_cell_average_all_frames':float(np.max(abs(rhod-rhot)/rhot)),
              'max_true_fluid_mass_in_dual_cell_relative_mismatch_sampled':float(np.max(abs(actual_mass-m)/m)),
              'max_true_material_face_position_proxy_error_all_frames':float(np.max(abs(dual-true))),
              'max_sampled_dual_mass_rate_abs':float(np.max(abs(mass_rate))),
              'true_face_velocity_numerical_slip_max':float(np.max(abs(true_velocity_numerical-true_velocity_oracle))),
              'initial_dual_vs_material_face_max_difference':float(np.max(abs(dual[0]-true[0]))),
              'mean_density_from_true_faces_is_exact_cell_average':True,
              'true_face_inputs_are_still_synthetic_oracle_geometry':True,
              'dual_proxy_valid_as_fixed_material_partition':False}
    np.savez_compressed(output/'geometry-audit.npz',time=times[sampled],material_faces=true[sampled],
                        point_positions=points[sampled],dual_faces=dual[sampled],mass=m,mass_edges=edges,c0=c0,
                        material_average_density=rhot[sampled],dual_density_proxy=rhod[sampled],
                        true_fluid_mass_in_dual_cells=actual_mass,dual_face_mass_flux=mass_flux,dual_cell_mass_rate=mass_rate,
                        point_labels=point_labels,point_mass_coordinates=point_mu,
                        **{key+'_conductance':value[sampled] for key,value in g.items()})
    rows=[];solutions={}
    for steps in config['steps']:
        with np.load(source/f'ordinary_volume_center-cells32-k{steps}.npz') as z:base=z['c']
        solutions['oracle_resistance',steps]=base
        rows.append({'method':'oracle_resistance','steps':steps,'normalized_error':normalized_error(m,c0,base-reference),'origin':'unchanged saved FV baseline'})
        stride=max(config['steps'])//steps
        for mode in MODES[1:]:
            tic=time.monotonic();c,diag=run_history(m,c0,g[mode][:-1:stride],T)
            solutions[mode,steps]=c
            rows.append({'method':mode,'steps':steps,'normalized_error':normalized_error(m,c0,c-reference),
                         'origin':'new density/geometry diagnostic','wall_seconds':time.monotonic()-tic,**diag})
            np.savez_compressed(output/f'{mode}-k{steps}.npz',mass=m,c0=c0,mass_edges=edges,c=c,reference=reference)
            print(mode,steps,'error',rows[-1]['normalized_error'],flush=True)
    norm=lambda v:normalized_error(m,c0,v)
    uncertainty={'reference_space_difference':norm(refspace),'reference_time_difference':norm(reftime),
                 'method_time_differences':{mode:norm(solutions[mode,4096]-solutions[mode,8192]) for mode in MODES}}
    final=[]
    for mode in MODES:
        row=next(r for r in rows if r['method']==mode and r['steps']==8192)
        indicator=uncertainty['method_time_differences'][mode]+uncertainty['reference_space_difference']+uncertainty['reference_time_difference']
        final.append({'method':mode,'normalized_error':row['normalized_error'],
                      'numerical_indicator':indicator,'indicator_over_error':indicator/row['normalized_error'],
                      'numerically_separated':indicator/row['normalized_error']<config['diagnostic_error_budget_limit']})
    chain=[];deltas=[]
    for a,b,name in zip(MODES[:-1],MODES[1:],('cellwise_density_representation','dual_volume_density_error','proxy_face_placement')):
        delta=solutions[b,8192]-solutions[a,8192];deltas.append(delta)
        indicator=uncertainty['method_time_differences'][a]+uncertainty['method_time_differences'][b]
        chain.append({'change':name,'from':a,'to':b,'state_change_norm':norm(delta),
                      'combined_time_difference':indicator,'time_indicator_over_state_change':indicator/norm(delta)})
    total=solutions[MODES[-1],8192]-solutions[MODES[0],8192]
    nullmeta,nullarrays=point_trace_nonidentifiability(flow,labels,T)
    np.savez_compressed(output/'same-point-traces-distinct-material-volumes.npz',**nullarrays,mass=m)
    # Independent physical-space quadrature audits one selected nonmaterial cell.
    cell=int(np.argmax(abs(actual_mass[-1]-m)/m));a,b=dual[-1,cell:cell+2]
    independent=quad(lambda x:float(flow.j/flow.u(flow.z_at_xi(flow.u1*T-x))),a,b,epsabs=1e-12)[0]
    result={'geometry_diagnostics':geometry,'candidate_runs':rows,'final_comparison':final,
            'uncertainty':uncertainty,'successive_change_chain':chain,
            'chain_vector_closure_max_abs':float(np.max(abs(sum(deltas)-total))),
            'no_additive_norm_claim':True,
            'independent_dual_cell_mass_check':{'cell':cell,'stored_material_mass':float(m[cell]),
                 'true_fluid_mass_in_dual_cell':float(actual_mass[-1,cell]),'independent_quadrature_mass':independent,
                 'mass_quadrature_difference':abs(independent-actual_mass[-1,cell])},
            'point_only_nonidentifiability':nullmeta,
            'physical_acceptance':False,
            'limits':['one smooth Becker case','mean density uses supplied true material-face histories',
                      'dual and hybrid variants are diagnostics, not accepted material FV schemes','no legacy 2D ledger or density reconstruction restored']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'change-chain.npz',mass=m,c0=c0,mass_edges=edges,reference=reference,
                        **{mode:solutions[mode,8192] for mode in MODES},successive_changes=np.array(deltas),total_proxy_change=total)
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps({'geometry':geometry,'final':final,'changes':chain,'point_only':nullmeta},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
