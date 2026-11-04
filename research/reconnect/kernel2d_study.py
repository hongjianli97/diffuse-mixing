"""One predeclared periodic Fourier probe; instantaneous action only."""
from pathlib import Path
import argparse,json,platform,sys
import numpy as np
import scipy
from scipy.special import roots_legendre,j0,j1
from trajectory_chain import TrajectorySamples,SI_UNITS,laplacian,operator_audit
from positive_kernel_2d import build_periodic_exchange,kernel_value,slope_over_radius
from chain_study import verify


def fixture(N=32):
    axis=(np.arange(N)+.5)/N;a=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    times=np.array([0.,.05]);velocity=np.array([.2,-.1]);x=np.mod(a[None,...]+times[:,None,None]*velocity,1.)
    factor=np.sinc(1/N)**2;c0=.5+.2*factor*np.cos(2*np.pi*a[:,0])*np.cos(2*np.pi*a[:,1])
    data=TrajectorySamples(times,x,np.arange(N*N),np.full(N*N,1/(N*N)),c0,SI_UNITS,rho=np.ones((2,N*N)))
    return data,a,factor


def continuum_decay(h,D,wavenumber,order,form='B'):
    z,w=roots_legendre(order);r=h*(z+1);weights=h*w
    if form=='B':return float(4*np.pi*D*np.sum(weights*slope_over_radius(r,h)*(1-j0(wavenumber*r))*r))
    if form=='W':return float(4*np.pi*D*wavenumber*np.sum(weights*kernel_value(r,h)*j1(wavenumber*r)))
    raise ValueError('unknown independent radial quadrature form')


def main(output):
    root=Path(__file__).resolve().parents[2];verify(root,root/'runs/reconnect-mainline');out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    config={'study':'one periodic homogeneous 2D oracle-density point-kernel instantaneous Fourier control',
        'N_per_direction':32,'nodes':1024,'periods_m':[1.,1.],'thickness_m':1.,'rho_kg_per_m3':1.,'D_m2_per_s':.02,'h_m':.09,
        'time_samples_s':[0.,.05],'frozen_frame':1,'translation_velocity_m_per_s':[.2,-.1],
        'state':'.5+.2*sinc(1/N)^2*cos(2*pi*a_x)*cos(2*pi*a_y), exact fixed uniform material-square mass average',
        'manufactured_state_is_diffusively_evolved_c0':False,'state_interpretation':'prescribed frozen diagnostic, no time integration',
        'continuum_local_rate':'-8*pi^2*D*(cbar-.5)',
        'finite_width_reference':'lambda_h=4*pi*D integral_0^(2h) B_h(r)*(1-J0(k*r))*r dr, k=2*pi*sqrt(2)',
        'independent_reference':'integration by parts: 4*pi*D*k integral W_h(r)*J1(k*r) dr; orders64/128',
        'quadrature_weight':'m/(rho*unit thickness), no true cell geometry supplied to candidate',
        'kernel':'2D normalized Wendland C4, C2=9/(4*pi); B=-Wprime/r analytic',
        'conductance':'2*area_i*area_j*unit_thickness*harmonic(rho_i*D,rho_j*D)*B_h(r_ij)',
        'h_and_state_fitted':False,'parameter_or_mesh_scan':False,'true_faces_or_F_C_candidate_inputs':False,
        'new_fluid_transient_runs':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'acceptance_limitations':['one smooth homogeneous periodic state, one grid and one h','no general mesh, density jump, wall, temporal or DNS validation']}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    data,a,factor=fixture();D=config['D_m2_per_s'];h=config['h_m'];k=2*np.pi*np.sqrt(2)
    L,arrays,diag=build_periodic_exchange(data,1,D,h,[1.,1.]);rate=-(L@data.c0)/data.mass
    reference=-k*k*D*(data.c0-.5);norm=float(np.sqrt(data.mass@(reference*reference)))
    lambdas={f'{form}{order}':continuum_decay(h,D,k,order,form) for form in ['B','W'] for order in [64,128]}
    finite=-lambdas['W128']*(data.c0-.5);width=finite-reference;quadrature=rate-finite;total=rate-reference
    error=lambda x:float(np.sqrt(data.mass@(x*x))/norm)
    dense=laplacian(len(data.mass),arrays['pairs'],arrays['conductances']);audit=operator_audit(data.mass,-dense/data.mass[:,None])
    pair_rate=np.zeros(len(data.mass));i,j=arrays['pairs'].T;flux=arrays['conductances']*(data.c0[j]-data.c0[i])
    np.add.at(pair_rate,i,flux);np.add.at(pair_rate,j,-flux);pair_rate/=data.mass
    # Direct stencil Fourier symbol, computed from geometry and g without a
    # least-squares fit to the scalar action. Both diagonal modes are audited.
    touch=(i==0)|(j==0);dx=arrays['minimum_image_displacement'][touch];gg=arrays['conductances'][touch]
    symbols=[float(np.sum(gg*(1-np.cos(2*np.pi*(dx[:,0]+sign*dx[:,1]))))/data.mass[0]) for sign in [1,-1]]
    discrete_decay=symbols[0]
    reconstruction=-discrete_decay*(data.c0-.5);reference_spread=max(lambdas.values())-min(lambdas.values())
    result={'operator_diagnostics':diag,'graph_layer_audit':audit,'relative_mass_weighted_instantaneous_rate_error':error(total),
        'finite_width_rate_bias_relative_to_local_PDE':error(width),'node_quadrature_rate_difference_relative_to_local_PDE':error(quadrature),
        'continuum_local_decay_per_s':float(k*k*D),'continuum_finite_width_decay_per_s':lambdas,
        'discrete_Fourier_symbols_plus_minus_per_s':symbols,
        'continuum_quadrature_and_form_spread_per_s':float(reference_spread),
        'reference_spread_relative_to_local_decay':float(reference_spread/(k*k*D)),
        'raw_action_vs_discrete_Fourier_mode_max_difference_per_s':float(np.max(abs(rate-reconstruction))),
        'pair_scatter_vs_sparse_action_max_difference_per_s':float(np.max(abs(rate-pair_rate))),
        'shared_graph_interface_vs_sparse_L_max_difference':float(np.max(abs(dense-L.toarray()))),
        'signed_error_decomposition_max_residual_per_s':float(np.max(abs(total-width-quadrature))),
        'max_point_sample_vs_mass_average_state_difference':float(np.max(abs(data.c0-(.5+.2*np.cos(2*np.pi*a[:,0])*np.cos(2*np.pi*a[:,1]))))),
        'exact_mass_average_attenuation_factor':float(factor),'sum_quadrature_area_m2':float(arrays['quadrature_area'].sum()),
        'raw_weighted_rate_mass_residual_kg_per_s':float(data.mass@rate),
        'new_fluid_transient_runs':0,'new_propagators_constructed':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'not_claimed':['spatial or kernel-width convergence','variable density or shock validation','true volume identifiability from arbitrary point tracks','accepted physical flow backend']}
    np.savez_compressed(out/'periodic-kernel2d.npz',time=data.time,positions=data.positions,ids=data.ids,mass=data.mass,c0=data.c0,rho=data.rho,
        material_label_centers=a,reference_rate=reference,kernel_rate=rate,continuum_finite_width_rate=finite,finite_width_error=width,node_quadrature_error=quadrature,
        L_data=L.data,L_indices=L.indices,L_indptr=L.indptr,L_shape=np.array(L.shape),**arrays)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
