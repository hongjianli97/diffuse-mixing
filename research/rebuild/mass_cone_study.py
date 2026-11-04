"""One preselected Becker column: necessary bulk moment-cone feasibility.

The 64-cell closed problem is unchanged. A virtual +/- one-row extension is
an explicitly different diagnostic context, never an implicit boundary fix.
Only two supports are evaluated: adjacent columns and one geometry-selected
bracket. Original mass/c0 are kept; no diffusion operator is constructed.
"""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from scipy.special import roots_legendre
from becker import Becker
from becker_polygon_benchmark import from_world
from kernel_joint_study import verify
from material_polygon_history import MaterialPolygonHistory
from mass_moment_cone import require_bulk_context,moment_features,solve_moment_cone


def stream_cell_moments(flow,edges,t,order=64):
    z,w=roots_legendre(order);a,b=edges[:-1],edges[1:]
    mu=(a+b)[:,None]/2+(b-a)[:,None]*z/2;x=flow.x_mu(mu,t)
    mean=x@(w/2);variance=((x-mean[:,None])**2)@(w/2)
    return mean,variance


def primitive_targets(flow,edges,means,column,t,D=.02,shear=.6):
    i=column;faces=flow.x_mu(edges,t);rho=flow.rho_mu(edges,t);dm=edges[i+1]-edges[i];center=means[i]
    drift=D*(rho[i+1]-rho[i])/dm
    J=((faces[i+1]-center)*rho[i+1]-(faces[i]-center)*rho[i]-dm)/dm
    beta=np.array([drift,-shear*drift]);Gamma=np.array([[2*D*(1+J),-D*shear*J],[-D*shear*J,2*D]])
    return beta,Gamma,float(J)


def independent_boundary_targets(flow,edges,column,mean,t,D=.02,shear=.6,hy=.5,order=64):
    # Direct physical boundary quadrature of one parallelogram, independent
    # of the integration-by-parts primitive formula used by primitive_targets.
    x=flow.x_mu(edges[[column,column+1]],t)
    quad=np.array([[x[0]-shear*hy/2,-hy/2],[x[1]-shear*hy/2,-hy/2],
                   [x[1]+shear*hy/2,hy/2],[x[0]+shear*hy/2,hy/2]])
    z,w=roots_legendre(order);s=(z+1)/2;beta=np.zeros(2);Gamma=np.zeros((2,2))
    for j in range(4):
        a,b=quad[j],quad[(j+1)%4];v=b-a;normal=np.array([v[1],-v[0]])
        p=a[None,:]+s[:,None]*v;xi=p[:,0]-shear*p[:,1];rho=flow.j/flow.u(flow.z_at_xi(flow.u1*t-xi))
        local=p-np.array([mean,0.]);weights=rho*w/2
        beta+=np.sum(weights)*normal
        first=np.sum(weights[:,None]*local,axis=0)
        Gamma+=first[:,None]*normal[None,:]+normal[:,None]*first[None,:]
    mass=(edges[column+1]-edges[column])*hy
    return D*beta/mass,D*Gamma/mass


def build_support(means,variances,column,columns,shear=.6,hy=.5):
    labels=np.array([(j,q) for q in (-1,0,1) for j in columns if not (j==column and q==0)],dtype=int)
    j,q=labels.T;dxi=means[j]-means[column]
    d=np.column_stack((dxi+shear*q*hy,q*hy))
    covariance=np.zeros((len(labels),2,2));covariance[:,0,0]=variances[j]-variances[column]
    return moment_features(d,covariance),{'neighbour_column_and_virtual_row':labels,'displacements':d,
        'covariance_difference':covariance,'source_mass_coordinate_variance':variances}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters'];t=p['T']/2;column=16;D=p['D'];hy=.5;shear=.6
    polygon=root/'runs/rebuild-stage3-polygon-transport';history=MaterialPolygonHistory.load(polygon/'polygon-record-32.npz')
    with np.load(root/'runs/rebuild-stage3-density-geometry/geometry-audit.npz') as z:edges=z['mass_edges']
    with np.load(source/'mpfa-t0.05.npz') as z:
        old_centers=z['mass_centroids'];mass=z['mass'];c0=z['original_c0'];positions=z['positions']
    boundary_cells=np.any(history.incidence[:,history.boundary_edge_mask]!=0,axis=1)
    # Guard against treating any of the actual two-strip cells as an interior stencil.
    require_bulk_context(bool(boundary_cells[column]),'explicit_virtual_transverse_extension')
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    means32,var32=stream_cell_moments(flow,edges,t,32);means,var=stream_cell_moments(flow,edges,t,64)
    beta,Gamma,J=primitive_targets(flow,edges,means,column,t,D,shear)
    target=np.array([*beta,Gamma[0,0],Gamma[0,1],Gamma[1,1]])
    ratio=Gamma[0,1]/Gamma[1,1];required=hy*abs(shear-ratio)
    left=int(np.searchsorted(means,means[column]-required,side='right')-1)
    right=int(np.searchsorted(means,means[column]+required,side='left'))
    if left<0 or right>=len(means):raise ValueError('physical source columns do not bracket the required bulk directions; no ghost x columns or fallback')
    cfg={'study':'one preselected mass-average bulk moment cone, not the 64-cell closed operator',
      'source':str(source),'physical_parameters':p,'time':t,'preselected_column':column,
      'selection_reason':'same column as previous MPFA sign counterexample (initial cell 16, observed cell 48)',
      'original_cells_touching_boundary':int(boundary_cells.sum()),'context':'explicit_virtual_transverse_extension',
      'virtual_extension':'translate the known material strip by +/- one transverse cell; diagnostic only, no change to original m/c0/boundary',
      'bulk_equations':'linear and quadratic exact mass-average moments including covariance differences and variable-density drift',
      'coordinates':'corotated physical axes of prescribed benchmark, no inverse-shear diffusion replacement or F/C fit',
      'supports':{'narrow':[column-1,column+1],'geometry_selected_wide':[left,right]},
      'wide_support_rule':'first source centroid on each side bracketing hy*abs(shear-Gamma_xy/Gamma_yy), no LP-based support scan',
      'hy':hy,'shear':shear,'cross_moment_required_bracketing_distance':float(required),
      'lp':'zero objective, default HiGHS tolerances; witnesses/certificates only, never used for propagation',
      'physical_acceptance':False,'fluid_orc_enabled':False,'global_conservative_operator_constructed':False,
      'original_closed_boundary_changed':False,'candidate_parameter_scan':False}
    output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    direct32=independent_boundary_targets(flow,edges,column,means[column],t,D,shear,hy,32)
    direct64=independent_boundary_targets(flow,edges,column,means[column],t,D,shear,hy,64)
    rows={};saved={'mass_edges':edges,'original_mass':mass,'original_c0':c0,'original_positions':positions,
        'original_mass_centroids':old_centers,'source_boundary_cell_mask':boundary_cells,
        'stream_mass_means':means,'stream_mass_variances':var,'stream_mass_means_order32':means32,
        'stream_mass_variances_order32':var32,'target':target,'target_beta':beta,'target_Gamma':Gamma,
        'direct_beta_order32':direct32[0],'direct_Gamma_order32':direct32[1],'direct_beta_order64':direct64[0],'direct_Gamma_order64':direct64[1]}
    for name,(lo,hi) in cfg['supports'].items():
        A,arrays=build_support(means,var,column,range(lo,hi+1),shear,hy)
        report,rates=solve_moment_cone(A,target,hy);rows[name]=report;saved[name+'_A']=A
        for key,value in arrays.items():saved[name+'_'+key]=value
        if rates is not None:
            saved[name+'_raw_rates']=rates
            report['mass_covariance_xx_contribution']=float(rates@arrays['covariance_difference'][:,0,0])
            report['centroid_displacement_xx_contribution']=float(rates@(arrays['displacements'][:,0]**2))
        print(name,'status',report['status'],'separator',report['cross_ratio_separator']['verified'],
            'exact basis',report.get('exact_basis_certificate',{}).get('verified'),flush=True)
    corotated=from_world(old_centers[None,...],np.array([t]),p['T'])[0]
    result={'supports':rows,'density_moment_J':J,'target_drift':beta.tolist(),'target_centered_quadratic_derivative':Gamma.tolist(),
        'target_cross_transverse_ratio':float(ratio),
        'stream_mass_centroid_32_64_max_difference':float(np.max(abs(means-means32))),
        'stream_mass_variance_32_64_max_difference':float(np.max(abs(var-var32))),
        'stream_centroid_vs_saved_source_max_difference':float(np.max(abs(corotated[:32,0]-means))),
        'primitive_vs_independent_boundary_beta_max_difference':float(np.max(abs(beta-direct64[0]))),
        'primitive_vs_independent_boundary_Gamma_max_difference':float(np.max(abs(Gamma-direct64[1]))),
        'boundary_quadrature_32_64_beta_max_difference':float(np.max(abs(direct32[0]-direct64[0]))),
        'boundary_quadrature_32_64_Gamma_max_difference':float(np.max(abs(direct32[1]-direct64[1]))),
        'physical_acceptance':False,'fluid_orc_enabled':False,'global_conservative_operator_constructed':False,
        'scope':'one explicitly extended bulk column; feasibility does not test the original closed boundary or all cells',
        'rates_used_for_evolution':False,'new_diffusion_backend':False}
    np.savez_compressed(output/'mass-moment-cones.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
