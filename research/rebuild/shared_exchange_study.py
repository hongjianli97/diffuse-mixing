"""One fixed 128-cell coupled system, with an embedded row-obstruction proof."""
from pathlib import Path
from fractions import Fraction
import argparse,json,platform
import numpy as np
import scipy
from becker import Becker
from kernel_joint_study import verify
from mass_cone_study import primitive_targets,independent_boundary_targets
from shared_exchange_feasibility import fixed_logical_pairs,pair_features,assemble_shared_moments,solve_shared_moments,exact_sparse_separator


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters'];t=old['time'];hy=old['refined_hy'];shear=.6
    moment_source=root/'runs'/Path(old['source']).name
    with np.load(moment_source/'mass-moment-cones.npz') as z:means=z['stream_mass_means'];variances=z['stream_mass_variances'];edges=z['mass_edges']
    with np.load(source/'half-transverse-cone.npz') as z:
        mass=z['child_record_mass'];c0=z['child_record_c0'];centers=z['child_record_centroids'];cov=z['child_record_covariance']
        quads=z['child_record_quads'];columns=z['child_column'];rows=z['child_row'];boundary=z['child_boundary_mask'];axes=z['axes']
        previous_rates=z['raw_rates'];previous_neighbours=z['neighbour_child_indices'];previous_child=int(z['selected_child_index'])
    inside=np.flatnonzero(~boundary);pairs=fixed_logical_pairs(columns,rows,6,1)
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    targets=[]
    for cell in inside:
        beta,Gamma,J=primitive_targets(flow,edges,means,int(columns[cell]),t,p['D'],shear)
        targets.append([*beta,Gamma[0,0],Gamma[0,1],Gamma[1,1]])
    targets=np.array(targets);margins=targets[:,4]-hy*abs(targets[:,1]);worst=int(np.argmin(margins));cell=int(inside[worst])
    A,b=assemble_shared_moments(pairs,columns,rows,means,variances,mass,inside,targets,hy,shear)
    cfg={'study':'one fixed coupled nonnegative shared-mass-exchange feasibility system',
      'source':str(source),'physical_parameters':p,'time':t,'cells':len(mass),'interior_cells':len(inside),
      'graph_rule':'one variable per unordered physical-cell pair, abs(column difference)<=6 and abs(row difference)<=1; no outside nodes',
      'pair_variables':len(pairs),'equations':len(b),'transverse_spacing_m':hy,
      'conductance_rule':'g_ij=g_ji>=0 imposed in the variables, never by averaging independent row weights',
      'bulk_constraints':'five exact linear/centered-quadratic mass-average moments at every actual interior cell; mass-scaled right hand side',
      'boundary_scope':'no moment equations on boundary cells, no external graph exchange; this does not certify the physical Neumann closure',
      'physical_face_flux_scope':'shared graph pairs only, not a recovered physical shared-face FV flux',
      'chosen_separator_cell':cell,'separator_selection':'minimum necessary drift margin over the fixed 60 interior equations, no parameter scan',
      'length_unit_m':old['solver_length_unit_m'],'solver':'zero objective, default HiGHS tolerances',
      'physical_acceptance':False,'fluid_orc_enabled':False,'parameter_or_mesh_search':False,'original_64_cell_baseline_changed':False}
    output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report,g=solve_shared_moments(A,b,old['solver_length_unit_m'])
    sign=1 if targets[worst,1]>=0 else -1
    dual={5*worst+1:-sign*Fraction.from_float(float(hy)),5*worst+4:Fraction(1)}
    proof=exact_sparse_separator(A,b,dual)
    # Retain the old valid row as a matched diagnostic embedding, not a candidate.
    lookup={tuple(e):i for i,e in enumerate(pairs.tolist())};embedding=np.zeros(len(pairs))
    for j,r in zip(previous_neighbours,previous_rates):embedding[lookup[tuple(sorted((previous_child,int(j))))]]=mass[previous_child]*r
    residual=(A@embedding-b).reshape(-1,5);previous_row=int(np.flatnonzero(inside==previous_child).item())
    direct=[]
    for order in (32,64):
        beta,Gamma=independent_boundary_targets(flow,edges,int(columns[cell]),means[columns[cell]],t,p['D'],shear,hy,order)
        target=np.array([*beta,Gamma[0,0],Gamma[0,1],Gamma[1,1]])
        direct.append(target)
    direct=np.array(direct);direct_margins=direct[:,4]-hy*abs(direct[:,1])
    # Compare the analytic feature model with the inherited true frozen geometry.
    d=(centers[pairs[:,1]]-centers[pairs[:,0]])@axes
    Sxx=np.einsum('a,cab,b->c',axes[:,0],cov,axes[:,0]);Sxy=np.einsum('a,cab,b->c',axes[:,0],cov,axes[:,1]);Syy=np.einsum('a,cab,b->c',axes[:,1],cov,axes[:,1])
    i,j=pairs.T;actual=np.column_stack((d,d[:,0]**2+Sxx[j]-Sxx[i],d[:,0]*d[:,1]+Sxy[j]-Sxy[i],d[:,1]**2+Syy[j]-Syy[i]))
    model=pair_features(i,j,columns,rows,means,variances,hy,shear)
    result={'coupled_solver':report,'exact_embedded_separator':proof,'actual_interior_cells':inside.tolist(),
      'drift_obstruction_cells':inside[margins<0].tolist(),'drift_obstruction_columns':sorted(set(columns[inside[margins<0]].tolist())),
      'worst_cell':cell,'worst_column':int(columns[cell]),'worst_row':int(rows[cell]),
      'worst_beta_y':float(targets[worst,1]),'worst_Gamma_yy':float(targets[worst,4]),
      'worst_unscaled_drift_margin_m2_per_s':float(margins[worst]),
      'maximum_beta_y_over_interior':float(np.max(abs(targets[:,1]))),
      'previous_child':previous_child,'previous_single_row_drift_margin':float(margins[previous_row]),
      'previous_row_embedding_mass_scaled_residual_max':float(np.max(abs(residual[previous_row]))),
      'previous_embedding_full_system_mass_scaled_residual_max':float(np.max(abs(residual))),
      'worst_target_vs_independent_boundary_max_abs_difference':float(np.max(abs(targets[worst]-direct[1]))),
      'boundary_target_32_64_max_abs_difference':float(np.max(abs(direct[0]-direct[1]))),
      'independent_boundary_drift_margins':direct_margins.tolist(),
      'analytic_pair_features_vs_saved_geometry_max_abs_difference':float(np.max(abs(model-actual))),
      'unconstrained_boundary_only_pair_columns':int(np.count_nonzero(np.diff(A.tocsc().indptr)==0)),
      'shared_global_conductance_witness_found':g is not None,'accepted_global_operator_constructed':False,
      'rates_used_for_evolution':False,'physical_boundary_or_face_acceptance':False,'physical_acceptance':False,'fluid_orc_enabled':False,
      'scope':'failure of the prescribed exact bulk moments and nonnegative linear exchange on this fixed partition, not a failure theorem for all convergent or boundary-auxiliary schemes'}
    # beta_y may vanish in flat cells: only nonzero drifts constrain h from above.
    nonzero=abs(targets[:,1])>0
    result['necessary_uniform_h_upper_m']=float(np.min(targets[nonzero,4]/abs(targets[nonzero,1])))
    saved={'pairs':pairs,'mass':mass,'c0':c0,'centroids':centers,'covariance':cov,'quads':quads,'columns':columns,'rows':rows,
      'boundary_mask':boundary,'interior_cells':inside,'targets_per_interior_cell':targets,'drift_margins':margins,
      'moment_rhs_mass_scaled':b,'A_data':A.data,'A_indices':A.indices,'A_indptr':A.indptr,'A_shape':np.array(A.shape),
      'single_row_embedding_not_candidate':embedding,'single_row_embedding_residual':residual,
      'independent_boundary_targets_worst_order32_64':direct,'actual_pair_features':actual,'analytic_pair_features':model,
      'certificate_dual_rows':np.array(list(dual)),'certificate_dual_values':np.array([float(v) for v in dual.values()])}
    if g is not None:saved['raw_shared_conductance_witness']=g
    np.savez_compressed(output/'shared-exchange-system.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
