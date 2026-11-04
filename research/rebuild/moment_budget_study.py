"""One diagnostic error envelope on the unchanged 128-cell shared graph."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from kernel_joint_study import verify
from shared_exchange_feasibility import conservative_graph
from moment_defect_budget import budget_problem,solve_budget


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    cfgold=json.loads((source/'config.json').read_text());p=cfgold['physical_parameters']
    with np.load(source/'shared-exchange-system.npz') as z:
        data={k:z[k] for k in ('pairs','mass','c0','centroids','covariance','quads','columns','rows','boundary_mask','interior_cells','targets_per_interior_cell','drift_margins','moment_rhs_mass_scaled','A_data','A_indices','A_indptr','A_shape')}
    A=csr_matrix((data['A_data'],data['A_indices'],data['A_indptr']),shape=tuple(data['A_shape']));b=data['moment_rhs_mass_scaled']
    problem=budget_problem(A,b,data['mass'],data['interior_cells'],p['D'],cfgold['length_unit_m'])
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one fixed-grid diagnostic minimax moment-defect budget, not a diffusion backend',
        'source':str(source),'physical_parameters':p,'time':cfgold['time'],'cells':len(data['mass']),
        'conductance_variables':A.shape[1],'first_moments':'all 120 exact constraints retained',
        'second_moments':'all 180 xx/xy/yy physical targets retained; measure maximum signed absolute residual against fixed 2D',
        'norm_scope':'entrywise maximum in inherited corotated physical axes; axes are not optimized, norm is not rotation invariant',
        'solver':'one HiGHS solve, default tolerances; equation unit conversions retained separately',
        'boundary_scope':'372 boundary-only pairs remain unidentified by interior moments; no physical boundary or face acceptance',
        'physical_acceptance':False,'fluid_orc_enabled':False,'mesh_or_parameter_search':False,
        'physical_targets_or_thresholds_changed':False,'rates_used_for_evolution':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report,arrays=solve_budget(problem)
    result={'budget_solver':report,'physical_acceptance':False,'fluid_orc_enabled':False,'new_transient_run':False,
        'physical_boundary_or_face_acceptance':False,'diagnostic_weights_are_not_a_selected_discretization':True}
    saved={**data,**arrays,'budget_second_scale':problem['scale'],'budget_first_equation_units':problem['eq_units']}
    if report['solver_reports_feasible']:
        g=arrays['raw_conductances'];defect=arrays['second_component_defect_relative_to_2D'].reshape(-1,3)
        drift=(arrays['first_residual_mass_scaled']/np.repeat(data['mass'][data['interior_cells']],2)).reshape(-1,2)
        k,f=np.unravel_index(np.argmax(abs(defect)),defect.shape);cell=int(data['interior_cells'][k]);component=['xx','xy','yy'][f]
        result.update({'worst_cell':cell,'worst_column':int(data['columns'][cell]),'worst_component':component,
            'worst_signed_relative_defect':float(defect[k,f]),'max_physical_first_moment_residual_m_per_s':float(np.max(abs(drift)))})
        tensor=np.zeros((len(defect),2,2));tensor[:,0,0]=defect[:,0]*2*p['D'];tensor[:,0,1]=tensor[:,1,0]=defect[:,1]*2*p['D'];tensor[:,1,1]=defect[:,2]*2*p['D']
        eigen=np.linalg.eigvalsh(tensor);result['minimum_defect_tensor_eigenvalue_m2_per_s']=float(eigen.min())
        result['second_defect_is_assumed_positive_semidefinite']=False
        saved['second_defect_tensor']=tensor;saved['second_defect_eigenvalues']=eigen
        if np.any(g<0):result['diagnostic_graph_audit']={'available':False,'reason':'raw negative rate retained; no clipping'}
        else:
            L,Q=conservative_graph(data['pairs'],g,data['mass']);off=Q.copy();np.fill_diagonal(off,np.inf)
            result['diagnostic_graph_audit']={'available':True,'max_mass_residual':float(np.max(abs(data['mass']@Q))),
                'max_constant_residual':float(np.max(abs(Q.sum(axis=1)))),'weighted_symmetry_defect':float(np.max(abs(L-L.T))),
                'min_raw_offdiagonal_rate':float(off.min()),'physical_acceptance':False}
            saved['diagnostic_L']=L;saved['diagnostic_Q']=Q
            # Independently integrated mass-average polynomials from the unchanged parent evidence.
            parent=root/'runs'/Path(cfgold['source']).name
            with np.load(parent/'half-transverse-cone.npz') as z:
                phi32=z['direct_poly_order32'];phi64=z['direct_poly_order64'];origin=z['origin'];axes=z['axes']
            relative=(data['centroids']-origin)@axes
            def centered_action(phi):
                v=Q@phi;v[:,2]-=2*relative[:,0]*v[:,0];v[:,3]-=relative[:,0]*v[:,1]+relative[:,1]*v[:,0];v[:,4]-=2*relative[:,1]*v[:,1]
                return v
            action32=centered_action(phi32);action64=centered_action(phi64);model=(A@g).reshape(-1,5)/data['mass'][data['interior_cells'],None]
            result['independent_mass_average_action_vs_assembly_max_abs_difference']=float(np.max(abs(action64[data['interior_cells']]-model)))
            result['independent_mass_average_action_32_64_max_abs_difference']=float(np.max(abs(action64[data['interior_cells']]-action32[data['interior_cells']])))
            saved['independent_mass_average_action_order32']=action32;saved['independent_mass_average_action_order64']=action64
    np.savez_compressed(output/'moment-budget.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
