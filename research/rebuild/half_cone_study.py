"""One predeclared transverse halving; fixed old stream support, no scan."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy
from becker import Becker
from becker_polygon_benchmark import quad_connectivity
from kernel_joint_study import verify
from material_polygon_history import MaterialPolygonHistory,signed_area
from mass_cone_study import build_support,independent_boundary_targets
from mass_moment_cone import require_bulk_context,solve_moment_cone,transverse_drift_separator
from transverse_partition import split_transverse_parallelograms,logical_child_layout,neighbour_child_indices,polynomial_averages,direct_mass_polynomials


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters'];t=old['time'];column=old['preselected_column']
    hy=old['hy']/2;shear=old['shear'];selected_child=2*column+1;lo,hi=old['supports']['geometry_selected_wide']
    history=MaterialPolygonHistory.load(root/'runs/rebuild-stage3-polygon-transport/polygon-record-32.npz')
    if not np.array_equal(history.cell_vertex_indices,quad_connectivity(32,2)):raise ValueError('source logical order differs; no remapping by guess')
    with np.load(source/'mass-moment-cones.npz') as z:
        mass=z['original_mass'];c0=z['original_c0'];positions=z['original_positions'];centers=z['original_mass_centroids']
        edges=z['mass_edges'];means=z['stream_mass_means'];variances=z['stream_mass_variances'];target=z['target']
    quads=positions[history.cell_vertex_indices]
    children=split_transverse_parallelograms(quads,mass,c0,centers,np.tile(variances,2),transverse_mass_uniform=True,transverse_initial_state_uniform=True,parent_ids=history.cell_ids)
    columns,rows,boundary=logical_child_layout();require_bulk_context(bool(boundary[selected_child]),'actual_interior')
    A,support=build_support(means,variances,column,range(lo,hi+1),shear,hy)
    neighbours=neighbour_child_indices(support['neighbour_column_and_virtual_row'],selected_child)
    origin=children['centroids'][selected_child];s=children['stream_unit_vector'][column];axes=np.column_stack((s,[-s[1],s[0]]))
    flow=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'one transverse material halving, one actual interior child, necessary mass-average row feasibility only',
      'source':str(source),'physical_parameters':p,'time':t,'source_cells':len(mass),'derived_children':len(children['mass']),
      'parent_column':column,'selected_parent':column,'selected_half_slot':1,'selected_child_index':selected_child,
      'selected_logical_refined_row':int(rows[selected_child]),'original_hy':old['hy'],'refined_hy':hy,
      'stream_columns':[lo,hi],'neighbours':len(neighbours),'stream_support_changed':False,
      'child_mass_rule':'exact physical half of each source material mass, justified by transverse-uniform benchmark density',
      'child_initial_average_rule':'same source c0, initially independent of transverse label',
      'context':'actual interior child of an explicitly separate frozen 128-cell partition; all neighbours inside original domain',
      'virtual_neighbours':0,'original_64_cell_baseline_changed':False,
      'target_rule':'reuse the exact stored five physical targets; independent half-cell boundary quadrature verifies invariance',
      'solver_length_unit_m':old['hy'],'solver_rule':'same zero objective and default tolerances; no support/spacing search, no clipped rates',
      'physical_acceptance':False,'global_operator_constructed':False,'rates_used_for_evolution':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report,rates=solve_moment_cone(A,target,old['hy']);drift=transverse_drift_separator(A,target)
    parent_poly=polynomial_averages(centers,children['parent_covariance'],origin,axes)
    child_poly=polynomial_averages(children['centroids'],children['covariance'],origin,axes)
    restricted=child_poly.reshape(len(mass),2,5).mean(axis=1)
    offsets=children['centroid_offsets'].reshape(len(mass),2,2)
    total_covariance=(children['covariance'].reshape(len(mass),2,2,2)+offsets[:,:,:,None]*offsets[:,:,None,:]).mean(axis=1)
    actual32,poly32=direct_mass_polynomials(flow,children['quads'],children['mass'],t,origin,axes,p['T'],32)
    actual64,poly64=direct_mass_polynomials(flow,children['quads'],children['mass'],t,origin,axes,p['T'],64)
    beta32,G32=independent_boundary_targets(flow,edges,column,means[column],t,p['D'],shear,hy,32)
    beta64,G64=independent_boundary_targets(flow,edges,column,means[column],t,p['D'],shear,hy,64)
    direct_target=np.array([*beta64,G64[0,0],G64[0,1],G64[1,1]])
    direct_A=(poly64[neighbours]-poly64[selected_child]).T
    result={'cone':report,'drift_necessary_condition':drift,'actual_interior_children':int(np.count_nonzero(~boundary)),
      'selected_child_touches_boundary':bool(boundary[selected_child]),'all_neighbours_in_refined_partition':True,
      'mass_restriction_max_abs_error':float(np.max(abs(children['mass'].reshape(len(mass),2).sum(axis=1)-mass))),
      'component_mass_restriction_max_abs_error':float(np.max(abs((children['mass']*children['c0']).reshape(len(mass),2).sum(axis=1)-mass*c0))),
      'centroid_restriction_max_abs_error':float(np.max(abs(children['centroids'].reshape(len(mass),2,2).mean(axis=1)-centers))),
      'quadratic_state_restriction_max_abs_error':float(np.max(abs(restricted-parent_poly))),
      'law_of_total_covariance_max_abs_error':float(np.max(abs(total_covariance-children['parent_covariance']))),
      'area_partition_max_abs_error':float(np.max(abs(signed_area(children['quads']).reshape(len(mass),2).sum(axis=1)-signed_area(quads)))),
      'child_mass_vs_independent_density_integral_max_relative_error':float(np.max(abs(actual64-children['mass'])/children['mass'])),
      'density_mass_32_64_max_relative_difference':float(np.max(abs(actual64-actual32)/children['mass'])),
      'polynomial_mass_means_32_64_max_abs_difference':float(np.max(abs(poly64-poly32))),
      'analytic_A_vs_independent_mass_average_A_max_abs_difference':float(np.max(abs(A-direct_A))),
      'old_target_vs_half_cell_boundary_beta_max_abs_difference':float(np.max(abs(target[:2]-beta64))),
      'old_target_vs_half_cell_boundary_Gamma_max_abs_difference':float(np.max(abs(target[2:]-direct_target[2:]))),
      'boundary_target_32_64_max_abs_difference':float(max(np.max(abs(beta32-beta64)),np.max(abs(G32-G64)))),
      'physical_acceptance':False,'global_operator_constructed':False,'rates_used_for_evolution':False,'fluid_orc_enabled':False}
    saved={'original_mass':mass,'original_c0':c0,'original_positions':positions,'original_mass_centroids':centers,'mass_edges':edges,
        'A':A,'target':target,'neighbour_child_indices':neighbours,'selected_child_index':selected_child,'axes':axes,'origin':origin,
        'child_column':columns,'child_row':rows,'child_boundary_mask':boundary,'parent_poly':parent_poly,'child_poly':child_poly,
        'direct_actual_mass_order32':actual32,'direct_actual_mass_order64':actual64,'direct_poly_order32':poly32,'direct_poly_order64':poly64,
        'direct_A_order64':direct_A,'direct_beta_order32':beta32,'direct_beta_order64':beta64,'direct_Gamma_order32':G32,'direct_Gamma_order64':G64}
    for key,value in children.items():saved['child_record_'+key]=value
    for key,value in support.items():saved['support_'+key]=value
    if rates is not None:
        saved['raw_rates']=rates;result['raw_rate_action_on_independent_mass_average_features']=(direct_A@rates).tolist()
        result['raw_rate_action_vs_independent_boundary_target_max_abs_error']=float(np.max(abs(direct_A@rates-direct_target)))
        result['active_neighbour_children']=neighbours[np.flatnonzero(rates!=0)].tolist()
    np.savez_compressed(output/'half-transverse-cone.npz',**saved)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
