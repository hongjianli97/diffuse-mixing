"""One saved CSR row on one fixed material-average probe; read only."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
from audit_contact_constitutive import digest
from package_compression_midpoint import dump
from point_row_response import row_response,signed_budget


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-ideal-row'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():assert digest(root/rel)==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_saved_point_row_review_predeclared']
    assert (protocol['target_index'],protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'])==(31,128,.04,.02)
    assert protocol['no_new_edge_weights_or_full_matrix'] and protocol['no_fitting_clipping_renormalization_or_threshold_changes']
    assert not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    path=root/protocol['source_graph'];assert digest(path)==protocol['source_graph_sha256']
    target=31;ell=1/128;h=.04;D=.02
    with np.load(path) as z:
        shape=z['L_shape'];ptr=z['L_indptr'];start,stop=map(int,ptr[target:target+2])
        # NPZ storage is decompressed; no new sparse/full matrix is constructed.
        values=z['L_data'][start:stop];cols=z['L_indices'][start:stop]
        saved={key:z[key][cols] for key in ['mass','ids','c0','material_label_centers','rho0_point','quadrature_area','true_cell_area_reference_only','reference_rate','kernel_rate']}
        input_dtype={key:str(value.dtype) for key,value in saved.items()}
    assert np.array_equal(shape,[128**2,128**2])
    at=int(np.flatnonzero(cols==target)[0]);mass=float(saved['mass'][at]);center=saved['material_label_centers'][at]
    assert int(saved['ids'][at])==protocol['target_id'] and center[0]==63/256 and np.all(saved['rho0_point']==1.)
    np.testing.assert_array_equal(saved['mass']/saved['rho0_point'],saved['quadrature_area'])
    np.testing.assert_array_equal(saved['quadrature_area'],saved['true_cell_area_reference_only'])
    np.testing.assert_array_equal(saved['quadrature_area'],ell**2)
    dx=saved['material_label_centers']-center;dx-=np.rint(dx)
    assert np.all(np.linalg.norm(dx,axis=1)<=2*h)
    arrays,metrics=row_response(values,cols,saved['material_label_centers'][:,0],target,mass,ell)
    arrays.update({'original_'+key:value for key,value in saved.items()});arrays['periodic_row_displacement']=dx
    ideal_path=root/protocol['saved_ideal_response_source'];ideal_result=json.loads(ideal_path.read_text())
    ideal=ideal_result['rules']['fine']['raw_direct_aggregate_rate_per_s']
    point=metrics['raw_CSR_quadratic_response_per_s'];budget=signed_budget(point,ideal,D)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'scope':'predeclared read-only saved point-candidate row 31 comparison',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'source_graph_sha256':digest(path),'source_ideal_results_sha256':digest(ideal_path),
        'CSR_shape':shape.tolist(),'CSR_source_row_start_stop':[start,stop],'source_row_columns_dtype':str(cols.dtype),
        'source_row_values_dtype':str(values.dtype),'original_array_dtypes':input_dtype,
        'target_index':target,'target_id':int(saved['ids'][at]),'target_mass_kg':mass,'original_c0_at_target':float(saved['c0'][at]),
        'probe':'auxiliary exact square-cell mass means of c=a^2/2, never point values or replacement original c0',
        'no_matrix_assembly_edge_recomputation_or_row_repair':True,'new_candidate_graphs':0,'fluid_orc_enabled':False})
    result={'row':metrics,'signed_budget':budget,'raw_point_response_per_s':point,'saved_ideal_direct_response_per_s':ideal,
        'PDE_unit_curvature_response_per_s':D,'point_relative_error_percent':100*budget['point_minus_PDE_relative_to_PDE'],
        'point_relative_error_ppm':1e6*budget['point_minus_PDE_relative_to_PDE'],
        'all_saved_row_points_same_pure_phase':True,'saved_point_quadrature_area_equals_original_square_area':True,
        'maximum_saved_row_neighbor_distance_m':float(np.max(np.linalg.norm(dx,axis=1))),
        'raw_CSR_value_selected_before_inspection':True,'original_c0_saved_cosine_PDE_and_candidate_unchanged':True,
        'saved_point_row_evaluations':1,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one pure-phase row and one auxiliary normal quadratic only','not interface or general geometry acceptance',
            'signed cancellation depends on the specified ideal intermediate, not unique causal fractions or a cosine-error diagnosis']}
    np.savez_compressed(out/'point-row.npz',**arrays);dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
