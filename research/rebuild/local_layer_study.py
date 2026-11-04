"""A priori static local refinement of the same scalar-step contact problem."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from contact import Contact,METHODS,semidiscrete,backward_euler
from contact_step_reference import StepReference
from local_layer_mesh import locally_refined_faces
from subcell_control import aggregation_and_lift
from spatial_study import normalized_error


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    manifest=json.loads((source/'sha256.json').read_text())
    for path,h in manifest.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:raise RuntimeError('source evidence mismatch: '+path)
    src_cfg=json.loads((source/'config.json').read_text());physical=src_cfg['physical_configuration'];times=src_cfg['times']
    baseline=json.loads((source/'results.json').read_text())
    f=Contact(rho_left=physical['rho_left'],rho_right=physical['rho_right'],D=physical['D'],half_length=physical['half_length'],speed=physical['speed'])
    ref=StepReference(f,left=physical['initial_left'],right=physical['initial_right'])
    with np.load(source/'partition-parent64.npz') as z:parent=z['parent_faces'];m=z['parent_mass'];c0=z['parent_c0']
    parent_refs={}
    for t in times:
        with np.load(source/f'result-parent64-t{t:g}.npz') as z:parent_refs[t]=z['aggregated_reference']
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'fixed contact problem, a priori locally refined material meshes',
            'source':str(source),'original_parent_cells':64,'physical_configuration':physical,'times':times,
            'resolutions':[2,4,8],'growth':.2,'minimum_target_time':min(times),
            'mesh_monitor':'h(a)=(sqrt(D*t_min)+growth*|a|)/resolution; every original face is retained',
            'geometry_rule':'insert exact contact, subdivide each original material interval before simulation; no runtime adaptation',
            'common_backend':'ordinary_volume_center; all pure-phase backend conductances compared',
            'main_time_evolution':'same semidiscrete dense exponential as earlier studies',
            'independent_check':'mass-symmetric tridiagonal eigen evolution; absolute concentration tolerance 1e-10',
            'BE_time_check':'finest level, t_min and t_max, 512/1024 steps',
            'fluid_orc_enabled':False,'third_stage_complete':False,
            'scope_limit':'known oracle contact and scalar-step reference; no general interface detector or physical parameter scan'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    meshes=[];rows=[]
    for q in config['resolutions']:
        faces=locally_refined_faces(parent,f.D,min(times),q,config['growth']);ms=f.mass(faces);cs0=ref.average(faces,0.)
        A,B,index=aggregation_and_lift(parent,faces,m,ms)
        widths=np.diff(faces);adjacent=(faces[:-1]<=0)&(faces[1:]>=0)
        contact_width_max=float(widths[adjacent].max());layer=np.sqrt(f.D*min(times));ratio=layer/contact_width_max
        if ratio<q*(1-1e-12):raise RuntimeError('planned layer geometry not met')
        g=f.conductance(faces,'ordinary_volume_center')
        gdiff={method:float(np.max(abs(f.conductance(faces,method)-g))/g.max()) for method in METHODS}
        if max(gdiff.values())>1e-10:raise RuntimeError('pure-phase backend disagreement')
        initerr=float(np.max(abs(A@cs0-c0)))
        if initerr>1e-12:raise RuntimeError('parent initial averages changed')
        meshes.append({'resolution':q,'cells':len(ms),'minimum_width':float(widths.min()),'maximum_width':float(widths.max()),
                       'maximum_contact_adjacent_width':contact_width_max,'earliest_layer_to_contact_width':ratio,
                       'total_mass':float(ms.sum()),'initial_scalar_mass':float(ms@cs0),
                       'initial_right_inventory':float(ms[faces[:-1]>=0]@cs0[faces[:-1]>=0]),
                       'initial_half_variance':float(.5*ms@((cs0-ref.mean)**2)),
                       'parent_initial_average_max_residual':initerr,'backend_relative_differences':gdiff})
        np.savez_compressed(output/f'mesh-q{q}.npz',faces=faces,child_mass=ms,child_c0=cs0,parent_faces=parent,parent_mass=m,parent_c0=c0,A=A,B=B,parent_index=index,g=g)
        dL=np.zeros(len(ms));dL[:-1]+=g;dL[1:]+=g
        eigenvalues,U=eigh_tridiagonal(dL/ms,-g/np.sqrt(ms[:-1]*ms[1:]))
        sqrtm=np.sqrt(ms);right=faces[:-1]>=0;previous_energy=float(.5*ms@((cs0-ref.mean)**2))
        for t in times:
            c,P,diag=semidiscrete(cs0,ms,g,t)
            check=(U@(np.exp(-t*eigenvalues)*(U.T@(sqrtm*cs0))))/sqrtm
            independent=float(np.max(abs(check-c)))
            if independent>1e-10:raise RuntimeError('independent evolution disagreement')
            K=ref.terms_for(t);exact=ref.average(faces,t,K);exact2=ref.average(faces,t,2*K)
            inventory=float(ms[right]@c[right]);inv_ref,tail,images=ref.right_inventory(0.,t)
            energy=float(.5*ms@((c-ref.mean)**2))
            if energy>previous_energy+1e-12:raise RuntimeError('observed energy increase')
            previous_energy=energy
            aggregation=float(np.max(abs(A@exact-parent_refs[t])))
            if aggregation>1e-12:raise RuntimeError('reference aggregation disagreement')
            row={'resolution':q,'cells':len(ms),'time':t,
                 'aggregated_error':normalized_error(m,c0,A@c-parent_refs[t]),
                 'full_resolution_error':normalized_error(ms,cs0,c-exact),
                 'right_inventory':inventory,'right_inventory_reference':inv_ref,
                 'right_inventory_signed_relative_error':float((inventory-inv_ref)/inv_ref),
                 'right_inventory_absolute_relative_error':float(abs(inventory-inv_ref)/inv_ref),
                 'semidiscrete_diagnostics':diag,'independent_evolution_max_abs_difference':independent,
                 'relative_mass_drift':abs(float(ms@(c-cs0)))/abs(float(ms@cs0)),
                 'observed_half_variance':energy,'reference_aggregation_max_residual':aggregation,
                 'reference_K_to_2K_difference':normalized_error(ms,cs0,exact-exact2),
                 'earliest_layer_to_contact_width':ratio}
            arr={'time':np.array(t),'child_solution':c,'child_reference':exact,'child_mass':ms,'child_c0':cs0,
                 'aggregated_solution':A@c,'parent_reference':parent_refs[t],'parent_mass':m,'parent_c0':c0}
            if q==8 and t in (min(times),max(times)):
                be={}
                for steps in (512,1024):
                    cn,d=backward_euler(cs0,ms,g,t,steps)
                    actual=float(ms[right]@cn[right]);arr[f'c_BE{steps}']=cn
                    be[str(steps)]={'time_error_to_semidiscrete':normalized_error(ms,cs0,cn-c),
                                    'aggregated_time_error':normalized_error(m,c0,A@(cn-c)),
                                    'aggregated_total_error':normalized_error(m,c0,A@cn-parent_refs[t]),
                                    'right_inventory_time_error':float((actual-inventory)/inv_ref),
                                    'right_inventory_total_relative_error':float((actual-inv_ref)/inv_ref),**d}
                row['BE_check']=be
            rows.append(row);np.savez_compressed(output/f'result-q{q}-t{t:g}.npz',**arr)
            print('q',q,'cells',len(ms),'time',t,'right error',row['right_inventory_absolute_relative_error'],'parent error',row['aggregated_error'],flush=True)
    old=[x for x in baseline['runs'] if x['parent_cells']==64]
    results={'mesh_diagnostics':meshes,'runs':rows,'previous_65_child_runs':old,
             'third_stage_physical_acceptance':False,
             'warning':'geometry-based local sensitivity, no universal convergence order or all-task readiness claim'}
    (output/'results.json').write_text(json.dumps(results,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                                                    'BLAS_threads_command':'4'},indent=2))
    files=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
