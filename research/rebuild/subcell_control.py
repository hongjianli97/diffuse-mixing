"""Fixed initial material subcells: retained information vs operator change.

No runtime remapping, scalar reconstruction fitting, clipping or mass repair.
"""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import scipy
from contact import Contact,METHODS,semidiscrete,laplacian,backward_euler
from contact_step_reference import StepReference
from spatial_study import normalized_error


def split_faces_at_contact(faces):
    faces=np.asarray(faces,dtype=float)
    if faces.ndim!=1 or len(faces)<3 or not np.isfinite(faces).all() or np.any(np.diff(faces)<=0):
        raise ValueError('strictly increasing finite material faces required')
    cut=np.flatnonzero((faces[:-1]<0)&(faces[1:]>0))
    if len(cut)!=1:raise ValueError('exactly one cell must straddle contact')
    return np.insert(faces,int(cut[0])+1,0.)


def aggregation_and_lift(coarse_faces,fine_faces,coarse_mass,fine_mass):
    coarse_faces=np.asarray(coarse_faces);fine_faces=np.asarray(fine_faces)
    coarse_mass=np.asarray(coarse_mass);fine_mass=np.asarray(fine_mass)
    if len(coarse_mass)!=len(coarse_faces)-1 or len(fine_mass)!=len(fine_faces)-1:
        raise ValueError('mass/face dimensions differ')
    if np.any(coarse_mass<=0) or np.any(fine_mass<=0):raise ValueError('positive masses required')
    if coarse_faces[0]!=fine_faces[0] or coarse_faces[-1]!=fine_faces[-1] or not np.all(np.isin(coarse_faces,fine_faces)):
        raise ValueError('subcells must partition exactly the original material intervals')
    mids=(fine_faces[:-1]+fine_faces[1:])/2
    parent=np.searchsorted(coarse_faces,mids,side='right')-1
    A=np.zeros((len(coarse_mass),len(fine_mass)));B=np.zeros(A.T.shape)
    A[parent,np.arange(len(fine_mass))]=fine_mass/coarse_mass[parent]
    B[np.arange(len(fine_mass)),parent]=1.
    if np.max(abs(np.bincount(parent,weights=fine_mass,minlength=len(coarse_mass))-coarse_mass))>1e-13:
        raise ValueError('subcell masses do not sum to parent masses')
    return A,B,parent


def compare(A,B,P,child_initial,coarse_initial,coarse_solution,mass):
    initial_hidden=child_initial-B@coarse_initial
    resolved=P@child_initial;homogeneous=P@(B@coarse_initial)
    retained=A@(P@initial_hidden)
    operator=A@homogeneous-coarse_solution
    change=A@resolved-coarse_solution
    norm=lambda x:normalized_error(mass,coarse_initial,x)
    norm0sq=float(mass@((coarse_initial-mass@coarse_initial/mass.sum())**2))
    return {'retained_information_norm':norm(retained),'homogenized_operator_change_norm':norm(operator),
            'total_parent_state_change_norm':norm(change),
            'normalized_cross_term':float(2*np.dot(mass,retained*operator)/norm0sq),
            'split_max_absolute_residual':float(np.max(abs(change-retained-operator)))}, {
                'child_solution':resolved,'homogenized_child_solution':homogeneous,
                'aggregated_solution':A@resolved,'aggregated_homogenized_solution':A@homogeneous,
                'retained_information_term':retained,'operator_change_term':operator,'total_parent_change':change}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    old=json.loads((source/'sha256.json').read_text())
    for path,h in old.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:raise RuntimeError('old evidence mismatch: '+path)
    cfg0=json.loads((source/'config.json').read_text());times=cfg0['times']
    flow=Contact(rho_left=cfg0['rho_left'],rho_right=cfg0['rho_right'],D=cfg0['D'],half_length=cfg0['half_length'],speed=cfg0['speed'])
    reference=StepReference(flow,left=cfg0['initial_left'],right=cfg0['initial_right'])
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'initial fixed material subcell representation control of the same contact scalar step',
            'source':str(source),'parent_cells':[32,64,128],'child_cells':[33,65,129],
            'times':times,'physical_configuration':cfg0,
            'change':'split exactly the straddling material cell at a=0, at t=0 only',
            'child_initial':'exact initial step average, no inferred concentration',
            'homogenized_control':'same child operator, initialize both child values from old parent mean',
            'comparison_norm':'aggregate to original cells, normalize by original parent initial fluctuation norm',
            'common_child_backend':'ordinary_volume_center; other definitions checked for equivalence only',
            'semigroup_probe_time':.05,'child_BE_check':'65 children at t=.1, 512 and 1024 steps',
            'no_runtime_remapping':True,'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];initial=[];closures=[]
    for cells in config['parent_cells']:
        with np.load(source/f'inputs-cut-{cells}.npz') as z:
            parent_faces=z['material_faces'];m=z['mass'];c0=z['c0'];ref_parent=z['reference']
        child_faces=split_faces_at_contact(parent_faces);ms=flow.mass(child_faces);child_c0=reference.average(child_faces,0.)
        A,B,membership=aggregation_and_lift(parent_faces,child_faces,m,ms)
        initial_residual=float(np.max(abs(A@child_c0-c0)))
        if initial_residual>1e-13:raise RuntimeError('initial parent averages changed')
        child_right=child_faces[:-1]>=0
        g=flow.conductance(child_faces,'ordinary_volume_center')
        gdiff={method:float(np.max(abs(flow.conductance(child_faces,method)-g))/np.max(g)) for method in METHODS}
        if max(gdiff.values())>1e-12:raise RuntimeError('unexpected child-backend difference')
        total_mass=float(ms.sum());true_E0=.5*float(ms@((child_c0-reference.mean)**2))
        homogenized_E0=.5*float(ms@((B@c0-reference.mean)**2))
        delta=child_c0-B@c0;Qchild=-laplacian(g)/ms[:,None]
        hidden_rate=A@(Qchild@delta)
        initial.append({'parent_cells':cells,'child_cells':len(ms),
                        'mass_sum':total_mass,'initial_scalar_mass':float(ms@child_c0),
                        'initial_parent_average_max_residual':initial_residual,
                        'true_child_initial_energy':true_E0,'homogenized_child_initial_energy':homogenized_E0,
                        'retained_subcell_energy':true_E0-homogenized_E0,
                        'true_initial_right_inventory':float(ms[child_right]@child_c0[child_right]),
                        'homogenized_initial_right_inventory':float(ms[child_right]@(B@c0)[child_right]),
                        'aggregation_lift_identity_residual':float(np.max(abs(A@B-np.eye(cells)))),
                        'mass_aggregation_identity_residual':float(np.max(abs(m@A-ms))),
                        'child_conductance_relative_differences':gdiff,
                        'hidden_state_initial_derivative_norm_per_second':normalized_error(m,c0,hidden_rate)})
        _,Phalf,_=semidiscrete(child_c0,ms,g,.05)
        _,Pfull,_=semidiscrete(child_c0,ms,g,.1)
        C=A@Pfull@B;Ch=A@Phalf@B;defect=C-Ch@Ch
        factor=A@Phalf@(np.eye(len(ms))-B@A)@Phalf@B
        closures.append({'parent_cells':cells,'time':.05,
                         'child_semigroup_max_residual':float(np.max(abs(Pfull-Phalf@Phalf))),
                         'projected_defect_factorization_max_residual':float(np.max(abs(defect-factor))),
                         'projected_map_semigroup_defect_max_abs':float(np.max(abs(defect))),
                         'projected_map_semigroup_defect_max_row_l1':float(np.max(np.sum(abs(defect),axis=1))),
                         'projected_map_row_sum_error':float(np.max(abs(C.sum(axis=1)-1))),
                         'projected_map_mass_invariant_relative_error':float(np.max(abs(m@C-m))/np.max(m)),
                         'projected_map_min_entry':float(C.min())})
        np.savez_compressed(output/f'partition-parent{cells}.npz',parent_faces=parent_faces,child_faces=child_faces,
                            parent_mass=m,child_mass=ms,parent_c0=c0,child_c0=child_c0,homogenized_child_c0=B@c0,
                            A=A,B=B,parent_index=membership,g_child=g,hidden_initial_state=delta,hidden_initial_rate=hidden_rate,
                            projected_map_full=C,projected_map_half=Ch,projected_semigroup_defect=defect,child_map_full=Pfull,child_map_half=Phalf)
        old_candidates={}
        for method in METHODS:
            with np.load(source/f'{method}-cut-{cells}.npz') as z:old_candidates[method]=z['c_semidiscrete']
        for it,t in enumerate(times):
            c,P,diag=semidiscrete(child_c0,ms,g,t)
            child_exact=reference.average(child_faces,t)
            ref_aggregation=float(np.max(abs(A@child_exact-ref_parent[it])))
            if ref_aggregation>1e-12:raise RuntimeError('reference aggregation mismatch')
            inventory=float(ms[child_right]@c[child_right]);exact_inventory,tail,images=reference.right_inventory(0.,t)
            row={'parent_cells':cells,'child_cells':len(ms),'time':t,
                 'aggregated_error':normalized_error(m,c0,A@c-ref_parent[it]),
                 'child_full_resolution_error':normalized_error(ms,child_c0,c-child_exact),
                 'right_phase_inventory':inventory,'right_phase_inventory_exact':exact_inventory,
                 'right_phase_inventory_relative_error':abs(inventory-exact_inventory)/exact_inventory,
                 'right_phase_inventory_signed_relative_error':(inventory-exact_inventory)/exact_inventory,
                 'layer_length_over_largest_contact_child_width':float(np.sqrt(flow.D*t)/max(np.diff(child_faces)[(child_faces[:-1]<=0)&(child_faces[1:]>=0)])),
                 'reference_parent_average_max_residual':ref_aggregation,'semidiscrete_diagnostics':diag,
                 'relative_mass_drift':abs(float(ms@(c-child_c0)))/abs(float(ms@child_c0)),
                 'comparisons_to_original':{}}
            arrays={'child_solution':c,'child_reference':child_exact,'aggregated_reference':ref_parent[it],
                    'aggregated_solution':A@c,'child_mass':ms,'parent_mass':m}
            for method in METHODS:
                d,a=compare(A,B,P,child_c0,c0,old_candidates[method][it],m)
                d['old_parent_error']=normalized_error(m,c0,old_candidates[method][it]-ref_parent[it])
                row['comparisons_to_original'][method]=d
                for name,value in a.items():arrays[method+'_'+name]=value
            if cells==64 and t==.1:
                be={}
                for steps in (512,1024):
                    cn,d=backward_euler(child_c0,ms,g,t,steps)
                    be[str(steps)]={'normalized_time_error':normalized_error(ms,child_c0,cn-c),**d}
                    arrays['child_BE'+str(steps)]=cn
                row['BE_check']=be
            rows.append(row)
            np.savez_compressed(output/f'result-parent{cells}-t{t:g}.npz',**arrays)
            print('parent',cells,'time',t,'aggregated error',row['aggregated_error'],'right inventory relative error',row['right_phase_inventory_relative_error'],flush=True)
    result={'initial_partition':initial,'runs':rows,'projected_semigroup_checks':closures,
            'third_stage_physical_acceptance':False,
            'interpretation':'operator identity is exact relative to the fixed child discretization; semigroup defect is not a continuum error bound'}
    (output/'results.json').write_text(json.dumps(result,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
