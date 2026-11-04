"""Fixed finest local mesh: modal BE forecast and verified actual stepping."""
from pathlib import Path
import argparse,json,hashlib,platform,time
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from fixed_be_time import integrate,spectral_backward_euler
from spatial_study import normalized_error


def time_indicators(delta,m,c0,A,mp,cp,right,inventory_ref,spatial):
    full=normalized_error(m,c0,delta);parent=normalized_error(mp,cp,A@delta)
    inventory=abs(float(m[right]@delta[right]))/inventory_ref
    ratios={'full_resolution':full/spatial['full_resolution_error'],
            'parent_observation':parent/spatial['aggregated_error'],
            'right_inventory':inventory/spatial['right_inventory_absolute_relative_error']}
    return {'time_error_full':full,'time_error_parent':parent,'time_error_inventory_relative':inventory,
            'time_to_spatial_ratios':ratios,'maximum_ratio':max(ratios.values())}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    old=json.loads((source/'sha256.json').read_text())
    for path,h in old.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:raise RuntimeError('source evidence changed: '+path)
    spatial=json.loads((source/'results.json').read_text());times=[x['time'] for x in spatial['runs'] if x['resolution']==8]
    with np.load(source/'mesh-q8.npz') as z:
        faces=z['faces'];m=z['child_mass'];c0=z['child_c0'];g=z['g'];A=z['A'];mp=z['parent_mass'];cp=z['parent_c0']
    right=faces[:-1]>=0;output.mkdir(parents=True,exist_ok=False)
    config={'study':'fixed 436-cell local contact step; uniform-time BE precision separation',
            'source':str(source),'cells':len(m),'times':times,'candidate_steps':[512,1024,2048,4096,8192,16384,32768,65536,131072],
            'time_to_spatial_diagnostic_limit':.1,
            'diagnostic_metrics':['full-resolution concentration','original 64-cell observation','whole right-phase inventory'],
            'selection':'first listed step count with all three modal time/spatial ratios <= .1; actual direct run must confirm',
            'warning':'numerical error-separation diagnostic, not a changed physical acceptance threshold',
            'actual_solver':'cached SPD tridiagonal factorization of the original backward Euler equation',
            'forecast':'exact fixed-operator BE modal multipliers, no Richardson state correction',
            'same_trajectory_warning':'each observation horizon has its own uniform grid; no single nonuniform trajectory claimed',
            'all_invariant_tolerances':1e-10,'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    diag=np.zeros(len(m));diag[:-1]+=g;diag[1:]+=g
    eigenvalues,U=eigh_tridiagonal(diag/m,-g/np.sqrt(m[:-1]*m[1:]))
    forecasts=[];runs=[];previous_checks=[]
    for t in times:
        row0=next(x for x in spatial['runs'] if x['resolution']==8 and x['time']==t)
        with np.load(source/f'result-q8-t{t:g}.npz') as z:
            semi=z['child_solution'];reference=z['child_reference'];parent_reference=z['parent_reference']
            prior_be={steps:z[f'c_BE{steps}'] for steps in (512,1024) if f'c_BE{steps}' in z.files}
        selected=None
        for steps in config['candidate_steps']:
            predicted,delta=spectral_backward_euler(eigenvalues,U,m,c0,t,steps)
            indicator=time_indicators(delta,m,c0,A,mp,cp,right,row0['right_inventory_reference'],row0)
            forecasts.append({'time':t,'steps':steps,**indicator})
            if steps in prior_be:
                previous_checks.append({'time':t,'steps':steps,'saved_BE_vs_modal_max_abs_difference':float(np.max(abs(predicted-prior_be[steps])))})
            if selected is None and indicator['maximum_ratio']<=config['time_to_spatial_diagnostic_limit']:
                selected=steps
        if selected is None:raise RuntimeError('declared step candidates did not isolate time error')
        start=time.monotonic();actual,diagnostics=integrate(c0,m,g,t,selected)
        forecast,delta=spectral_backward_euler(eigenvalues,U,m,c0,t,selected)
        direct=time_indicators(actual-semi,m,c0,A,mp,cp,right,row0['right_inventory_reference'],row0)
        if direct['maximum_ratio']>config['time_to_spatial_diagnostic_limit']:
            raise RuntimeError('actual stepping failed the unchanged error-separation diagnostic')
        algorithm_diff=float(np.max(abs(actual-forecast)))
        if algorithm_diff>1e-10:raise RuntimeError('actual BE differs from modal BE')
        spatial_vec=semi-reference;time_vec=actual-semi;total=actual-reference
        norm0=float(m@((c0-m@c0/m.sum())**2));inventory=float(m[right]@actual[right]);inventory_ref=row0['right_inventory_reference']
        row={'time':t,'selected_steps':selected,'wall_seconds':time.monotonic()-start,
             'forecast_vs_actual_max_abs_difference':algorithm_diff,**direct,
             'full_total_error':normalized_error(m,c0,total),'full_spatial_error':row0['full_resolution_error'],
             'parent_total_error':normalized_error(mp,cp,A@actual-parent_reference),'parent_spatial_error':row0['aggregated_error'],
             'right_inventory':inventory,'right_inventory_reference':inventory_ref,
             'right_inventory_signed_total_relative_error':float((inventory-inventory_ref)/inventory_ref),
             'right_inventory_signed_spatial_relative_error':row0['right_inventory_signed_relative_error'],
             'normalized_spatial_time_cross_term':float(2*np.dot(m,spatial_vec*time_vec)/norm0),
             'error_vector_identity_max_residual':float(np.max(abs(total-spatial_vec-time_vec))),
             'BE_diagnostics':diagnostics}
        runs.append(row)
        np.savez_compressed(output/f'result-t{t:g}.npz',time=np.array(t),steps=np.array(selected),actual_BE=actual,modal_BE=forecast,semidiscrete=semi,
                            reference=reference,mass=m,c0=c0,time_error_vector=time_vec,spatial_error_vector=spatial_vec,total_error_vector=total,
                            parent_actual=A@actual,parent_reference=parent_reference,parent_mass=mp,parent_c0=cp)
        print('time',t,'steps',selected,'max time/spatial',direct['maximum_ratio'],'inventory relative error',row['right_inventory_signed_total_relative_error'],flush=True)
    result={'runs':runs,'forecasts':forecasts,'reuse_saved_BE_checks':previous_checks,
            'all_actual_error_separation_diagnostics_met':True,'third_stage_physical_acceptance':False}
    (output/'results.json').write_text(json.dumps(result,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
