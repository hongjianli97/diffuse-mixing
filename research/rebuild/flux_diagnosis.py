"""Postprocess saved results: common-state flux closure and terminal-state split.

No time integration, weight fitting, threshold changes or physics acceptance.
"""
from pathlib import Path
import hashlib,json,argparse,platform
import numpy as np
import scipy
from becker import Becker,initial
from experiment import conductance
from reference import project

METHODS=('ordinary_volume_center','ordinary_mass_center','mass_average_resistance')


def norm(a,w):return float(np.sqrt(np.dot(w,np.asarray(a)**2)))


def tendency(flux,mass):
    flux=np.asarray(flux,dtype=float);mass=np.asarray(mass,dtype=float)
    if flux.shape != (len(mass)-1,):raise ValueError('internal flux/cell dimensions differ')
    # Exact no-flux endpoints; positive J is rightward.
    return -np.diff(np.r_[0.,flux,0.])/mass


def sampled_reference_flux(flow,edges,c,t,D,target_faces):
    """Fine-grid centered flux, then linearly interpolated onto coarse faces.

    This is a numerical flux estimate, not an exact continuum flux. Its
    interpolation/gradient errors are assessed using saved refinements.
    """
    widths=np.diff(edges)
    if not np.allclose(widths,widths[0],rtol=1e-10,atol=1e-14):
        raise ValueError('reference flux extraction requires uniform mass mesh')
    if len(c)!=len(widths):raise ValueError('reference dimensions')
    fine=-flow.rho_mu(edges[1:-1],t)**2*D*np.diff(c)/widths[0]
    return np.interp(target_faces,edges,np.r_[0.,fine,0.])


def split(flow,edges,c_ref,j_ref,t,D,c_candidates=None):
    m=np.diff(edges);face_weights=(m[:-1]+m[1:])/2
    jnorm=norm(j_ref,face_weights)
    if jnorm<=0:raise ValueError('zero reference flux norm: relative diagnostic undefined')
    c0=initial(edges);initial_norm=norm(c0-m@c0/m.sum(),m)
    g_avg=conductance(flow,edges,t,D,'mass_average_resistance')
    j_avg=-g_avg*np.diff(c_ref);average_bias=j_avg-j_ref
    arrays={'mass_edges':edges,'mass':m,'reference_c':c_ref,'reference_flux':j_ref,
            'reference_tendency':tendency(j_ref,m),'face_weights':face_weights}
    rows=[]
    for method in METHODS:
        g=conductance(flow,edges,t,D,method)
        j_closure=-g*np.diff(c_ref)
        scale=(g/g_avg-1)*j_avg
        closure=j_closure-j_ref
        arrays[method+'_g']=g
        arrays[method+'_common_state_flux']=j_closure
        arrays[method+'_average_bias']=average_bias
        arrays[method+'_resistance_scale_term']=scale
        arrays[method+'_closure_error']=closure
        row={'method':method,
             'relative_common_state_flux_error':norm(closure,face_weights)/jnorm,
             'relative_average_bias_norm':norm(average_bias,face_weights)/jnorm,
             'relative_resistance_scale_norm':norm(scale,face_weights)/jnorm,
             'normalized_average_scale_cross_term':float(2*np.dot(face_weights,average_bias*scale)/jnorm**2),
             'closure_split_max_residual':float(np.max(abs(closure-average_bias-scale))),
             'common_state_tendency_defect_per_second':norm(tendency(closure,m),m)/initial_norm,
             'max_relative_g_to_average_change':float(np.max(abs(g/g_avg-1)))}
        if c_candidates is not None:
            state=-g*np.diff(c_candidates[method]-c_ref)
            total=-g*np.diff(c_candidates[method])-j_ref
            arrays[method+'_state_error_flux']=state
            arrays[method+'_total_terminal_flux_error']=total
            row.update(relative_terminal_state_flux_term=norm(state,face_weights)/jnorm,
                       relative_total_terminal_flux_error=norm(total,face_weights)/jnorm,
                       normalized_closure_state_cross_term=float(2*np.dot(face_weights,closure*state)/jnorm**2),
                       state_split_max_residual=float(np.max(abs(total-closure-state))))
        rows.append(row)
    return rows,arrays


def main(run,output):
    run=Path(run).resolve();output=Path(output).resolve()
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((run/'sha256.json').read_text())
    for rel,digest in manifest.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('previous-run hash mismatch: '+rel)
    cfg=json.loads((run/'config.json').read_text())
    f=Becker(delta=cfg['delta'],mach=cfg['Mach'],gamma=cfg['gamma'],rho1=cfg['rho1'],p1=cfg['p1'],R=cfg['R'])
    D,T=cfg['D'],cfg['T']
    with np.load(run/'inputs.npz',allow_pickle=False) as z:
        edges=z['mass_edges'];m=z['mass'];c0=z['c0']
    np.testing.assert_allclose(np.diff(edges),m,rtol=1e-13)
    np.testing.assert_allclose(initial(edges),c0,rtol=1e-13)
    candidates={}
    for method in METHODS:
        with np.load(run/f'{method}-k512.npz',allow_pickle=False) as z:
            if not np.array_equal(z['mass_edges'],edges) or not np.array_equal(z['c0'],c0):
                raise RuntimeError('candidate identity mismatch')
            candidates[method]=z['c']
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'common-state closure diagnosis of saved reconstructed Becker case',
            'source_run':str(run),'D':D,'T':T,'saved_references_only':True,
            'methods':METHODS,'face_norm_weights':'(mass_i+mass_i+1)/2 on internal faces',
            'flux_orientation':'positive rightward, J=-rho^2 D c_mu; endpoints zero',
            'candidate_flux_time':'all diagnostic conductances evaluated at T, not last BE left endpoint',
            'common_state':'same projected reference cell means for every method at a given level',
            'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    j0=f.rho_mu(edges[1:-1],0)**2*D*.25*np.pi/f.total*np.sin(np.pi*edges[1:-1]/f.total)
    rows0,arr0=split(f,edges,c0,j0,0.,D)
    np.savez_compressed(output/'initial-exact-diagnosis.npz',**arr0)
    snapshots={};ref_rows={}
    for cells,steps in ((1024,2048),(2048,2048),(4096,2048),(4096,1024),(4096,4096)):
        with np.load(run/f'reference-n{cells}-k{steps}.npz',allow_pickle=False) as z:
            e,c=z['mass_edges'],z['c']
        cbar=project(e,c,edges)
        j=sampled_reference_flux(f,e,c,T,D,edges[1:-1])
        rows,arrays=split(f,edges,cbar,j,T,D,candidates)
        key=f'n{cells}-k{steps}';snapshots[key]=arrays;ref_rows[key]=rows
        np.savez_compressed(output/f'final-{key}.npz',**arrays)
    best=snapshots['n4096-k4096'];w=best['face_weights'];jnorm=norm(best['reference_flux'],w)
    checks={}
    pairs={'space_1024_2048':('n1024-k2048','n2048-k2048'),
           'space_2048_4096':('n2048-k2048','n4096-k2048'),
           'time_1024_2048':('n4096-k1024','n4096-k2048'),
           'time_2048_4096':('n4096-k2048','n4096-k4096')}
    for name,(a,b) in pairs.items():
        checks[name]={'relative_sampled_reference_flux_difference':norm(snapshots[a]['reference_flux']-snapshots[b]['reference_flux'],w)/jnorm,
                     'common_state_closure_difference':{method:norm(snapshots[a][method+'_closure_error']-snapshots[b][method+'_closure_error'],w)/jnorm for method in METHODS}}
    report={'initial_exact':rows0,'final_finest':ref_rows['n4096-k4096'],
            'all_final_reference_levels':ref_rows,'refinement_indicators':checks,
            'interpretation_limit':'instantaneous t=0 and t=T diagnostic, not a proof of the time-integrated concentration-error ranking',
            'uncertainty_warning':'refinement differences, not rigorous continuum flux-error bounds'}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    files=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'initial_exact':rows0,'final_finest':ref_rows['n4096-k4096'],'refinement_indicators':checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);args=p.parse_args();main(args.run,args.output)
