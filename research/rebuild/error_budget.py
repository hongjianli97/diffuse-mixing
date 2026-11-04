"""Exact discrete error budget relative to recorded numerical reference paths.

New history recording is needed: prior exports contain only final snapshots.
No candidate/physics definitions or original outputs are modified.
"""
from pathlib import Path
import argparse,hashlib,json,platform
import numpy as np
import scipy
from scipy.linalg import solve_banded
from becker import Becker,initial
from reference import project
from experiment import conductance,step
from flux_diagnosis import METHODS,tendency,sampled_reference_flux

PARTS=('initial_projection','average_closure','center_scale','time_remainder')


def apply_q(c,m,g):
    return tendency(-g*np.diff(c),m)


def resolvent(rhs,m,g,dt):
    """Apply (M+dt L)^-1 M to a vector or independent RHS columns."""
    rhs=np.asarray(rhs,dtype=float)
    if rhs.ndim not in (1,2) or rhs.shape[0]!=len(m):raise ValueError('RHS dimensions')
    band=np.zeros((3,len(m)));band[1]=m
    band[1,:-1]+=dt*g;band[1,1:]+=dt*g
    band[0,1:]=-dt*g;band[2,:-1]=-dt*g
    return solve_banded((1,1),band,m*rhs if rhs.ndim==1 else m[:,None]*rhs)


def record_reference(flow,cells,steps,coarse_edges,samples=512,T=.1,D=.02):
    """Instrument the existing CN reference scheme; retain only coarse data.

    Also record f=P Q_fine(t)c_fine via interpolated fine face fluxes.
    This is the projected semidiscrete reference derivative, not a derivative
    obtained by finite differencing the coarse history.
    """
    if steps%samples:raise ValueError('sampling must align with reference times')
    edges=np.linspace(0,flow.total,cells+1);dm=flow.total/cells;dt=T/steps
    c0=initial(edges);c=c0.copy();band=np.zeros((3,cells));every=steps//samples
    coarse=[];derivatives=[];fluxes=[];drift=lo=hi=0.;mass0=dm*c0.sum()
    lo=float(c0.min());hi=float(c0.max());projection_identity=0.
    for k in range(steps+1):
        t=k*dt
        if k%every==0:
            j=sampled_reference_flux(flow,edges,c,t,D,coarse_edges[1:-1])
            f=tendency(j,np.diff(coarse_edges))
            fine_g=flow.rho_mu(edges[1:-1],t)**2*D/dm
            projected_rhs=project(edges,apply_q(c,np.full(cells,dm),fine_g),coarse_edges)
            projection_identity=max(projection_identity,float(np.max(abs(f-projected_rhs))))
            coarse.append(project(edges,c,coarse_edges));derivatives.append(f);fluxes.append(j)
        if k==steps:break
        g=flow.rho_mu(edges[1:-1],(k+.5)*dt)**2*D/dm
        half=.5*dt*g
        diagonal=np.full(cells,dm);diagonal[:-1]+=half;diagonal[1:]+=half
        rhs=dm*c;jump=half*np.diff(c);rhs[:-1]+=jump;rhs[1:]-=jump
        band[1]=diagonal;band[0,1:]=-half;band[2,:-1]=-half
        c=solve_banded((1,1),band,rhs,check_finite=False)
        if not np.isfinite(c).all():raise RuntimeError('nonfinite reference')
        drift=max(drift,abs(float(dm*c.sum()-mass0))/abs(float(mass0)))
        lo=min(lo,float(c.min()));hi=max(hi,float(c.max()))
    if drift>1e-9 or lo<c0.min()-1e-10 or hi>c0.max()+1e-10:
        raise RuntimeError('recorded reference failed conservation/bounds checks')
    return {'time':np.linspace(0,T,samples+1),'c':np.array(coarse),
            'derivative':np.array(derivatives),'flux':np.array(fluxes),
            'fine_mass_edges':edges,'fine_c0':c0,'fine_final':c}, {
                'cells':cells,'steps':steps,'max_relative_mass_drift':drift,
                'min_all_times':lo,'max_all_times':hi,
                'projection_divergence_identity_max_residual':projection_identity}


def budget(m,c0,times,reference,reference_derivative,g_method,g_average,numerical=None):
    """Propagate four signed components through each method's own resolvent.

    s_A=Q_A r-f; s_C=(Q-Q_A)r;
    tau=Q(r_next-r)+f-(r_next-r)/dt.
    Their sum is Q r_next-(r_next-r)/dt. Initial mismatch is not discarded.
    """
    times=np.asarray(times);r=np.asarray(reference);f=np.asarray(reference_derivative)
    steps=len(times)-1;n=len(m)
    if r.shape!=(steps+1,n) or f.shape!=r.shape:raise ValueError('history dimensions')
    if len(g_method)!=steps or len(g_average)!=steps:raise ValueError('conductance history dimensions')
    if np.any(np.diff(times)<=0) or np.any(m<=0):raise ValueError('positive times/masses required')
    parts=np.zeros((n,4));parts[:,0]=c0-r[0]
    c=c0.copy();part_hist=[parts.copy()];c_hist=[c.copy()];forcing=[];residual=source_residual=0.
    norm0=np.sqrt(m@((c0-m@c0/m.sum())**2))
    if norm0<=0:raise ValueError('nonconstant initial state required for normalized budget')
    for k,dt in enumerate(np.diff(times)):
        g,ga=g_method[k],g_average[k];dr=r[k+1]-r[k]
        average=apply_q(r[k],m,ga)-f[k]
        scale=apply_q(r[k],m,g)-apply_q(r[k],m,ga)
        time_part=apply_q(dr,m,g)+f[k]-dr/dt
        rates=np.column_stack((np.zeros(n),average,scale,time_part))
        direct=apply_q(r[k+1],m,g)-dr/dt
        source_residual=max(source_residual,float(np.max(abs(rates.sum(axis=1)-direct))))
        parts=resolvent(parts+dt*rates,m,g,dt)
        # Use the original candidate solver for the independent direct path.
        c=step(c,m,g,dt)
        residual=max(residual,float(np.sqrt(m@((parts.sum(axis=1)-(c-r[k+1]))**2))/norm0))
        part_hist.append(parts.copy());c_hist.append(c.copy());forcing.append(rates)
    parts_history=np.array(part_hist);c_history=np.array(c_hist)
    if numerical is not None:np.testing.assert_allclose(c_history[-1],numerical,rtol=0,atol=2e-13)
    if residual>1e-10:raise RuntimeError('error reconstruction failure')
    rms=lambda a:float(np.sqrt(m@(a*a))/norm0)
    gram=(parts.T*m)@parts/norm0**2
    error=c-r[-1];space=parts[:,1]+parts[:,2]
    sums=parts.sum(axis=1)
    summary={'terminal_normalized_error':rms(error),
             'contribution_norms':{name:rms(parts[:,i]) for i,name in enumerate(PARTS)},
             'propagated_spatial_sum_norm':rms(space),
             'normalized_gram_matrix':gram.tolist(),
             'average_center_cross_term':float(2*gram[1,2]),
             'space_time_cross_term':float(2*m@(space*parts[:,3])/norm0**2),
             'max_history_reconstruction_residual':residual,
             'max_source_identity_absolute_residual':source_residual,
             'squared_error_minus_all_gram_entries':float(rms(error)**2-gram.sum()),
             'terminal_reconstructed_error_norm':rms(sums)}
    return summary,{'time':times,'numerical_c':c_history,'reference_c':r,'reference_derivative':f,
                    'contributions':parts_history,'forcing_rates':np.array(forcing),'mass':m,'c0':c0,
                    'g_method':np.array(g_method),'g_average':np.array(g_average)}


def main(run,output):
    root=Path(__file__).resolve().parents[2];run=Path(run).resolve();output=Path(output).resolve()
    old_manifest=json.loads((run/'sha256.json').read_text())
    for rel,digest in old_manifest.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('prior-run hash mismatch: '+rel)
    cfg=json.loads((run/'config.json').read_text());D,T=cfg['D'],cfg['T']
    flow=Becker(delta=cfg['delta'],mach=cfg['Mach'],gamma=cfg['gamma'],rho1=cfg['rho1'],p1=cfg['p1'],R=cfg['R'])
    with np.load(run/'inputs.npz',allow_pickle=False) as z:edges=z['mass_edges'];m=z['mass'];c0=z['c0']
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'Stage 3 discrete full-history error budget, one reconstructed physical case',
            'source_run':str(run),'D':D,'T':T,'n':cfg['n'],'material_columns':len(m),
            'candidate_steps':[256,512],'reference_levels':[[2048,2048],[4096,2048],[4096,4096]],
            'contribution_order':PARTS,'methods':METHODS,
            'time_remainder_definition':'Q_n(r_next-r_n)+f_n-(r_next-r_n)/dt',
            'reference_derivative':'projected fine semidiscrete RHS, not temporal differencing',
            'warning':'exact algebra relative to numerical reference, not exact continuum attribution',
            'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    times=np.linspace(0,T,513)
    conductances={method:np.array([conductance(flow,edges,t,D,method) for t in times[:-1]]) for method in METHODS}
    histories={};reference_rows=[];rows=[];arrays={}
    for cells,steps in config['reference_levels']:
        key=f'n{cells}-k{steps}';history,diag=record_reference(flow,cells,steps,edges,T=T,D=D)
        with np.load(run/f'reference-{key}.npz',allow_pickle=False) as z:
            endpoint_difference=float(np.max(abs(history['fine_final']-z['c'])))
            np.testing.assert_allclose(history['fine_final'],z['c'],rtol=0,atol=2e-13)
        diag['saved_endpoint_max_abs_difference']=endpoint_difference;reference_rows.append(diag)
        histories[key]=history
        np.savez_compressed(output/f'reference-history-{key}.npz',**history)
        print('reference history',key,'endpoint difference',endpoint_difference,flush=True)
        for candidate_steps in config['candidate_steps']:
            stride=512//candidate_steps;indices=np.arange(0,513,stride)
            for method in METHODS:
                g=conductances[method][indices[:-1]];ga=conductances['mass_average_resistance'][indices[:-1]]
                with np.load(run/f'{method}-k{candidate_steps}.npz',allow_pickle=False) as z:saved=z['c']
                summary,a=budget(m,c0,times[indices],history['c'][indices],history['derivative'][indices],g,ga,saved)
                row={'reference':key,'method':method,'steps':candidate_steps,**summary};rows.append(row)
                arrays[key,method,candidate_steps]=a
                np.savez_compressed(output/f'budget-{key}-{method}-k{candidate_steps}.npz',**a)
                print(key,method,candidate_steps,'error',summary['terminal_normalized_error'],'parts',summary['contribution_norms'],flush=True)
    norm0=np.sqrt(m@((c0-m@c0/m.sum())**2));rms=lambda a:float(np.sqrt(m@(a*a))/norm0)
    checks={}
    for label,(coarse,fine) in {'reference_space':('n2048-k2048','n4096-k2048'),
                              'reference_time':('n4096-k2048','n4096-k4096')}.items():
        checks[label]={}
        for method in METHODS:
            a=arrays[coarse,method,512];b=arrays[fine,method,512]
            checks[label][method]={name:rms(a['contributions'][-1,:,i]-b['contributions'][-1,:,i]) for i,name in enumerate(PARTS)}
    time_checks={}
    for method in METHODS:
        a=arrays['n4096-k4096',method,256];b=arrays['n4096-k4096',method,512]
        time_checks[method]={'time_component_norm_ratio_256_to_512':rms(a['contributions'][-1,:,3])/rms(b['contributions'][-1,:,3]),
                             'component_changes':{name:rms(a['contributions'][-1,:,i]-b['contributions'][-1,:,i]) for i,name in enumerate(PARTS)}}
    report={'reference_recording':reference_rows,'budgets':rows,'reference_refinement':checks,
            'candidate_time_refinement':time_checks,'third_stage_physical_acceptance':False}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);args=p.parse_args();main(args.run,args.output)
