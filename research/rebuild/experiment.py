"""One reconstructed physical case; no historical inputs/results claimed."""
from pathlib import Path
import sys,json,hashlib,platform,time
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from becker import Becker,initial
from reference import solve,project
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'material_fv'))
from material_fv import step


def integrate(flow,left,right,t,D,order=32,weight=None):
    nodes,weights=leggauss(order)
    x=(left[:,None]+right[:,None])/2+(right-left)[:,None]*nodes/2
    y=1/(flow.rho_mu(x,t)**2*D)
    if weight is not None:y*=weight(x)
    return (right-left)/2*(y@weights)


def conductance(flow,edges,t,D,method,order=32):
    if D==0:return np.zeros(len(edges)-2)
    centers=(edges[:-1]+edges[1:])/2
    if method=='ordinary_mass_center':
        resistance=integrate(flow,centers[:-1],centers[1:],t,D,order)
    elif method=='ordinary_volume_center':
        faces=flow.x_mu(edges,t)
        x=(faces[:-1]+faces[1:])/2
        center_mu=flow.B*(flow.z_at_xi(1.)+flow.j*t/flow.B-flow.z_at_xi(flow.u1*t-x))
        resistance=integrate(flow,center_mu[:-1],center_mu[1:],t,D,order)
    elif method=='mass_average_resistance':
        m=np.diff(edges)
        part_left=integrate(flow,edges[:-1],edges[1:],t,D,order,lambda x:(x-edges[:-1,None])/m[:,None])
        part_right=integrate(flow,edges[:-1],edges[1:],t,D,order,lambda x:(edges[1:,None]-x)/m[:,None])
        resistance=part_left[:-1]+part_right[1:]
    else:raise ValueError(method)
    return 1/resistance


def candidate(flow,edges,steps,method,T=.1,D=.02):
    m=np.diff(edges);c0=initial(edges);c=c0.copy();dt=T/steps;mean=m@c0/m.sum()
    drift=residual=identity=0.;lo=c.min();hi=c.max();maxquad=0.
    for k in range(steps):
        g=conductance(flow,edges,k*dt,D,method)
        if k in (0,steps//2,steps-1):
            g64=conductance(flow,edges,k*dt,D,method,64)
            maxquad=max(maxquad,float(np.max(abs(g-g64)/g64)))
        nxt=step(c,m,g,dt)
        lhs=m*(nxt-c);jump=dt*g*np.diff(nxt);lhs[:-1]-=jump;lhs[1:]+=jump
        residual=max(residual,float(np.linalg.norm(lhs)/np.linalg.norm(m*c)))
        energy=lambda a:.5*np.dot(m,(a-mean)**2)
        bal=energy(nxt)-energy(c)+dt*np.dot(g,np.diff(nxt)**2)+.5*np.dot(m,(nxt-c)**2)
        identity=max(identity,abs(float(bal))/float(np.dot(m,c*c)))
        drift=max(drift,abs(float(m@(nxt-c0)))/abs(float(m@c0)))
        lo=min(lo,float(nxt.min()));hi=max(hi,float(nxt.max()));c=nxt
    if max(drift,residual,identity)>1e-10 or lo<c0.min()-1e-10 or hi>c0.max()+1e-10:
        raise RuntimeError('candidate algebra check failed')
    return c,{'max_relative_mass_drift':drift,'max_relative_solve_residual':residual,
              'max_relative_energy_balance_residual':identity,'min_all_times':float(lo),'max_all_times':float(hi),
              'sampled_32_vs_64_gauss_conductance_relative_difference':maxquad}


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    f=Becker();labels=np.linspace(-1,1,33);edges=f.mu(labels);m=np.diff(edges);c0=initial(edges)
    norm=np.sqrt(m@((c0-m@c0/m.sum())**2))
    error=lambda a:float(np.sqrt(m@(a*a))/norm)
    config={'study':'independent reconstruction, not restored v080','delta':f.delta,'delta_definition':'steady-velocity 10--90 width',
            'Mach':f.mach,'gamma':f.gamma,'rho1':f.rho1,'p1':f.p1,'R':f.R,'Pr':.75,'dynamic_viscosity':f.eta,
            'D':.02,'T':.1,'n':16,'material_columns':32,'initial_labels':[-1.,1.],
            'initial_field':'.5+.25*cos(pi*mu/total_mass), exact material-cell averages',
            'boundary':'zero scalar flux at moving material endpoints; unbounded prescribed fluid',
            'baseline_center':'geometric midpoint of true moving material faces (ordinary_volume_center)',
            'center_convention_control':'ordinary_mass_center uses midpoint of each mass interval mapped to physical space',
            'reference':'uniform mass FV, midface rho^2 D, midpoint CN; independently assembled',
            'reference_projection':'conservative piecewise-constant integration of reference cell means',
            'normalization':'sqrt(sum m*(c-c_ref)^2)/sqrt(sum m*(c0-mean_m(c0))^2)',
            'fluid_orc_enabled':False,'historical_kernel_comparison_available':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(output/'inputs.npz',label_edges=labels,mass_edges=edges,mass=m,c0=c0,faces_t0=f.x_mu(edges,0),faces_tfinal=f.x_mu(edges,.1))
    references={};rows=[]
    # Only the refinements needed to assess this one case's reference uncertainty.
    for cells,steps in ((1024,2048),(2048,2048),(4096,2048),(4096,1024),(4096,4096)):
        start=time.monotonic();e,i,c,d=solve(f,cells,steps)
        if d['max_relative_mass_drift']>1e-9 or d['min_all_times']<d['initial_min']-1e-10 or d['max_all_times']>d['initial_max']+1e-10:
            raise RuntimeError('reference conservation or bounds check failed')
        p=project(e,c,edges);references[cells,steps]=p
        np.savez_compressed(output/f'reference-n{cells}-k{steps}.npz',mass_edges=e,c0=i,c=c,projected=p)
        rows.append({'cells':cells,'steps':steps,'wall_seconds':time.monotonic()-start,**d})
        print('reference',cells,steps,'done',flush=True)
    best=references[4096,4096]
    checks={'space_1024_2048':error(references[1024,2048]-references[2048,2048]),
            'space_2048_4096':error(references[2048,2048]-references[4096,2048]),
            'time_1024_2048':error(references[4096,1024]-references[4096,2048]),
            'time_2048_4096':error(references[4096,2048]-best),
            'warning':'refinement differences, not rigorous continuum error bounds'}
    results=[];arrays={}
    for method in ('ordinary_volume_center','ordinary_mass_center','mass_average_resistance'):
        for steps in (128,256,512):
            c,d=candidate(f,edges,steps,method);arrays[method,steps]=c
            np.savez_compressed(output/f'{method}-k{steps}.npz',mass=m,c0=c0,c=c,reference=best,mass_edges=edges)
            results.append({'method':method,'steps':steps,'normalized_error':error(c-best),**d})
            print('candidate',method,steps,results[-1]['normalized_error'],flush=True)
    differences={method:{'128_256':error(arrays[method,128]-arrays[method,256]),
                          '256_512':error(arrays[method,256]-arrays[method,512])}
                 for method in ('ordinary_volume_center','ordinary_mass_center','mass_average_resistance')}
    finest={r['method']:r['normalized_error'] for r in results if r['steps']==512}
    ref_indicator=checks['space_2048_4096']+checks['time_2048_4096']
    comparison={}
    for method in ('ordinary_volume_center','ordinary_mass_center'):
        gap=abs(finest[method]-finest['mass_average_resistance'])
        indicator=differences[method]['256_512']+differences['mass_average_resistance']['256_512']+2*ref_indicator
        comparison[method]={'absolute_error_gap':gap,'combined_empirical_uncertainty_indicator':indicator,
                            'gap_over_indicator':gap/indicator,'lower_error_method':method if finest[method]<finest['mass_average_resistance'] else 'mass_average_resistance',
                            'warning':'empirical refinement indicators, not rigorous error bars'}
    report={'pairwise_interpretation':comparison,'reference_runs':rows,'reference_refinement':checks,'candidate_runs':results,'candidate_time_differences':differences,
            'third_stage_physical_acceptance':False,'historical_numbers_reproduced':False}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    files=list(output.glob('*'))+list(Path(__file__).parent.glob('*.py'))+[Path(__file__).resolve().parents[1]/'material_fv/material_fv.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    (output/'sha256.json').write_text(json.dumps(hashes,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();main(args.output)
