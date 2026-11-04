"""One fixed physical Becker problem: three material spatial resolutions.

No fitted conductances, Richardson correction or physical parameter scan.
The 10% diagnostic is declared in config before candidate runs; it is not
Stage 3 physical acceptance and does not replace existing algebra tolerances.
"""
from pathlib import Path
import argparse,json,hashlib,platform,time
import numpy as np
import scipy
from becker import Becker,initial
from reference import solve,project
from experiment import candidate
from flux_diagnosis import METHODS,split,sampled_reference_flux


def normalized_error(m,c0,difference):
    m=np.asarray(m);c0=np.asarray(c0);difference=np.asarray(difference)
    variance=float(m@((c0-m@c0/m.sum())**2))
    if variance<=0:raise ValueError('nonzero initial fluctuation required')
    return float(np.sqrt(m@(difference*difference)/variance))


def empirical_order(e_coarse,e_fine):
    if e_coarse<=0 or e_fine<=0:raise ValueError('positive errors required')
    return float(np.log2(e_coarse/e_fine))


def reference_indicators(refs,m,c0):
    norm=lambda d:normalized_error(m,c0,d)
    return {'space_4096_8192':norm(refs['n4096-k4096']-refs['n8192-k4096']),
            'time_4096_8192':norm(refs['n8192-k4096']-refs['n8192-k8192'])}


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    manifest=json.loads((source/'sha256.json').read_text())
    for path,digest in manifest.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('prior-run mismatch: '+path)
    old=json.loads((source/'config.json').read_text())
    flow=Becker(delta=old['delta'],mach=old['Mach'],gamma=old['gamma'],rho1=old['rho1'],p1=old['p1'],R=old['R'])
    T,D=old['T'],old['D']
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'three material spatial resolutions of one reconstructed Becker case',
            'physical_source':str(source),'delta':flow.delta,'D':D,'T':T,'Mach':flow.mach,'gamma':flow.gamma,
            'Pr':.75,'dynamic_viscosity':flow.eta,'material_columns':[32,64,128],
            'initial_labels':[-1.,1.],'candidate_steps':[4096,8192],'methods':METHODS,
            'reference_levels':[[4096,4096],[8192,4096],[8192,8192]],
            'reused_reference':'n4096-k4096; validated previous SHA256',
            'normalization':'mass RMS concentration error / each grid initial mass fluctuation norm',
            'error_separation_diagnostic_limit':.10,
            'diagnostic_definition':'(candidate 4096->8192 vector difference + reference space difference + reference time difference)/8192-step error',
            'diagnostic_warning':'empirical differences, not rigorous error bounds or a physical acceptance threshold',
            'conductances_and_initial_field':'unchanged from source experiment',
            'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    references={};refrows=[]
    key='n4096-k4096'
    with np.load(source/f'reference-{key}.npz',allow_pickle=False) as z:
        references[key]=(z['mass_edges'],z['c0'],z['c'])
    refrows.append({'cells':4096,'steps':4096,'origin':'reused previously validated result'})
    for steps in (4096,8192):
        tic=time.monotonic();e,c0,c,diag=solve(flow,8192,steps,T=T,D=D)
        if diag['max_relative_mass_drift']>1e-9 or diag['min_all_times']<diag['initial_min']-1e-10 or diag['max_all_times']>diag['initial_max']+1e-10:
            raise RuntimeError('reference bounds/conservation failure')
        key=f'n8192-k{steps}';references[key]=(e,c0,c)
        refrows.append({'cells':8192,'steps':steps,'origin':'new required refinement','wall_seconds':time.monotonic()-tic,**diag})
        print('reference',key,'done',flush=True)
    for key,(e,c0,c) in references.items():
        np.savez_compressed(output/f'reference-{key}.npz',mass_edges=e,c0=c0,c=c)
    results=[];convergence={};all_vectors={};diagnoses={}
    for cells in config['material_columns']:
        labels=np.linspace(-1,1,cells+1);edges=flow.mu(labels);m=np.diff(edges);c0=initial(edges)
        norm=lambda d:normalized_error(m,c0,d)
        refs={key:project(e,c,edges) for key,(e,i,c) in references.items()};best=refs['n8192-k8192']
        indicators=reference_indicators(refs,m,c0)
        np.savez_compressed(output/f'inputs-cells{cells}.npz',label_edges=labels,mass_edges=edges,mass=m,c0=c0,reference=best,**{f'reference_{key}':r for key,r in refs.items()})
        vectors={}
        for method in METHODS:
            for steps in config['candidate_steps']:
                tic=time.monotonic();c,diag=candidate(flow,edges,steps,method,T=T,D=D)
                vectors[method,steps]=c;all_vectors[cells,method,steps]=c
                results.append({'material_columns':cells,'method':method,'steps':steps,'normalized_error':norm(c-best),'wall_seconds':time.monotonic()-tic,**diag})
                np.savez_compressed(output/f'{method}-cells{cells}-k{steps}.npz',mass=m,c0=c0,c=c,reference=best,mass_edges=edges)
                print('candidate',cells,method,steps,'error',results[-1]['normalized_error'],flush=True)
        for method in METHODS:
            row=next(r for r in results if r['material_columns']==cells and r['method']==method and r['steps']==8192)
            temporal=norm(vectors[method,4096]-vectors[method,8192])
            ratio=(temporal+sum(indicators.values()))/row['normalized_error']
            row.update(reference_refinement=indicators,candidate_temporal_difference=temporal,
                       combined_indicator_over_error=ratio,
                       error_separation_diagnostic_met=bool(ratio<config['error_separation_diagnostic_limit']))
        ref_e,ref_i,ref_c=references['n8192-k8192']
        jf=sampled_reference_flux(flow,ref_e,ref_c,T,D,edges[1:-1])
        diag,arrays=split(flow,edges,best,jf,T,D)
        diagnoses[str(cells)]=diag
        np.savez_compressed(output/f'common-state-flux-cells{cells}.npz',**arrays)
    for method in METHODS:
        values=[next(r['normalized_error'] for r in results if r['material_columns']==cells and r['method']==method and r['steps']==8192) for cells in config['material_columns']]
        convergence[method]={'columns':config['material_columns'],'errors':values,
                             'pairwise_orders':[empirical_order(a,b) for a,b in zip(values[:-1],values[1:])],
                             'monotone_error_decrease':bool(all(a>b for a,b in zip(values[:-1],values[1:])))}
    ranks={str(cells):[r['method'] for r in sorted([r for r in results if r['material_columns']==cells and r['steps']==8192],key=lambda x:x['normalized_error'])] for cells in config['material_columns']}
    separation=all(r['error_separation_diagnostic_met'] for r in results if r['steps']==8192)
    report={'reference_runs':refrows,'candidate_runs':results,'observed_spatial_convergence':convergence,
            'common_state_flux_diagnosis_at_T':diagnoses,'error_ranking':ranks,
            'all_error_separation_diagnostics_met':separation,'third_stage_physical_acceptance':False,
            'limitation':'three levels of one physical case; pairwise orders are observations, not fitted or universal orders'}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print('error separation diagnostics all met:',separation,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
