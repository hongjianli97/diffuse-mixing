"""One saved Becker case: explicit face records and temporal interpolation.

Coefficients consume serialized sampled coordinates and fixed mass only.
Analytic flow is confined to independent geometry/materiality diagnostics.
"""
from pathlib import Path
import argparse,hashlib,json,platform
import numpy as np
import scipy
from becker import Becker
from density_geometry_study import run_history,mass_coordinate_at_x,velocity_at_x
from kernel_joint_study import verify
from material_face_history import MaterialFaceHistory,SI_UNITS
from spatial_study import normalized_error


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve();verify(root,source)
    old=json.loads((source/'config.json').read_text());p=old['physical_parameters']
    space=root/'runs'/Path(old['source']).name;verify(root,space)
    f=Becker(delta=p['delta'],mach=p['Mach'],gamma=p['gamma'],rho1=p['rho1'],p1=p['p1'],R=p['R'])
    T,D=p['T'],p['D']
    with np.load(source/'geometry-audit.npz') as z:
        input_times=z['time'];input_faces=z['material_faces'];m=z['mass'];c0=z['c0'];edges=z['mass_edges']
    with np.load(space/'inputs-cells32.npz') as z:
        reference=z['reference'];refspace=z['reference_n4096-k4096']-z['reference_n8192-k4096']
        reftime=z['reference_n8192-k4096']-z['reference_n8192-k8192']
    n=len(m);face_ids=np.array([f'face-{i:03d}' for i in range(n+1)])
    cell_ids=np.array([f'cell-{i:03d}' for i in range(n)])
    pairs=np.column_stack((face_ids[:-1],face_ids[1:]))
    output.mkdir(parents=True,exist_ok=False)
    cfg={'study':'same Becker32 FV, explicit material-face input and sampled-time transport',
         'source':str(source),'physical_parameters':p,'sampling_intervals':[8,16,32],
         'candidate_steps':[4096,8192],'interpolation':'shared piecewise-linear face coordinates; interval-constant face velocity',
         'inputs':'fixed face/cell IDs and oriented connectivity, fixed m/c0, SI units, no-flux material boundary',
         'oracle_usage':'benchmark materiality audit only; serialized input positions are synthetic exact faces at sampling knots',
         'same_backend':'cellwise mean-density center-resistance FV, reused prior B baseline',
         'diagnostic_error_budget_limit':old['diagnostic_error_budget_limit'],
         'diagnostic_scope':'empirical numerical separation only, not physical acceptance',
         'no_extrapolation_sorting_remap_clipping_or_normalization':True,
         'physical_acceptance':False,'fluid_orc_enabled':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    times=np.linspace(0,T,8193);true=f.x_mu(edges[None,:],times[:,None]);Vtrue=np.diff(true,axis=1)
    base={}
    for steps in cfg['candidate_steps']:
        with np.load(source/f'material_faces_mean_density-k{steps}.npz') as z:base[steps]=z['c']
    oldr=json.loads((source/'results.json').read_text())
    reference_indicator=normalized_error(m,c0,refspace)+normalized_error(m,c0,reftime)
    sampled=np.arange(0,len(times),64);geometries=[];runs=[];vectors={};geodata={}
    baseline_identity=None
    for intervals in cfg['sampling_intervals']:
        take=np.arange(0,len(input_times),(len(input_times)-1)//intervals)
        record=MaterialFaceHistory(input_times[take],input_faces[take],face_ids,cell_ids,pairs,m,c0,
                  SI_UNITS,{'geometry_kind':'material_faces','source_file_sha256':hashlib.sha256((source/'geometry-audit.npz').read_bytes()).hexdigest(),
                            'provenance':'synthetic exact material faces at knots; between-knot materiality remains approximate'})
        record.save(output/f'face-record-{intervals}.npz')
        record=MaterialFaceHistory.load(output/f'face-record-{intervals}.npz')
        if baseline_identity is None:baseline_identity=record.identity_digest
        if record.identity_digest!=baseline_identity:raise RuntimeError('sampling changed identities, mass, c0 or connectivity')
        modeled=record.evaluate(times);x=modeled['positions'];V=modeled['volume'];rho=modeled['density']
        mass_in_cells=np.diff(mass_coordinate_at_x(f,x,times[:,None]),axis=1)
        u=velocity_at_x(f,x,times[:,None]);rho_actual=f.j/f.u(f.z_at_xi(f.u1*times[:,None]-x))
        fluid_flux=rho_actual*(u-modeled['face_velocity'])
        g=record.conductance(times,D)
        gc=record.geometric_conservation()
        geometry={'sampling_intervals':intervals,'frames':len(take),'sample_dt':T/intervals,
                  'max_face_position_error':float(np.max(abs(x-true))),
                  'max_relative_volume_error':float(np.max(abs(V-Vtrue)/Vtrue)),
                  'max_relative_mean_density_error':float(np.max(abs(m/V-m/Vtrue)/(m/Vtrue))),
                  'max_actual_fluid_mass_relative_mismatch':float(np.max(abs(mass_in_cells-m)/m)),
                  'max_knot_fluid_mass_relative_mismatch':float(np.max(abs(mass_in_cells[np.arange(0,len(times),8192//intervals)]-m)/m)),
                  'maximum_face_fluid_mass_flux':float(np.max(abs(fluid_flux))),
                  **gc}
        geometries.append(geometry);geodata[intervals]=geometry
        np.savez_compressed(output/f'geometry-{intervals}.npz',time=times[sampled],modeled_faces=x[sampled],true_faces=true[sampled],
                            modeled_volume=V[sampled],true_volume=Vtrue[sampled],density=rho[sampled],
                            true_fluid_mass_in_modeled_cells=mass_in_cells[sampled],face_velocity=modeled['face_velocity'][sampled],
                            face_fluid_mass_flux=fluid_flux[sampled],conductance=g[sampled],mass=m,c0=c0,cell_ids=cell_ids)
        for steps in cfg['candidate_steps']:
            stride=8192//steps;c,checks=run_history(m,c0,g[:-1:stride],T);vectors[intervals,steps]=c
            row={'sampling_intervals':intervals,'steps':steps,'total_error':normalized_error(m,c0,c-reference),
                 'extra_error_vs_exact_face_history':normalized_error(m,c0,c-base[steps]),**checks}
            runs.append(row)
            np.savez_compressed(output/f'scalar-intervals{intervals}-steps{steps}.npz',c=c,reference=reference,mass=m,c0=c0,
                                exact_face_baseline=base[steps],cell_ids=cell_ids,extra_change=c-base[steps])
            print('face intervals',intervals,'steps',steps,'sampling change',row['extra_error_vs_exact_face_history'],flush=True)
    comparisons=[]
    for intervals in cfg['sampling_intervals']:
        c=vectors[intervals,8192];change=c-base[8192]
        time_diff=normalized_error(m,c0,vectors[intervals,4096]-c)
        paired=normalized_error(m,c0,(vectors[intervals,4096]-base[4096])-change)
        total=normalized_error(m,c0,c-reference);extra=normalized_error(m,c0,change)
        comparisons.append({'sampling_intervals':intervals,'total_error':total,'extra_error':extra,
                            'candidate_time_difference':time_diff,'reference_indicator':reference_indicator,
                            'total_numerical_indicator_over_error':(time_diff+reference_indicator)/total,
                            'paired_time_difference_in_sampling_change':paired,
                            'paired_time_difference_over_sampling_change':paired/extra,
                            'total_numerical_separation_met':(time_diff+reference_indicator)/total<cfg['diagnostic_error_budget_limit'],
                            'sampling_effect_numerical_separation_met':paired/extra<cfg['diagnostic_error_budget_limit']})
    orders={}
    for key in ['max_relative_volume_error','max_actual_fluid_mass_relative_mismatch']:
        e=[q[key] for q in geometries];orders[key]=[float(np.log2(a/b)) for a,b in zip(e[:-1],e[1:])]
    e=[q['extra_error'] for q in comparisons];orders['scalar_sampling_change']=[float(np.log2(a/b)) for a,b in zip(e[:-1],e[1:])]
    result={'input_contract_identity_digest':baseline_identity,'geometry_controls':geometries,'candidate_runs':runs,
            'final_comparison':comparisons,'pairwise_observed_orders':orders,
            'unchanged_exact_face_baseline_error':next(q['normalized_error'] for q in oldr['final_comparison'] if q['method']=='material_faces_mean_density'),
            'exact_face_baseline_temporal_difference':oldr['uncertainty']['method_time_differences']['material_faces_mean_density'],
            'physical_acceptance':False,
            'limits':['producer materiality declaration is not independently provable from this contract',
                      'linear GCL is not exact fluid-material tracking between frames',
                      'one smooth 1D Becker problem, no ideal discontinuity or 2D acceptance',
                      'sampling refinement differences are observations, not rigorous error bounds']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'BLAS_threads_command':4},indent=2)+'\n')
    print(json.dumps({'geometry':geometries,'comparison':comparisons,'orders':orders},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
