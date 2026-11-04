"""One exact contact problem: aligned control and three cut-cell resolutions."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import scipy
from contact import Contact,METHODS,mesh,semidiscrete,backward_euler
from spatial_study import normalized_error


def main(output):
    output=Path(output).resolve();root=Path(__file__).resolve().parents[2]
    output.mkdir(parents=True,exist_ok=False)
    f=Contact();T=.1
    config={'study':'independent translating contact with exact finite-domain scalar eigenmode',
            'rho_left':f.rho_left,'rho_right':f.rho_right,'D':f.D,'T':T,'half_length':f.half_length,
            'contact_speed':f.speed,'pressure':'prescribed same constant on both sides',
            'background':'inviscid translating contact, prescribed; passive scalar only, no heat/density feedback',
            'interface_conditions':['c continuous (perfect transmission assumption)','rho D c_x continuous'],
            'boundary':'zero scalar flux at two translating material endpoints',
            'scalar_amplitude_left':f.amplitude,'scalar_amplitude_right':-f.amplitude*f.rho_left/f.rho_right,
            'physical_reference':'exact transmission eigenmode and exact cell mass averages',
            'placements':{'aligned':[32],'cut':[32,64,128]},
            'cut_placement':'move only face at 0 to nominal_h/3; contact lies 3/4 through its cell',
            'methods':METHODS,'time_steps':[512,1024],
            'spatial_error':'matrix exponential of each candidate minus exact physical cell averages',
            'time_error':'backward Euler minus that same candidate matrix exponential',
            'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];vectors={};inputs={}
    for placement,cells in [('aligned',32)]+[('cut',n) for n in (32,64,128)]:
        faces=mesh(cells,placement,f.half_length);m=f.mass(faces);c0=f.cell_average(faces,0.);ref=f.cell_average(faces,T)
        inputs[placement,cells]=(m,c0,ref);err=lambda a:normalized_error(m,c0,a)
        norm0=np.sqrt(m@((c0-m@c0/m.sum())**2));exact_j=f.flux(faces[1:-1],T)
        np.savez_compressed(output/f'inputs-{placement}-cells{cells}.npz',material_faces=faces,lab_faces_t0=faces,lab_faces_final=faces+f.speed*T,
                            mass=m,c0=c0,exact_reference=ref,exact_face_flux=exact_j,rho_cell_average=m/np.diff(faces))
        for method in METHODS:
            g=f.conductance(faces,method);semi,P,diag=semidiscrete(c0,m,g,T)
            numerical={};time_diags={}
            for steps in (512,1024):
                c,d=backward_euler(c0,m,g,T,steps);numerical[steps]=c;time_diags[str(steps)]={'error_to_semidiscrete':err(c-semi),'error_to_physical':err(c-ref),**d}
            closure=-g*np.diff(ref)-exact_j
            fw=(m[:-1]+m[1:])/2
            flux_error=float(np.sqrt(fw@(closure*closure)/(fw@(exact_j*exact_j))))
            row={'placement':placement,'cells':cells,'method':method,'spatial_error':err(semi-ref),
                 'time_runs':time_diags,'BE512_1024_time_error_ratio':time_diags['512']['error_to_semidiscrete']/time_diags['1024']['error_to_semidiscrete'],
                 'relative_common_state_face_flux_error':flux_error,'semidiscrete_diagnostics':diag,
                 'scalar_mass_reference_error':abs(float(m@(ref-c0)))/abs(float(m@c0)),
                 'semidiscrete_relative_mass_drift':abs(float(m@(semi-c0)))/abs(float(m@c0))}
            rows.append(row);vectors[placement,cells,method]=semi
            np.savez_compressed(output/f'{method}-{placement}-cells{cells}.npz',mass=m,c0=c0,g=g,P=P,c_semidiscrete=semi,c_BE512=numerical[512],c_BE1024=numerical[1024],reference=ref,
                                exact_face_flux=exact_j,closure_face_flux=-g*np.diff(ref),material_faces=faces)
            print(placement,cells,method,'spatial',row['spatial_error'],flush=True)
    orders={}
    for method in METHODS:
        es=[next(r['spatial_error'] for r in rows if r['placement']=='cut' and r['cells']==n and r['method']==method) for n in (32,64,128)]
        orders[method]={'cells':[32,64,128],'errors':es,'pairwise_orders':[float(np.log2(es[i]/es[i+1])) for i in (0,1)]}
    aligned_equal={method:float(np.max(abs(f.conductance(mesh(32,'aligned'),method)-f.conductance(mesh(32,'aligned'),'ordinary_volume_center')))) for method in METHODS}
    report={'runs':rows,'cut_cell_spatial_convergence':orders,'aligned_conductance_max_abs_difference':aligned_equal,
            'limitation':'one smooth transmission eigenmode, not initial concentration step, imperfect partition, variable D or coupled thermodynamics',
            'third_stage_physical_acceptance':False}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
