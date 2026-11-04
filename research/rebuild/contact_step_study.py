"""One contact scalar-step problem: early layer, mass averaging and exchange."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import scipy
from scipy.linalg import eigh
from contact import Contact,METHODS,mesh,laplacian,semidiscrete,backward_euler
from contact_step_reference import StepReference,right_phase_inventory_bounds
from spatial_study import normalized_error


def main(output):
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    flow=Contact();ref=StepReference(flow);times=(1e-4,1e-3,1e-2,.1)
    config={'study':'same contact with initial scalar step and finite-domain exact series',
            'rho_left':flow.rho_left,'rho_right':flow.rho_right,'D':flow.D,'half_length':flow.half_length,'speed':flow.speed,
            'initial_left':ref.left,'initial_right':ref.right,'times':times,'placements':{'aligned':[32],'cut':[32,64,128]},
            'methods':METHODS,'reference_tolerance':1e-13,'reference_tail':'infinite geometric majorant in concentration norm',
            'relative_inventory_diagnostic':'undefined when reference inventory is below requested concentration tolerance times region mass; absolute errors and raw reference retained',
            'time_separation':'candidate semidiscrete matrix exponential; BE512/1024 check only cut64',
            'subcell_inventory_warning':'whole right-phase inventory is not uniquely determined by a single straddling-cell average',
            'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    refmeta={};reference_inventory={};all_rows=[];initial_rows=[]
    for t in times:
        K=ref.terms_for(t);spectral_inv=float(flow.rho_right*flow.half_length*ref.average(np.array([0.,flow.half_length]),t,K)[0])
        inv,image_bound,image_terms=ref.right_inventory(0.,t)
        refmeta[str(t)]={'terms':K,'uniform_concentration_tail_bound':ref.tail_bound(t,K),'uniform_flux_tail_bound':ref.flux_tail_bound(t,K),
                         'exact_interface_concentration':ref.mean,'interface_flux':float(ref.flux(np.array(0.),t,K)),
                         'whole_right_phase_inventory':inv,'inventory_image_tail_bound':image_bound,'inventory_image_log_tail_bound':float(np.log(flow.rho_right*flow.half_length)+ref.image_log_tail_bound(t,image_terms)),'image_terms':image_terms,
                         'inventory_spectral_vs_images_absolute_difference':abs(inv-spectral_inv)}
        reference_inventory[t]=inv
    for placement,cells in [('aligned',32)]+[('cut',n) for n in (32,64,128)]:
        faces=mesh(cells,placement);m=flow.mass(faces);c0=ref.average(faces,0.)
        ml=flow.rho_left*(np.minimum(faces[1:],0)-np.minimum(faces[:-1],0))
        mr=flow.rho_right*(np.maximum(faces[1:],0)-np.maximum(faces[:-1],0))
        pure_right=faces[:-1]>=0;cut=(faces[:-1]<0)&(faces[1:]>0)
        continuum_E0=.5*flow.half_length*flow.rho_left*flow.rho_right/(flow.rho_left+flow.rho_right)*(ref.left-ref.right)**2
        discrete_E0=.5*float(m@((c0-ref.mean)**2))
        uniform_R0=float(mr@c0);bounds0=right_phase_inventory_bounds(c0,m,ml,mr)
        initial_rows.append({'placement':placement,'cells':cells,'continuum_initial_energy':continuum_E0,'cell_average_initial_energy':discrete_E0,
                             'initial_projection_energy_loss':continuum_E0-discrete_E0,
                             'uniform_subcell_reconstruction_initial_right_mass':uniform_R0,
                             'actual_initial_right_mass':0.,'initial_right_inventory_admissible_bounds':bounds0})
        refs={};refs2={}
        for t in times:
            K=refmeta[str(t)]['terms'];refs[t]=ref.average(faces,t,K);refs2[t]=ref.average(faces,t,2*K)
        np.savez_compressed(output/f'inputs-{placement}-{cells}.npz',material_faces=faces,mass=m,left_subcell_mass=ml,right_subcell_mass=mr,c0=c0,
                            times=np.array(times),reference=np.array([refs[t] for t in times]))
        for method in METHODS:
            g=flow.conductance(faces,method);c_out=[];be512_out=[];be1024_out=[]
            for t in times:
                c,P,diag=semidiscrete(c0,m,g,t);c_out.append(c)
                # Independent mass-symmetric eigendecomposition cross-check.
                rm=np.sqrt(m);lam,U=eigh(laplacian(g)/rm[:,None]/rm[None,:])
                independent=(U@(np.exp(-t*lam)*(U.T@(rm*c0))))/rm
                cross=float(np.max(abs(c-independent)))
                if cross>2e-12:raise RuntimeError('semidiscrete algorithm disagreement')
                norm=lambda d:normalized_error(m,c0,d)
                pure_num=float(m[pure_right]@c[pure_right]);pure_spectral=float(m[pure_right]@refs[t][pure_right])
                region_start=float(faces[np.flatnonzero(pure_right)[0]])
                pure_exact,image_bound,image_terms=ref.right_inventory(region_start,t)
                lower,upper=right_phase_inventory_bounds(c,m,ml,mr)
                row={'placement':placement,'cells':cells,'method':method,'time':t,
                     'spatial_error':norm(c-refs[t]),'reference_K_to_2K_difference':norm(refs[t]-refs2[t]),
                     'semidiscrete_relative_mass_drift':abs(float(m@(c-c0)))/abs(float(m@c0)),
                     'semidiscrete_crosscheck_max_abs_difference':cross,'semidiscrete_diagnostics':diag,
                     'right_resolved_region_num_mass':pure_num,'right_resolved_region_exact_mass':pure_exact,
                     'right_resolved_region_spectral_mass_raw':pure_spectral,
                     'right_resolved_region_images_tail_bound':image_bound,
                     'right_resolved_region_images_log_tail_bound':float(np.log(flow.rho_right*(flow.half_length-region_start))+ref.image_log_tail_bound(t,image_terms)),
                     'right_resolved_region_reference_method':'stable finite-domain erfc-image integral',
                     'right_resolved_region_absolute_error':abs(pure_num-pure_exact),
                     'right_resolved_region_reference_mass_resolution':float(config['reference_tolerance']*m[pure_right].sum()),
                     'right_resolved_region_relative_error':abs(pure_num-pure_exact)/pure_exact if pure_exact>config['reference_tolerance']*m[pure_right].sum() else None,
                     'right_resolved_region_relative_status':'resolved_reference_mass' if pure_exact>config['reference_tolerance']*m[pure_right].sum() else 'below_requested_reference_mass_resolution',
                     'right_resolved_region_start':float(faces[np.flatnonzero(pure_right)[0]]),
                     'whole_right_phase_exact_mass':reference_inventory[t],
                     'whole_right_phase_uniform_reconstruction':float(mr@c),
                     'whole_right_phase_state_inventory_bounds':[lower,upper],
                     'whole_right_phase_identifiable_from_means':bool(not cut.any()),
                     'layer_length_over_nominal_h':float(np.sqrt(flow.D*t)/(2*flow.half_length/cells))}
                if placement=='cut' and cells==64:
                    c512,d512=backward_euler(c0,m,g,t,512);c1024,d1024=backward_euler(c0,m,g,t,1024)
                    be512_out.append(c512);be1024_out.append(c1024)
                    row['BE_time_errors']={'512':norm(c512-c),'1024':norm(c1024-c),'diagnostics512':d512,'diagnostics1024':d1024}
                all_rows.append(row)
                print(placement,cells,method,t,'error',row['spatial_error'],'resolved-right-relative',row['right_resolved_region_relative_error'],flush=True)
            arrays={'times':np.array(times),'c_semidiscrete':np.array(c_out),'reference':np.array([refs[t] for t in times]),'mass':m,'c0':c0,'g':g,'material_faces':faces}
            if be512_out:arrays.update(c_BE512=np.array(be512_out),c_BE1024=np.array(be1024_out))
            np.savez_compressed(output/f'{method}-{placement}-{cells}.npz',**arrays)
    report={'reference':refmeta,'initial_projection':initial_rows,'runs':all_rows,
            'third_stage_physical_acceptance':False,
            'warning':'subcell inventory bounds are identifiability bounds of each represented state, not discretization error bounds; no inventory estimate used to renormalize the solution'}
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    files=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
