"""Matched material-average comparison: new 1D kernel versus validated FV."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import scipy
from scipy.linalg import expm,eigh
from positive_kernel_1d import build_exchange
from spatial_study import normalized_error


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    old=json.loads((source/'sha256.json').read_text())
    for path,digest in old.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source evidence changed: '+path)
    cfg=json.loads((source/'config.json').read_text());rows0=json.loads((source/'results.json').read_text());times=cfg['times']
    with np.load(source/'mesh-q8.npz') as z:faces=z['faces'];m=z['child_mass'];c0=z['child_c0'];A=z['A'];mp=z['parent_mass'];cp=z['parent_c0']
    physical=cfg['physical_configuration'];rhoL=physical['rho_left'];rhoR=physical['rho_right'];D=physical['D']
    x=(faces[:-1]+faces[1:])/2;V=np.diff(faces);rho=np.where(x<0,rhoL,rhoR);h=2*V
    output.mkdir(parents=True,exist_ok=False)
    config={'study':'new 1D oracle kernel exchange on exactly the fixed 436-cell material-average problem',
            'source':str(source),'physical_configuration':physical,'times':times,
            'state':'same fixed material cell mass averages, not trajectory point samples',
            'positions':'geometric centers of the same actual material cells',
            'volumes':'actual material cell lengths, unit transverse area; m/rho checked, not estimated',
            'rho_source':'prescribed exact point oracle; pure-phase cells make it equal to cell average',
            'kernel':'C1/h*(1-q/2)^6*(1+3q+35q²/12), q<2; C1=27/32',
            'conductance':'2 V_i V_j harmonic(rho_i D,rho_j D)*[-Wprime_hij(r)/r]',
            'width_rule':'h_i=2*actual_cell_width_i, h_ij=(h_i+h_j)/2; fixed before solving',
            'boundary':'even scalar/density reflection at the same two no-flux material endpoints; no ghost unknowns',
            'neighbor_rule':'all unordered pairs evaluated, including every pair inside symmetric support',
            'main_time_evolution':'semidiscrete matrix exponential for both candidate and prior FV spatial comparison',
            'limitations':['1D candidate, not original 2D kernel restoration','finite kernel width and node quadrature both present','oracle geometry and rho only'],
            'physical_acceptance':False,'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    arrays,diag=build_exchange(x,m,V,rho,D,h,(faces[0],faces[-1]));L=arrays['L']
    np.savez_compressed(output/'kernel-operator.npz',**arrays,material_faces=faces,c0=c0,A=A,parent_mass=mp,parent_c0=cp)
    rootm=np.sqrt(m);vals,U=eigh(L/rootm[:,None]/rootm[None,:]);right=x>0;rows=[]
    cross=(x[:,None]<0)&(x[None,:]>0)
    for t in times:
        P=expm(-t*L/m[:,None]);c=P@c0
        independent=(U@(np.exp(-t*vals)*(U.T@(rootm*c0))))/rootm
        difference=float(np.max(abs(c-independent)))
        pd={'minimum_entry':float(P.min()),'row_sum_residual':float(np.max(abs(P.sum(axis=1)-1))),
            'relative_mass_invariant_residual':float(np.max(abs(m@P-m))/m.max())}
        if difference>1e-10 or pd['minimum_entry']< -1e-10 or max(pd['row_sum_residual'],pd['relative_mass_invariant_residual'])>1e-10:
            raise RuntimeError('kernel algebra or independent evolution check failed')
        if c.min()<c0.min()-1e-10 or c.max()>c0.max()+1e-10:raise RuntimeError('kernel concentration bound failure')
        with np.load(source/f'result-q8-t{t:g}.npz') as z:ref=z['child_reference'];pref=z['parent_reference'];fv=z['child_solution']
        baseline=next(r for r in rows0['runs'] if r['resolution']==8 and r['time']==t);inv_ref=baseline['right_inventory_reference']
        inventory=float(m[right]@c[right]);flux=float(np.sum((arrays['G']*(c[:,None]-c[None,:]))[cross]))
        row={'time':t,'kernel_full_error':normalized_error(m,c0,c-ref),'FV_full_error':baseline['full_resolution_error'],
             'kernel_parent_error':normalized_error(mp,cp,A@c-pref),'FV_parent_error':baseline['aggregated_error'],
             'kernel_right_inventory':inventory,'right_inventory_reference':inv_ref,
             'kernel_inventory_signed_relative_error':float((inventory-inv_ref)/inv_ref),
             'FV_inventory_signed_relative_error':baseline['right_inventory_signed_relative_error'],
             'kernel_total_cross_contact_flux':flux,'relative_scalar_mass_drift':abs(float(m@(c-c0)))/abs(float(m@c0)),
             'independent_evolution_max_abs_difference':difference,'propagator_checks':pd}
        rows.append(row)
        np.savez_compressed(output/f'result-t{t:g}.npz',time=np.array(t),kernel_c=c,FV_c=fv,reference=ref,mass=m,c0=c0,positions=x,h=h,parent_kernel=A@c,parent_FV=A@fv,parent_reference=pref)
        print('time',t,'kernel inventory error',row['kernel_inventory_signed_relative_error'],'FV',row['FV_inventory_signed_relative_error'],flush=True)
    (output/'results.json').write_text(json.dumps({'operator_diagnostics':diag,'runs':rows,'physical_acceptance':False,
         'warning':'kernel graph algebra is not PDE consistency; no width tuning, mass normalization or density substitution'},indent=2))
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
    paths=[p for p in output.iterdir() if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
    (output/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
