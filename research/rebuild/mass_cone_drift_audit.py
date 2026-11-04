"""Independently certify the stored wide-cone failure; no new support search."""
from pathlib import Path
import argparse,json
import numpy as np
from mass_moment_cone import transverse_drift_separator


def main(source):
    source=Path(source);output=source/'drift-certificates.json'
    if output.exists():raise RuntimeError('refusing to overwrite drift certificates')
    cfg=json.loads((source/'config.json').read_text())
    with np.load(source/'mass-moment-cones.npz') as z:
        b=z['target'];rows={};controls={}
        for name in cfg['supports']:
            A=z[name+'_A'];rows[name]=transverse_drift_separator(A,b)
            controls[name]={}
            for order in (32,64):
                beta=z[f'direct_beta_order{order}'];G=z[f'direct_Gamma_order{order}']
                target=np.array([*beta,G[0,0],G[0,1],G[1,1]])
                controls[name][str(order)]=transverse_drift_separator(A,target)
    h=cfg['hy'];beta_y=float(b[1]);Gamma_yy=float(b[4]);bound=Gamma_yy/abs(beta_y)
    result={'certificates':rows,'independent_boundary_target_controls':controls,
      'actual_transverse_spacing_m':h,'necessary_upper_transverse_spacing_m':bound,
      'Gamma_yy_minus_h_abs_beta_y_m2_per_s':Gamma_yy-h*abs(beta_y),
      'h_abs_beta_y_over_Gamma_yy':h*abs(beta_y)/Gamma_yy,
      'scope':'one frozen bulk-extended column, state-independent nonnegative generator, identical linear and quadratic mass-average moment constraints',
      'any_integer_virtual_transverse_row_reach':'For q in Z, (q*h)^2 >= h*abs(q*h); adding farther transverse rows does not remove this bound while the minimum row spacing stays h.',
      'horizontal_support_independence':'certificate uses only transverse first and second moments; no horizontal widening can remove it',
      'no_additional_support_or_spacing_search_run':True,'physical_acceptance':False,'original_closed_problem_accepted':False,
      'original_closed_boundary_changed':False,'nonlinear_schemes_ruled_out':False,'necessary_spacing_bound_is_sufficient':False,
      'new_diffusion_backend':False,'fluid_orc_enabled':False}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');a=p.parse_args();main(a.source)
