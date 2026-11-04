"""Exact fixed-ratio limit and one theory-directed uniform joint-refinement path.

This does not modify the primary nonuniform contact benchmark or its widths.
"""
from pathlib import Path
from fractions import Fraction
import argparse,json
import numpy as np
from positive_kernel_1d import slope_over_radius


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config={'scope':'uniform-lattice analytic controls only; no contact PDE simulation',
            'fixed_ratio':2,'normalization_or_diffusivity_correction':False,
            'joint_path_cells':[128,512,2048],
            'joint_path_rule':'h=.25*sqrt(dx), dx=2/N; anchors the existing N128,h/dx=2 point and enforces h->0, dx/h->0',
            'D':.02,'k':float(np.pi/2),'not_a_parameter_fit':True,
            'primary_contact_width_rule_unchanged':'h_i=2*actual_cell_width_i'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    moment=Fraction(0)
    for j in (1,2,3):
        q=Fraction(j,2)
        B=Fraction(63,32)/8*(2+5*q)*(1-q/2)**5
        moment+=2*B*j*j
    rows=[]
    for N in config['joint_path_cells']:
        dx=2/N;h=.25*np.sqrt(dx);j=np.arange(1,int(np.ceil(2*h/dx)));r=j*dx
        rate=4*config['D']*dx*np.sum(slope_over_radius(r,h)*(1-np.cos(config['k']*r)))
        M=2*dx*np.sum(slope_over_radius(r,h)*r*r)
        rows.append({'cells':N,'dx':dx,'h':h,'h_over_dx':h/dx,'decay_rate_over_PDE':float(rate/(config['D']*config['k']**2)),
                     'lattice_second_moment':float(M)})
    result={'exact_fixed_ratio_second_moment':str(moment),'exact_fixed_ratio_deficit':str(1-moment),
            'fixed_ratio_limit_diffusivity_ratio':float(moment),'fixed_ratio_local_PDE_consistency':'failed: limit is not one',
            'joint_refinement_rows':rows,
            'limits':'joint-path scalar Fourier-rate evidence is not proof for nonuniform contact grids, variable h or 2D; no primary coefficients changed'}
    (output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
