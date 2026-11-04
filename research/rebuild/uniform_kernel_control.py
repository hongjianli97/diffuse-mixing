"""One fixed uniform-medium mode separates finite-h and node-quadrature rates."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.integrate import quad
from positive_kernel_1d import build_exchange,slope_over_radius


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg={'control':'single uniform-density Neumann cosine mode; no width scan',
         'cells':128,'bounds':[-1.,1.],'rho':1.,'D':.02,'h_to_cell_width':2.,'wavenumber':float(np.pi/2),
         'normalization_or_moment_correction_applied':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    faces=np.linspace(-1,1,129);x=(faces[:-1]+faces[1:])/2;V=np.diff(faces);h=2*V;D=cfg['D'];k=cfg['wavenumber']
    a,d=build_exchange(x,V,V,np.ones_like(x),D,h,(-1,1));dx=V[0];H=h[0]
    offsets=np.arange(1,5)*dx
    continuum_rate=4*D*quad(lambda r:float(slope_over_radius(r,H))*(1-np.cos(k*r)),0,2*H,epsabs=1e-13)[0]
    discrete_rate=4*D*dx*np.sum(slope_over_radius(offsets,H)*(1-np.cos(k*offsets)))
    local_rate=D*k*k
    lattice_second=2*dx*np.sum(slope_over_radius(offsets,H)*offsets*offsets)
    c=np.cos(k*(x+1))*np.sinc(dx/4)
    eigen_error=float(np.max(abs(-(a['L']@c)/V+discrete_rate*c)))
    result={'local_PDE_decay_rate':local_rate,'continuous_kernel_decay_rate':continuum_rate,
            'discrete_kernel_decay_rate':float(discrete_rate),
            'finite_width_rate_difference':continuum_rate-local_rate,
            'node_quadrature_rate_difference':float(discrete_rate-continuum_rate),
            'lattice_second_moment_ratio':float(lattice_second),'discrete_mode_max_eigen_residual':eigen_error,
            'warning':'rate decomposition only for this uniform mode, not a decomposition of actual nonuniform contact errors'}
    (output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
