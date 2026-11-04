"""Fixed two-mode contact control, not a scan or changed baseline."""
from dataclasses import dataclass
from pathlib import Path
import argparse,json
import numpy as np
from contact import Contact,METHODS,mesh,semidiscrete,backward_euler
from spatial_study import normalized_error


@dataclass(frozen=True)
class TwoModeContact(Contact):
    amplitude: float=.1
    even_amplitude: float=.1

    def scalar(self,a,t):
        a=np.asarray(a);q=np.pi/self.half_length
        return super().scalar(a,t)+self.even_amplitude*np.exp(-self.D*q*q*t)*np.cos(q*(a+self.half_length))

    def cell_average(self,faces,t):
        a=np.asarray(faces[:-1]);b=np.asarray(faces[1:]);q=np.pi/self.half_length
        F=lambda x:np.sin(q*(x+self.half_length))/q
        integral=self.rho_left*(F(np.minimum(b,0))-F(np.minimum(a,0)))
        integral+=self.rho_right*(F(np.maximum(b,0))-F(np.maximum(a,0)))
        return super().cell_average(faces,t)+self.even_amplitude*np.exp(-self.D*q*q*t)*integral/self.mass(faces)

    def flux(self,a,t):
        a=np.asarray(a);q=np.pi/self.half_length;rho=np.where(a<=0,self.rho_left,self.rho_right)
        return super().flux(a,t)+rho*self.D*self.even_amplitude*q*np.exp(-self.D*q*q*t)*np.sin(q*(a+self.half_length))


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg={'scope':'single 64-column cut-cell compatible two-mode contact control',
         'purpose':'remove the first eigenmode property c_t(contact)=0; do not extrapolate baseline order',
         'columns':64,'odd_amplitude_left':.1,'even_amplitude':.1,'D':.02,'T':.1,
         'placement':'same cut-cell mesh as baseline','methods':METHODS,
         'physical_background':'same rho=2/1, rigid speed=.3, perfect scalar transmission',
         'time_check_steps':1024,'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    f=TwoModeContact();faces=mesh(64,'cut');m=f.mass(faces);c0=f.cell_average(faces,0.);ref=f.cell_average(faces,.1)
    err=lambda a:normalized_error(m,c0,a);rows=[]
    for method in METHODS:
        g=f.conductance(faces,method);c,P,d=semidiscrete(c0,m,g,.1)
        be,bd=backward_euler(c0,m,g,.1,1024)
        rows.append({'method':method,'spatial_error':err(c-ref),'BE1024_error_to_semidiscrete':err(be-c),
                     'semidiscrete_diagnostics':d,'BE_diagnostics':bd})
        np.savez_compressed(output/f'{method}.npz',material_faces=faces,mass=m,c0=c0,reference=ref,c_semidiscrete=c,c_BE1024=be,g=g)
    (output/'results.json').write_text(json.dumps({'runs':rows,'third_stage_physical_acceptance':False},indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
