"""Four necessary checks for global periodic FV face accumulation."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import unittest

from cartesian_fv_stream import MATERIAL_PRIOR, stream_action


def fixture(nx=4,ny=3,rho_constant=None,periods=(F(2),F(3)),thickness=F(2)):
    dx,dy = periods[0]/nx,periods[1]/ny
    p = [((F(i)+F(1,2))*dx,(F(j)+F(1,2))*dy) for j in range(ny) for i in range(nx)]
    rho = [F(rho_constant) if rho_constant is not None else F(1+(i%3)) for i in range(nx*ny)]
    return dict(positions=p,ids=list(range(nx*ny)),mass=[r*thickness*dx*dy for r in rho],rho=rho,
        periods=periods,D=F(1,5),thickness=thickness,model_prior=MATERIAL_PRIOR,
        material_means=[F((i*7)%13,13) for i in range(nx*ny)])


class StreamChecks(unittest.TestCase):
    def test_all_periodic_faces_match_independent_cell_gather_and_conserve(self):
        data=fixture();before=deepcopy(data);a=stream_action(**data)
        nx,ny=a.shape;dx,dy=a.spacing;expected=[]
        for j in range(ny):
            for i in range(nx):
                k=j*nx+i;value=F(0)
                for other,axis in [(j*nx+(i-1)%nx,0),(j*nx+(i+1)%nx,0),
                                   (((j-1)%ny)*nx+i,1),(((j+1)%ny)*nx+i,1)]:
                    step=[dx,dy][axis];area=data['thickness']*[dy,dx][axis]
                    g=2*area*data['D']*data['rho'][k]*data['rho'][other]/(step*(data['rho'][k]+data['rho'][other]))
                    value+=g*(data['material_means'][other]-data['material_means'][k])
                expected.append(value/data['mass'][k])
        self.assertEqual(a.rates,tuple(expected));self.assertEqual(a.face_counts_by_axis,(nx*ny,nx*ny))
        self.assertEqual(a.exact_mass_balance,0);self.assertEqual(a.exact_energy_residual,0)
        self.assertGreater(a.minimum_g,0);self.assertGreaterEqual(a.dissipation,0)
        self.assertEqual(data,before)
        order=list(reversed(range(nx*ny)))
        changed={**data,**{key:[data[key][i] for i in order] for key in ['positions','ids','mass','rho','material_means']}}
        permuted=stream_action(**changed)
        self.assertEqual(dict(zip(permuted.ids,permuted.rates)),dict(zip(a.ids,a.rates)))

    def test_homogeneous_material_mean_Fourier_eigenmode_and_extrema(self):
        data=fixture(4,6,rho_constant=3,periods=(F(2),F(3)))
        x=[F(1),F(0),F(-1),F(0)];y=[F(1),F(1,2),F(-1,2),F(-1),F(-1,2),F(1,2)]
        data['material_means']=[F(1,2)+x[i]/10+y[j]/5 for j in range(6) for i in range(4)]
        a=stream_action(**data);dx,dy=a.spacing
        expected=[-2*data['D']/dx**2*x[i]/10-data['D']/dy**2*y[j]/5 for j in range(6) for i in range(4)]
        self.assertEqual(a.rates,tuple(expected))
        values=data['material_means']
        self.assertLessEqual(a.rates[values.index(max(values))],0)
        self.assertGreaterEqual(a.rates[values.index(min(values))],0)
        constant=stream_action(**{**data,'material_means':[F(3,7)]*24})
        self.assertEqual(constant.rates,(F(0),)*24)

    def test_piecewise_linear_transmission_flux_has_correct_area_and_zero_local_divergence(self):
        data=fixture(8,4,periods=(F(1),F(2)),thickness=F(3))
        data['rho']=[F(1) if i%8<4 else F(4) for i in range(32)]
        data['mass']=[r*F(3,16) for r in data['rho']]
        J=F(1,100)
        data['material_means']=[F(1,2)+J*(p[0]-F(1,2))/(r*data['D']) for p,r in zip(data['positions'],data['rho'])]
        a=stream_action(**data,selected_edges=[(3,4,0)])
        self.assertEqual(a.selected_faces[0][4],data['thickness']*a.spacing[1]*J)
        self.assertEqual(a.rates[3],0);self.assertEqual(a.rates[4],0)
        self.assertEqual(stream_action(**{**data,'D':F(0)}).rates,(F(0),)*32)

    def test_bad_mass_or_unstated_geometry_is_rejected_without_repair(self):
        data=fixture();before=deepcopy(data)
        bad=[{'mass':[data['mass'][0]+F(1,10**30),*data['mass'][1:]]},
             {'model_prior':'points only'}, {'D':-1}, {'rho':[0,*data['rho'][1:]]},
             {'positions':[(F(1,7),data['positions'][0][1]),*data['positions'][1:]]},
             {'ids':[1,*data['ids'][1:]]}, {'material_means':data['material_means'][:-1]}]
        for changes in bad:
            with self.subTest(changes=list(changes)):
                with self.assertRaises((ValueError,TypeError)):stream_action(**{**data,**changes})
        with self.assertRaises(TypeError):stream_action(**{**data,'reference_PDE':None})
        self.assertEqual(data,before)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(StreamChecks))
    summary={'checks_run':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),
             'errors':len(result.errors),'skipped':len(result.skipped),'previous_checks_rerun':0}
    if args.json:
        with Path(args.json).open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
