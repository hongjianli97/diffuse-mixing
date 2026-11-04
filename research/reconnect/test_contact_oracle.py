"""Three small independent fixtures; not extra scientific contact runs."""
import argparse,copy,json,math,unittest
from fractions import Fraction as F
import numpy as np
from t1_uniform_core import pure_labels
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from audit_shear_voronoi_time import audit_geometry
from voronoi_time_action import operators
from voronoi_time16_core import ordered_path
from contact_oracle_core import pure_contact,face_control,derive_material,cartesian_control,evaluate


def fixture():
    N=4;pure=pure_contact(pure_labels(N),np.arange(N*N))
    case={'K':128,'requested_decimal_digits':90,'alpha':['0']*129,'beta':['0']*129}
    data,meta=derive_material(case,pure,N)
    return pure,data,meta


class Checks(unittest.TestCase):
    def test_point_faces_exact_contact_laws_independent_bisectors_and_bad_inputs(self):
        N=8;p=pure_contact(pure_labels(N),np.arange(N*N));graph=build_periodic_faces(p['ids'],p['positions'],p['rho'],.02);packet=coefficient_packet(graph)
        self.assertTrue(face_control(packet,p,N)['all_world_axis_faces_and_two_half_resistances_exact'])
        _,g,_,cert=audit_geometry(packet,p['positions'],p['rho'],p['ids'],[0,4,7,32,63]);self.assertEqual(len(g),128)
        self.assertTrue(cert['all_face_counterparts_and_two_half_resistances_exact'])
        bad=copy.deepcopy(packet);bad['conductances_exact'][0]=str(2*F(bad['conductances_exact'][0]))
        with self.assertRaises(AssertionError):face_control(bad,p,N)
        badlabels=p['positions'].copy();badlabels[0,0]+=1/64
        with self.assertRaises(ValueError):pure_contact(badlabels,p['ids'])

    def test_mass_means_independent_quadrature_and_separate_time_tail_bounds(self):
        p,d,meta=fixture();N=4;z,w=np.polynomial.legendre.leggauss(16);expected=[]
        for x,y in p['positions']:
            xx=x+z[:,None]/(2*N);yy=y+z[None,:]/(2*N)
            a,phase=(16/5,1/5) if x<.5 else (4/5,7/5)
            expected.append(float(np.sum((.5+.2*np.cos(math.pi*(a*xx+phase))*np.cos(2*math.pi*yy))*w[:,None]*w[None,:])/4))
        np.testing.assert_allclose(d['c0'],expected,rtol=0,atol=3*np.finfo(float).eps)
        np.testing.assert_array_equal(d['mass'],p['rho']/16);self.assertGreater(meta['initial_serialization_bridge_absolute'],0)
        tails=meta['tail_certificates_by_time'];self.assertIsNone(tails[0]);self.assertGreater(F(tails[1]['material_mass_norm_tail_bound_exact']),F(tails[2]['material_mass_norm_tail_bound_exact']))
        self.assertTrue(meta['reference_initial_direct_formula_not_truncated_spectral_sum'])

    def test_unchanged_action_vs_cartesian_full_modes_and_adverse_controls(self):
        p,d,meta=fixture();packet=coefficient_packet(build_periodic_faces(p['ids'],p['positions'],p['rho'],.02))
        op=operators(d['mass'],np.array(packet['edges']),map(F,packet['conductances_exact']));control,cm=cartesian_control(d,4)
        run,path=ordered_path(d['mass'],d['c0'],lambda i:op,16)
        # Synthetic reference is the independent semidiscrete solution, not a new PDE case.
        d['reference']=control['states'][[0,8,16]].copy();values=[d['mass'],d['c0']]+list(path.values());before=[a.tobytes() for a in values]
        for a in values:a.setflags(write=False)
        result=evaluate(d,meta,run,path,control,cm);self.assertTrue(result['all_action_and_original_H_controls_passed'])
        bad={key:v.copy() for key,v in path.items()};bad['independent_states'][8,0]+=.01
        self.assertFalse(evaluate(d,meta,run,bad,control,cm)['all_action_and_original_H_controls_passed'])
        bc={key:v.copy() for key,v in control.items()};bc['states'][8,0]+=.01
        self.assertFalse(evaluate(d,meta,run,path,bc,cm)['all_action_and_original_H_controls_passed'])
        self.assertEqual(before,[a.tobytes() for a in values])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
