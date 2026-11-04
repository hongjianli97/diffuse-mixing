"""Independent zero-limit geometry audit and strict nonzero/one-ulp rejection."""
from fractions import Fraction as F
import argparse,copy,json,unittest
import numpy as np
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from audit_d0_geometry_core import audit_zero_geometry
from voronoi_time_action import operators
from voronoi_time16_core import ordered_path
from t2_d0_core import evaluate


class Checks(unittest.TestCase):
    def test_zero_limit_geometry_and_corrupted_conductance_rejected(self):
        axis=(np.arange(4)+.5)/4;a=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
        ids=np.arange(16);rho=1+.2*np.cos(2*np.pi*a[:,0])
        graph=build_periodic_faces(ids,a,rho,0.);packet=coefficient_packet(graph)
        edges,g,degree,cert=audit_zero_geometry(packet,a,rho,ids,[])
        self.assertTrue(cert['all_vertices_independently_rebuilt_from_bisectors']);self.assertTrue(all(w==0 for w in g))
        self.assertEqual(len(edges),32);self.assertTrue(all(w==0 for w in degree))
        bad=copy.deepcopy(packet);bad['conductances_exact'][0]='1/1000000000000000000000000000000'
        with self.assertRaises(AssertionError):audit_zero_geometry(bad,a,rho,ids,[])
        self.assertTrue(all(F(v)==0 for v in packet['conductances_exact']))

    def test_identity_acceptance_rejects_one_ulp_and_tiny_diffusion(self):
        m=np.array([.125,.25,.625]);c0=np.array([.2,.4,.8]);ids=np.array([4,7,3]);ij=np.array([[0,1],[1,2]])
        op=operators(m,ij,[F(0),F(0)]);saved=[]
        def persist(step,action,weights):
            saved.append({'shape':np.array(action.symmetric_generator.shape),'symmetric_data':action.symmetric_generator.data.copy(),
              'symmetric_indices':action.symmetric_generator.indices.copy(),'symmetric_indptr':action.symmetric_generator.indptr.copy(),
              'uniformization_data':action.uniformization_matrix.data.copy(),'uniformization_indices':action.uniformization_matrix.indices.copy(),
              'uniformization_indptr':action.uniformization_matrix.indptr.copy(),'g_float':action.conductance.copy(),'weights':weights.copy()})
        before=m.tobytes(),c0.tobytes();m.setflags(write=False);c0.setflags(write=False)
        run,path=ordered_path(m,c0,lambda step:op,16,persist)
        packet={'D_exact':'0','ids':ids.tolist(),'conductances_exact':['0','0']}
        d={'mass':m,'c0':c0,'ids':ids};packets=[packet]*16
        self.assertTrue(evaluate(d,run,path,packets,saved)['T2_registered_D0_case_passed'])
        bad={k:v.copy() for k,v in path.items()};bad['states'][8,0]=np.nextafter(bad['states'][8,0],np.inf)
        self.assertFalse(evaluate(d,run,bad,packets,saved)['T2_registered_D0_case_passed'])
        positive=dict(packet,conductances_exact=['1/1000000000000000000000000000000','0'])
        self.assertFalse(evaluate(d,run,path,[positive]*16,saved)['T2_registered_D0_case_passed'])
        changed=[{k:v.copy() for k,v in item.items()} for item in saved];changed[0]['uniformization_data'][0]=np.nextafter(1.,0.)
        self.assertFalse(evaluate(d,run,path,packets,changed)['T2_registered_D0_case_passed'])
        self.assertEqual(before,(m.tobytes(),c0.tobytes()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
