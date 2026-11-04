"""New material-identity/parent-observation checks; no fluid or solver reruns."""
from fractions import Fraction as F
import argparse,json,unittest
import numpy as np
from voronoi_spatial_projection import validate_record,project_path


def fixture():
    n=2;row,col=np.indices((4,4));index=((row//2)*2+col//2).ravel()
    m=np.array([1,2,1,3,2,3,2,1,1,2,1,1,3,2,2,4],dtype=float)/8
    mp=np.array([sum(F(float(m[j])) for j in range(16) if index[j]==i) for i in range(4)],dtype=float)
    pids=np.array([19,7,31,13]);ids=np.arange(16)*3+17
    return {'mass':m,'parent_mass':mp,'ids':ids,'parent_ids':pids,'parent_index':index,'parent_id_per_child':pids[index]}


class Checks(unittest.TestCase):
    def test_independent_exact_component_integrals(self):
        d=fixture();c=np.arange(16,dtype=float)/32
        expected=[]
        for i,mp in enumerate(d['parent_mass']):
            q=sum((F(float(m))*F(float(v)) for m,v,j in zip(d['mass'],c,d['parent_index']) if j==i),F(0))
            expected.append(float(q/F(float(mp))))
        observed=project_path(np.array([c]),d)[0]
        np.testing.assert_array_equal(observed,expected)
        self.assertEqual(float(d['parent_mass']@observed),float(d['mass']@c))

    def test_raw_parent_rounding_not_normalized(self):
        d={'mass':np.array([.125,.125,.25,.5]),'parent_mass':np.array([np.nextafter(1.,np.inf)]),
           'ids':np.array([8,2,7,4]),'parent_ids':np.array([20]),'parent_index':np.zeros(4,dtype=int),'parent_id_per_child':np.full(4,20)}
        child,rounding,audit=validate_record(d)
        after=project_path(np.ones((1,4)),d)[0,0]
        self.assertEqual(after,1/d['parent_mass'][0]);self.assertLess(after,1.)
        self.assertNotEqual(rounding[0],0.)
        self.assertFalse(audit['child_mass_renormalized']);self.assertFalse(audit['saved_parent_mass_modified'])

    def test_resolved_and_subcell_variance_identity(self):
        d=fixture();c=np.arange(16,dtype=float)/32
        p=project_path(np.array([c]),d)[0];mean=float(d['mass']@c/d['mass'].sum())
        fine=float(d['mass']@((c-mean)**2));parent=float(d['parent_mass']@((p-mean)**2))
        subcell=float(d['mass']@((c-p[d['parent_index']])**2))
        self.assertAlmostEqual(fine,parent+subcell,places=15)
        self.assertGreater(subcell,0.);self.assertLess(parent,fine)

    def test_invalid_material_identity_or_state_rejected(self):
        d=fixture()
        for key,value in [('parent_index',d['parent_index'][::-1]),('parent_id_per_child',np.zeros(16,dtype=int)),
                          ('ids',np.zeros(16,dtype=int)),('parent_index',d['parent_index'].astype(float))]:
            wrong=dict(d);wrong[key]=value
            with self.assertRaises(ValueError):validate_record(wrong)
        with self.assertRaises(ValueError):project_path(np.zeros((1,15)),d)
        with self.assertRaises(ValueError):project_path(np.full((1,16),np.nan),d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
