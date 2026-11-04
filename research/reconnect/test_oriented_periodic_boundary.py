"""Three preregistered exact-boundary/lossless/world-audit checks."""
from fractions import Fraction as F
import argparse,copy,json,unittest
import numpy as np
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from audit_oriented_world_geometry import audit_world_geometry


def fixture():
    axis=(np.arange(4)+.5)/4;local=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    boundary=OrientedPeriodicRectangle((F(1,3),F(-1,7)),(1,1),((F(3,5),F(-4,5)),(F(4,5),F(3,5))))
    world=[boundary.push(tuple(F(float(v)) for v in q)) for q in local]
    return local,np.arange(16),np.ones(16),boundary,world


class Checks(unittest.TestCase):
    def test_exact_isometry_roundtrip_and_invalid_boundary(self):
        local,ids,rho,boundary,world=fixture();before=local.tobytes();local.setflags(write=False)
        self.assertTrue(any(p[0]<0 for p in world))
        for a,p in zip(local,world):self.assertEqual(boundary.pull(p),tuple(F(float(v)) for v in a))
        packet,wp,cert=build_oriented_periodic_faces(ids,world,rho,.02,boundary)
        self.assertTrue(cert['local_coordinates_float64_lossless']);self.assertEqual(packet['ids'],ids.tolist())
        self.assertTrue(all(F(g)==F(float(.02)) for g in packet['conductances_exact']))
        self.assertEqual(before,local.tobytes())
        for E in (((1,F(1,3)),(0,1)),((1,0),(0,-1)),((.6,-.8),(.8,.6))):
            with self.assertRaises(ValueError):OrientedPeriodicRectangle((0,0),(1,1),E)

    def test_lossy_coordinate_period_identity_and_domain_rejection(self):
        local,ids,rho,boundary,world=fixture()
        bad=list(world);bad[0]=boundary.push((F(1,3),F(1,8)))
        with self.assertRaises(ValueError):build_oriented_periodic_faces(ids,bad,rho,.02,boundary)
        with self.assertRaises(ValueError):OrientedPeriodicRectangle((0,0),(F(1,3),1),((1,0),(0,1)))
        bad=list(world);bad[0]=boundary.push((F(-1,8),F(1,8)))
        with self.assertRaises(ValueError):build_oriented_periodic_faces(ids,bad,rho,.02,boundary)
        with self.assertRaises(ValueError):build_oriented_periodic_faces(np.zeros(16,dtype=int),world,rho,.02,boundary)
        with self.assertRaises(ValueError):build_oriented_periodic_faces(ids.astype(float),world,rho,.02,boundary)
        badrho=rho.copy();badrho[0]=0
        with self.assertRaises(ValueError):build_oriented_periodic_faces(ids,world,badrho,.02,boundary)

    def test_independent_world_audit_rejects_corrupt_lattice_vertex_and_g(self):
        local,ids,rho,boundary,world=fixture();packet,wp,cert=build_oriented_periodic_faces(ids,world,rho,.02,boundary)
        source={'boundary':boundary.packet(),'world_positions_exact':[[str(q) for q in p] for p in world]}
        edges,g,degree,proof=audit_world_geometry(wp,source,ids,rho,.02,[0,5,15])
        self.assertTrue(proof['all_world_vertices_rebuilt_directly_from_world_bisectors']);self.assertEqual(len(edges),32)
        corrupt=[]
        bad=copy.deepcopy(wp);bad['basis_columns_exact'][0][0]='1';corrupt.append(bad)
        bad=copy.deepcopy(wp);bad['vertices_world_exact'][0][0]=str(F(bad['vertices_world_exact'][0][0])+F(1,1000));corrupt.append(bad)
        bad=copy.deepcopy(wp);bad['conductances_exact'][0]=str(F(bad['conductances_exact'][0])+F(1,10**30));corrupt.append(bad)
        for bad in corrupt:
            with self.assertRaises(AssertionError):audit_world_geometry(bad,source,ids,rho,.02,[0,5,15])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
