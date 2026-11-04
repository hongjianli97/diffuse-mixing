"""Six meaningful geometry/action checks; no real shear snapshot is loaded."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import unittest
import numpy as np

from periodic_voronoi_fv import build_periodic_faces,exact_action,polygon_moments,dot,cross,sub
from auxiliary_tensor_fv import build_auxiliary_faces


def oblique():
    x=np.array([(((c+.5)/4+r/8)%1,(r+.5)/4) for r in range(4) for c in range(4)])
    return np.arange(16)*3+7,x


def independent_cell(points,i):
    """All 9-image halfplanes, enumerate their feasible pair intersections."""
    p=[tuple(F(float(v)) for v in row) for row in points];origin=p[i]
    planes=[((F(1),F(0)),F(1,2)),((F(-1),F(0)),F(1,2)),
        ((F(0),F(1)),F(1,2)),((F(0),F(-1)),F(1,2))]
    for j,q in enumerate(p):
        if j==i:continue
        for sx in [-1,0,1]:
            for sy in [-1,0,1]:
                v=(q[0]+sx-origin[0],q[1]+sy-origin[1]);planes.append((v,dot(v,v)/2))
    vertices=set()
    for k,(a,c) in enumerate(planes):
        for b,d in planes[:k]:
            det=cross(a,b)
            if not det:continue
            q=((c*b[1]-a[1]*d)/det,(a[0]*d-c*b[0])/det)
            if all(dot(v,q)<=z for v,z in planes):vertices.add(q)
    ordered=sorted(vertices)
    def hull(seq):
        result=[]
        for q in seq:
            while len(result)>1 and cross(sub(result[-1],result[-2]),sub(q,result[-1]))<=0:result.pop()
            result.append(q)
        return result
    return hull(ordered)[:-1]+hull(ordered[::-1])[:-1]


def by_ids(graph):
    values={}
    for (i,j),g in zip(graph.edges,graph.conductances):
        key=(graph.ids[i],graph.ids[j]);values[key]=values.get(key,F(0))+g
    return values


class PeriodicGeometryChecks(unittest.TestCase):
    def test_exact_nonuniform_tensor_recovery_against_previous_adapter(self):
        xs=[.0625,.25,.5625,.8125];ys=[.125,.375,.625,.875]
        x=np.array([(a,b) for b in ys for a in xs]);ids=np.arange(16);rho=1+np.arange(16)/16
        current=build_periodic_faces(ids,x,rho,.02);old=build_auxiliary_faces(ids,x,rho,.02)
        self.assertEqual(by_ids(current),by_ids(old))
        self.assertEqual(sum(current.areas,F(0)),1)
        self.assertTrue(all(len(v)==4 for v in current.polygons))
        self.assertTrue(current.diagnostics['all_opposite_faces_and_bisectors_exact'])

    def test_nontensor_bravais_hexagons_match_independent_analytic_geometry(self):
        ids,x=oblique();g=build_periodic_faces(ids,x,np.ones(16),.02)
        vertices={(F(0),F(5,32)),(F(1,8),F(3,32)),(F(1,8),-F(3,32)),
            (F(0),-F(5,32)),(-F(1,8),-F(3,32)),(-F(1,8),F(3,32))}
        for area,poly,center in zip(g.areas,g.polygons,g.centroids):
            self.assertEqual(area,F(1,16));self.assertEqual({p for p,_ in poly},vertices)
            self.assertEqual(center,(0,0))
        self.assertEqual(sorted(g.ratios),[F(1,2)]*32+[F(3,4)]*16)

    def test_complete_search_matches_exhaustive_independent_halfplane_intersections(self):
        x=np.array([(a,b) for b in [.125,.5,.875] for a in [.125,.5,.875]])
        x[4]+=[1/32,1/64];g=build_periodic_faces(np.arange(9),x,np.ones(9),.02)
        for i in [0,4,8]:
            expected=independent_cell(x,i)
            self.assertEqual({p for p,_ in g.polygons[i]},set(expected))
            twice=sum((cross(expected[k-1],v) for k,v in enumerate(expected)),F(0))
            self.assertEqual(g.areas[i],twice/2)

    def test_permutation_and_periodic_translation_preserve_exact_shared_action(self):
        ids,x=oblique();rho=1+np.arange(16)/32;m=.03+np.arange(16)/128;c=.4+np.arange(16)/64
        g=build_periodic_faces(ids,x,rho,.02);q=exact_action(g,m,c)
        p=np.array([7,1,12,3,4,15,6,0,8,9,10,11,2,13,14,5])
        other=build_periodic_faces(ids[p],x[p],rho[p],.02)
        self.assertEqual(by_ids(other),by_ids(g));self.assertEqual(exact_action(other,m[p],c[p]),tuple(q[i] for i in p))
        shifted=build_periodic_faces(ids,(x+[.25,.125])%1,rho,.02)
        self.assertEqual(by_ids(shifted),by_ids(g));self.assertEqual(exact_action(shifted,m,c),q)

    def test_mass_energy_dense_action_constant_and_D_zero_controls(self):
        ids,x=oblique();rho=1+np.arange(16)/32;m=.03+np.arange(16)/128;c=.4+np.arange(16)/64
        g=build_periodic_faces(ids,x,rho,.02);q=exact_action(g,m,c);M=list(map(lambda v:F(float(v)),m));u=list(map(lambda v:F(float(v)),c))
        self.assertEqual(sum((a*b for a,b in zip(M,q)),F(0)),0)
        energy=sum((w*(u[j]-u[i])**2 for (i,j),w in zip(g.edges,g.conductances)),F(0))
        self.assertEqual(-sum((a*b*d for a,b,d in zip(M,u,q)),F(0)),energy)
        L=np.zeros((16,16))
        for (i,j),w in zip(g.edges,g.conductances):
            w=float(w);L[i,i]+=w;L[j,j]+=w;L[i,j]-=w;L[j,i]-=w
        np.testing.assert_allclose(-L@c/m,list(map(float,q)),rtol=2e-13,atol=2e-14)
        self.assertTrue(all(v==0 for v in exact_action(g,m,np.ones(16)*.37)))
        zero=build_periodic_faces(ids,x,rho,0)
        self.assertTrue(all(v==0 for v in zero.conductances));self.assertTrue(all(v==0 for v in exact_action(zero,m,c)))

    def test_invalid_inputs_and_reference_keyword_are_rejected_without_repair(self):
        ids,x=oblique();rho=np.ones(16)
        for d in [-1,float('nan'),True]:
            with self.assertRaises(ValueError):build_periodic_faces(ids,x,rho,d)
        with self.assertRaises(ValueError):build_periodic_faces(ids,x,rho*0,.02)
        with self.assertRaises(ValueError):build_periodic_faces(ids,x+2,rho,.02)
        with self.assertRaises(ValueError):build_periodic_faces(np.zeros(16,dtype=int),x,rho,.02)
        bad=x.copy();bad[0]=bad[1]
        with self.assertRaises(ValueError):build_periodic_faces(ids,bad,rho,.02)
        with self.assertRaises(TypeError):build_periodic_faces(ids,x,rho,.02,c0=np.ones(16))
        g=build_periodic_faces(ids,x,rho,.02)
        with self.assertRaises(ValueError):exact_action(g,np.zeros(16),np.ones(16))
        with self.assertRaises(ValueError):exact_action(g,np.ones(16),np.ones(16)*np.nan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PeriodicGeometryChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(not result.wasSuccessful())
