"""Five new checks for the restricted positive auxiliary-face adapter."""
from pathlib import Path
import argparse
import json
import math
import unittest
import numpy as np

from auxiliary_tensor_fv import build_auxiliary_faces,exact_action,independent_float_action


def fixture():
    xs=np.array([.07,.28,.56,.83]);ys=np.array([.06,.25,.46,.66,.87])
    x=np.array([(a,b) for b in ys for a in xs]);ids=np.arange(len(x))*3+10
    rho=1+.13*np.arange(len(x));mass=.03+.002*np.arange(len(x))
    state=np.array([.4+.02*((i*i+3*i)%13) for i in range(len(x))])
    return ids,x,rho,mass,state


def dense_oracle(x,rho,mass,D):
    # Independent midpoint geometry, resistance denominator and dense incidence.
    xs=sorted(set(x[:,0]));ys=sorted(set(x[:,1]));n=len(x);Q=np.zeros((n,n))
    index={tuple(v):i for i,v in enumerate(x)}
    for row,y in enumerate(ys):
        for col,a in enumerate(xs):
            i=index[(a,y)]
            for axis,values,k,other,otherk in [(0,xs,col,ys,row),(1,ys,row,xs,col)]:
                nxt=(k+1)%len(values);d=(values[nxt]-values[k])%1
                left=(other[otherk]-other[otherk-1])%1;right=(other[(otherk+1)%len(other)]-other[otherk])%1
                face=(left+right)/2
                j=index[(values[nxt],y)] if axis==0 else index[(a,values[nxt])]
                g=0 if D==0 else face/(d/(2*rho[i]*D)+d/(2*rho[j]*D))
                Q[i,j]+=g/mass[i];Q[i,i]-=g/mass[i];Q[j,i]+=g/mass[j];Q[j,j]-=g/mass[j]
    return Q


class AuxiliaryFaceChecks(unittest.TestCase):
    def test_variable_density_unequal_capacity_actions_match_independent_dense_operator(self):
        ids,x,rho,m,c=fixture();before=[a.tobytes() for a in [ids,x,rho,m,c]]
        g=build_auxiliary_faces(ids,x,rho,.02);actual=np.array(list(map(float,exact_action(g,m,c))))
        expected=dense_oracle(x,rho,m,.02)@c
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
        np.testing.assert_allclose(independent_float_action(g,m,c,rho),actual,rtol=2e-13,atol=2e-14)
        self.assertEqual(before,[a.tobytes() for a in [ids,x,rho,m,c]])
        self.assertEqual(len(g.edges),2*len(ids));self.assertTrue(all(v>0 for v in g.conductances))

    def test_uniform_mass_mean_fourier_mode_has_correct_discrete_diffusion_factor(self):
        nx,ny=8,6;rho=np.ones(nx*ny)*1.25;D=.02
        x=np.array([((i+.5)/nx,(j+.5)/ny) for j in range(ny) for i in range(nx)])
        m=rho/(nx*ny);ids=np.arange(nx*ny)
        # The sinc factors make these exact uniform cell averages of the mode.
        wave=np.cos(2*np.pi*2*x[:,0])*np.cos(2*np.pi*x[:,1])*np.sinc(2/nx)*np.sinc(1/ny)
        c=.5+.2*wave;g=build_auxiliary_faces(ids,x,rho,D)
        actual=np.array(list(map(float,exact_action(g,m,c))))
        lam=4*D*(nx*nx*math.sin(2*math.pi/nx)**2+ny*ny*math.sin(math.pi/ny)**2)
        np.testing.assert_allclose(actual,-lam*(c-.5),rtol=2e-13,atol=2e-14)

    def test_exact_mass_energy_constant_and_D_zero_controls(self):
        from fractions import Fraction as F
        ids,x,rho,m,c=fixture();g=build_auxiliary_faces(ids,x,rho,.02);rate=exact_action(g,m,c)
        mm=list(map(lambda v:F(float(v)),m));u=list(map(lambda v:F(float(v)),c))
        self.assertEqual(sum((a*b for a,b in zip(mm,rate)),F(0)),0)
        chi=sum((w*(u[j]-u[i])**2 for (i,j),w in zip(g.edges,g.conductances)),F(0))
        self.assertEqual(-sum((a*b*d for a,b,d in zip(mm,u,rate)),F(0)),chi)
        self.assertGreater(chi,0);self.assertTrue(all(v==0 for v in exact_action(g,m,np.ones(len(m))*.37)))
        zero=build_auxiliary_faces(ids,x,rho,0)
        self.assertTrue(all(v==0 for v in zero.conductances))
        self.assertTrue(all(v==0 for v in exact_action(zero,m,c)))
        np.testing.assert_array_equal(independent_float_action(zero,m,c,rho),np.zeros_like(c))

    def test_identity_permutation_and_periodic_translation_equivariance(self):
        ids,x,rho,m,c=fixture();g=build_auxiliary_faces(ids,x,rho,.02);rate=np.array(list(map(float,exact_action(g,m,c))))
        p=np.array([7,1,12,3,4,19,6,0,8,9,10,11,2,13,14,15,16,17,18,5])
        gp=build_auxiliary_faces(ids[p],x[p],rho[p],.02);rp=np.array(list(map(float,exact_action(gp,m[p],c[p]))))
        np.testing.assert_array_equal(rp,rate[p])
        shifted=(x+np.array([.25,.125]))%1
        gt=build_auxiliary_faces(ids,shifted,rho,.02);rt=np.array(list(map(float,exact_action(gt,m,c))))
        np.testing.assert_allclose(rt,rate,rtol=3e-14,atol=3e-14)

    def test_invalid_and_nontensor_inputs_rejected_without_state_or_geometry_repair(self):
        ids,x,rho,m,c=fixture()
        for D in [-1,float('nan')]:
            with self.assertRaises(ValueError):build_auxiliary_faces(ids,x,rho,D)
        for bad in [rho*0,np.where(np.arange(len(rho))==0,np.nan,rho)]:
            with self.assertRaises(ValueError):build_auxiliary_faces(ids,x,bad,.02)
        bad=x.copy();bad[0,0]+=.001
        with self.assertRaises(ValueError):build_auxiliary_faces(ids,bad,rho,.02)
        with self.assertRaises(ValueError):build_auxiliary_faces(np.zeros_like(ids),x,rho,.02)
        with self.assertRaises(TypeError):build_auxiliary_faces(ids,x,rho,.02,c0=c)
        g=build_auxiliary_faces(ids,x,rho,.02)
        with self.assertRaises(ValueError):exact_action(g,m*0,c)
        with self.assertRaises(ValueError):exact_action(g,m,c*np.nan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuxiliaryFaceChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(not result.wasSuccessful())
