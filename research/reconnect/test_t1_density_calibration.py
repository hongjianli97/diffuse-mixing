"""Two frozen software groups, at most four nodes and three frames per fixture."""
from pathlib import Path
from fractions import Fraction as F
import argparse,copy,json,math,signal,time,unittest
import numpy as np
from audit_route_decision import ROOT,read,digest
from t1_density_sources import kernel_density,density_from_ring,reweight


class Checks(unittest.TestCase):
    def test_density_semantics_and_rejections(self):
        ids=np.array([20,10,30]);x=np.array([[.01,.25],[.99,.25],[.5,.75]]);m=np.array([1.,2.,4.]);h=1/16
        before=[v.tobytes() for v in (ids,x,m)]
        for v in (ids,x,m):v.setflags(write=False)
        got=kernel_density(ids,x,ids,m,h,[1,1])
        def w(r):
            q=r/h
            return 0. if q>=2 else 9/(4*math.pi*h*h)*(1-q/2)**6*(1+3*q+35*q*q/12)
        expected=np.array([w(0)+2*w(abs((x[1,0]-x[0,0])-1)),2*w(0)+w(abs((x[1,0]-x[0,0])-1)),4*w(0)])
        np.testing.assert_allclose(got,expected,rtol=64*np.finfo(float).eps,atol=0)
        np.testing.assert_allclose(kernel_density(ids,x,ids,3*m,h,[1,1]),3*got,rtol=64*np.finfo(float).eps,atol=0)
        p=[2,0,1];np.testing.assert_array_equal(kernel_density(ids[p],x[p],ids[p],m[p],h,[1,1]),got[p])
        for bad in ([0,2,4],[-1,2,4],[math.nan,2,4],[math.inf,2,4]):
            with self.assertRaises(ValueError):kernel_density(ids,x,ids,bad,h,[1,1])
        for bad_h in (0,-1,math.nan,.25):
            with self.assertRaises(ValueError):kernel_density(ids,x,ids,m,bad_h,[1,1])
        for bad_x in (np.array([[.01,.25],[.01,.25],[.5,.75]]),x+1,np.full((3,2),math.nan)):
            with self.assertRaises(ValueError):kernel_density(ids,bad_x,ids,m,h,[1,1])
        with self.assertRaises(ValueError):kernel_density(ids,x,ids[::-1],m,h,[1,1])
        with self.assertRaises(ValueError):kernel_density([1,1,2],x,[1,1,2],m,h,[1,1])
        v=np.array([[.125,0],[0,.125],[-.125,0],[0,-.125]])
        rho,a0,a=density_from_ring(2.,v,v);self.assertEqual((rho,a0,a),(2.,F(1,32),F(1,32)))
        self.assertEqual(density_from_ring(2.,v,v+np.array([.25,.5]))[0],2.)
        affine=v@np.diag([2.,3.]);self.assertEqual(density_from_ring(2.,v,affine)[0],float(F(1,3)))
        self.assertEqual(a0/F(1,64),2) # proxy area cannot replace a material mass
        for invalid in (v[::-1],v[[0,2,1,3]],v*0,np.full((4,2),math.nan)):
            with self.assertRaises(ValueError):density_from_ring(1.,v,invalid)
        for bad_rho in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError):density_from_ring(bad_rho,v,v)
        self.assertEqual(before,[v.tobytes() for v in (ids,x,m)])

    def test_same_backend_interface_and_rejections(self):
        p={'ids':[0,1],'edges':[[0,1]],'D_exact':str(F(.02)),'thickness_exact':'1',
           'face_over_distance_exact':['3/2'],'conductances_exact':[str(F(3,2)*F(.02))],
           'image_shifts':[[1,0]],'geometry_marker':{'untouched':True}}
        original=copy.deepcopy(p);m=np.array([.3,.7]);m.setflags(write=False)
        self.assertEqual(reweight(p,[0,1],[1,1],m),p)
        r=reweight(p,[0,1],[2,3],m)
        self.assertEqual(F(r['conductances_exact'][0]),F(3,2)/(1/(2*F(2)*F(.02))+1/(2*F(3)*F(.02))))
        self.assertEqual({k:v for k,v in r.items() if k!='conductances_exact'}, {k:v for k,v in p.items() if k!='conductances_exact'})
        zero=copy.deepcopy(p);zero['D_exact']='0';self.assertEqual(reweight(zero,[0,1],[2,3],m)['conductances_exact'],['0'])
        for bad in ([0,1],[-1,1],[math.nan,1],[math.inf,1],[1]):
            with self.assertRaises(ValueError):reweight(p,[0,1],bad,m)
            with self.assertRaises(ValueError):reweight(zero,[0,1],bad,m)
        with self.assertRaises(ValueError):reweight(p,[1,0],[1,1],m)
        with self.assertRaises(ValueError):reweight(p,[0,1],[1,1],[0,.7])
        self.assertEqual(p,original);np.testing.assert_array_equal(m,[.3,.7])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();out=Path(a.directory)
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    def timeout(*_):raise TimeoutError('frozen120second software cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);started=time.monotonic()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with (out/'tests.json').open('x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'elapsed_seconds':time.monotonic()-started,'maximum_fixture_nodes':4,'maximum_fixture_frames':3},f,indent=2);f.write('\n')
    signal.alarm(0);raise SystemExit(0 if r.wasSuccessful() else 1)
