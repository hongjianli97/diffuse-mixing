"""Exactly two frozen software groups; each at most16nodes/eight frames."""
import argparse,time,unittest,signal
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from compression_density_sources import periodic_winding,anchor_rings,ring_areas,ledger_density,kernel_density
from compression_density_audit_core import bin_density,verify_ring_density
from compression_density_compare import reference_parts,classification
from t1_density_sources import signed_ring_area,reweight


class Controls(unittest.TestCase):
    def test_01_lifts_fixed_rings(self):
        ids=np.arange(2);t=np.array([0.,.1,.2]);x=np.array([[[.9,.2],[.1,.8]],[[.05,.2],[.95,.8]],[[.2,.2],[.8,.8]]])
        w=periodic_winding(ids,t,x,t)
        np.testing.assert_array_equal(w[:,0,0],[0,1,1]);np.testing.assert_array_equal(w[:,1,0],[0,-1,-1])
        np.testing.assert_array_equal(periodic_winding(ids,t[::2],x[::2],t[::2]),w[::2])
        for a,b,c,d in ((ids,t[::-1],x[::-1],t[::-1]),(ids,t[::2],x[::2],t),(np.array([0,0]),t,x,t)):
            with self.assertRaises(ValueError):periodic_winding(a,b,c,d)
        half=x.copy();half[1,0,0]=.4
        with self.assertRaises(ValueError):periodic_winding(ids,t,half,t)
        labels=np.array([[(j+.5)/4,(i+.5)/4] for i in range(4) for j in range(4)])
        edges=sorted({tuple(sorted((i,j))) for i in range(16) for j in ((i//4)*4+(i%4+1)%4,((i//4+1)%4)*4+i%4)})
        rings,a0=anchor_rings(np.arange(16),labels,edges)
        self.assertTrue(all(a==F(1,8) for a in a0))
        # Three frames; fixed rings across a positive wrap, plus nonlinear distortion.
        frames=[]
        for s in (0.,.1,.2):
            y=labels.copy();y[:,0]+=.15*s/(2*np.pi)*np.sin(2*np.pi*labels[:,0])+s;y[:,1]-=.1*s
            frames.append(y%1)
        frames=np.array(frames);ww=periodic_winding(np.arange(16),t,frames,t)
        r,area=ledger_density(frames[-1],ww[-1],np.ones(16),rings,a0)
        verify_ring_density(frames[-1],ww[-1],np.ones(16),rings,a0,r,area)
        expected=1/(1+.03*np.sin(np.pi/2)/(np.pi/2)*np.cos(2*np.pi*labels[:,0]))
        np.testing.assert_allclose(r,expected,rtol=1e-14,atol=1e-14)
        self.assertTrue(np.any(ww[-1,:,0]))
        with self.assertRaises(ValueError):signed_ring_area([[0,0],[0,1],[1,1],[1,0]])
        with self.assertRaises(ValueError):signed_ring_area([[0,0],[1,1],[0,1],[1,0]])
        reverse=[list(reversed(ring)) for ring in rings]
        with self.assertRaises(ValueError):ring_areas(labels,np.zeros_like(labels,dtype=int),reverse)

    def test_02_kernel_rates_budgets(self):
        ids=np.arange(16);x=np.array([[(j+.5)/4,(i+.5)/4] for i in range(4) for j in range(4)])
        x[:4]=[[0,.125],[.125,.125],[np.nextafter(.125,0),.125],[np.nextafter(.125,1),.125]]
        x[4]=[.99,.125];m=np.arange(1.,17.)/100;h=1/16
        rho,record=kernel_density(ids,x,ids,m,h);expected,counts=bin_density(ids,x,m,h)
        np.testing.assert_allclose(rho,expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(kernel_density(ids,x,ids,2*m,h)[0],2*rho)
        self.assertFalse(record['normalization']);self.assertTrue(record['self_retained']);self.assertGreater(rho.max(),1)
        self.assertEqual(record['support_count_min'],min(counts));self.assertEqual(record['support_count_max'],max(counts))
        self.assertEqual(float(np.linalg.norm(x[1]-x[0])),2*h)
        self.assertLess(float(np.linalg.norm(x[2]-x[0])),2*h);self.assertGreater(float(np.linalg.norm(x[3]-x[0])),2*h)
        for bad in (np.zeros(16),-m):
            with self.assertRaises(ValueError):kernel_density(ids,x,ids,bad,h)
        with self.assertRaises(ValueError):kernel_density(ids,x,ids[::-1],m,h)
        with self.assertRaises(ValueError):kernel_density(ids,x,ids,m,.25)
        packet={'ids':ids.tolist(),'D_exact':'1/50','thickness_exact':'1','edges':[[0,1]],'face_over_distance_exact':['3/7'],
                'conductances_exact':['0'],'unchanged_geometry':['saved',3]}
        new=reweight(packet,ids,rho,m);expected_g=F(3,7)/(1/(2*F(float(rho[0]))/50)+1/(2*F(float(rho[1]))/50))
        self.assertEqual(F(new['conductances_exact'][0]),expected_g)
        self.assertEqual({k:v for k,v in new.items() if k!='conductances_exact'},{k:v for k,v in packet.items() if k!='conductances_exact'})
        self.assertEqual(reweight(dict(packet,D_exact='0'),ids,rho,m)['conductances_exact'],['0'])
        with self.assertRaises(ValueError):reweight(dict(packet,D_exact='0'),ids,-rho,m)
        c=np.array([.2,.8]);d={'space':np.array([[0.,0.],[.01,-.01]]),'time':np.array([[0.,0.],[.02,-.02]]),
            'reference':np.array([[.21,.79],[.3,.7]]),'reference_space_coarse':np.array([[.2,.8],[.28,.72]]),
            'reference_space_fine':np.array([[.2,.8],[.29,.71]]),'reference_time_check':np.array([[.2,.8],[.295,.705]])}
        norm=lambda v:float(np.sqrt(np.sum(v*v)/2));H=lambda v:1-float(np.sum((v-v.mean())**2)/np.sum((c-c.mean())**2))
        cp,hp=reference_parts(d,1,norm,H,c)
        self.assertEqual(cp['four_reference_space'],4*norm(d['space'][1]));self.assertEqual(cp['four_reference_time'],4*norm(d['time'][1]))
        self.assertGreater(cp['two_initial_bridge'],0);self.assertGreater(hp['two_initial_bridge'],0)
        self.assertEqual(hp['two_reference_space_same_basis'],2*abs(H(d['reference_space_coarse'][1])-H(d['reference_space_fine'][1])))
        self.assertEqual(hp['two_reference_time_same_basis'],2*abs(H(d['reference_time_check'][1])-H(d['reference'][1])))
        self.assertTrue(all(v>=0 for v in (*cp.values(),*hp.values())))
        self.assertEqual(classification(2.,1.),'error_larger_resolvably');self.assertEqual(classification(-2.,1.),'error_smaller_resolvably')
        self.assertEqual(classification(.5,1.),'unresolved');self.assertEqual(classification(1.,1.),'unresolved')


def main(directory):
    out=Path(directory);lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    def timeout(*_):raise TimeoutError('frozen120second software cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);start=time.monotonic()
    try:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
        save(out/'tests.json',{'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'elapsed_seconds':time.monotonic()-start,'max_nodes_per_group':16,'max_frames_per_group':3,
            'recorded_checks_total':506 if result.wasSuccessful() else 504,'old504_checks_rerun':False})
        if not result.wasSuccessful():raise SystemExit(1)
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
