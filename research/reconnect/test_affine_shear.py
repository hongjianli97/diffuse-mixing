"""Four small fixtures: boundary, independent world geometry, references and adverse controls."""
from fractions import Fraction as F
import argparse,copy,json,math,unittest
import numpy as np
from sliding_periodic_voronoi import SlidingPeriodicDomain,build_sliding_faces
from audit_sliding_world_geometry import audit_sliding_world_geometry
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from affine_shear_core import COUNTS,D,pure_inputs,directional_certificate,physical_clocks,reference_bundle,frozen_coefficients,evaluate
from t1_uniform_core import pure_labels


def geom(N,s):
    labels=pure_labels(N);ids=np.arange(N*N);b=SlidingPeriodicDomain(s)
    positions=[list(map(str,b.push(tuple(F(float(x)) for x in p)))) for p in labels]
    frame={'boundary':b.packet(),'shear_exact':str(F(s)),'world_positions_exact':positions,'rho':[1.]*len(ids),'D_float':.02}
    return ids,b,frame


def fixture():
    N=12;a=pure_labels(N);m=np.full(N*N,1/(N*N));c0=.5+.2*(math.sin(math.pi/N)/(math.pi/N))**2*np.sin(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1])
    d={'mass':m,'c0':c0};world=pure_inputs(a,np.arange(N*N));b=reference_bundle(d,a,world,N);modes=b['modes'];u=math.sin(math.pi/N)**2;v=math.sin(2*math.pi/N)**2
    runs={};paths={};res={}
    for K in COUNTS:
        states=[c0.copy()]
        for i in range(1,K+1):
            U,V=frozen_coefficients(world,K,i)
            states.append(.5+modes[0]*math.exp(-4*N*N*u*float(U))+modes[1]*math.exp(-4*N*N*(u*float(U)+v*float(V))))
        states=np.array(states);paths[K]={'states':states,'independent_states':states.copy()};res[K]=[0.]*K
        runs[K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':0.} for i in range(K)]}
    return d,a,world,runs,paths,res


class Checks(unittest.TestCase):
    def test_boundary_original_s0_dispatch_and_rejections(self):
        ids,b,frame=geom(4,F(0));packet,world,cert=build_sliding_faces(ids,frame['world_positions_exact'],frame['rho'],.02,b)
        old=coefficient_packet(build_periodic_faces(ids,pure_labels(4),np.ones(16),.02))
        self.assertEqual(packet,old);self.assertTrue(cert['s0_original_rectangle_dispatched'])
        ids,b,frame=geom(4,F(1,8));packet,world,cert=build_sliding_faces(ids,frame['world_positions_exact'],frame['rho'],.02,b)
        self.assertTrue(directional_certificate(packet,frame,4)['all_geometry_controls_passed']);self.assertFalse(cert['s0_original_rectangle_dispatched'])
        for bad in (F(-1,8),F(1,3),True):
            with self.assertRaises(ValueError):SlidingPeriodicDomain(bad)
        duplicate=copy.deepcopy(frame['world_positions_exact']);duplicate[1]=duplicate[0]
        with self.assertRaises(ValueError):build_sliding_faces(ids,duplicate,frame['rho'],.02,b)
        outside=copy.deepcopy(frame['world_positions_exact']);outside[0]=['2','0']
        with self.assertRaises(ValueError):build_sliding_faces(ids,outside,frame['rho'],.02,b)
        with self.assertRaises(ValueError):build_sliding_faces(ids,frame['world_positions_exact'],[0.]*16,.02,b)

    def test_independent_world_faces_moments_and_corruption_rejection(self):
        ids,b,frame=geom(4,F(3,16));packet,world,cert=build_sliding_faces(ids,frame['world_positions_exact'],frame['rho'],.02,b)
        edges,g,degree,result=audit_sliding_world_geometry(world,frame,ids,frame['rho'],.02,[0,7,15])
        self.assertEqual(len(g),48);self.assertTrue(result['all_world_vertices_rebuilt_directly_from_world_bisectors'])
        self.assertTrue(directional_certificate(packet,frame,4)['all_cell_second_moment_tensors_exact'])
        bad=copy.deepcopy(world);bad['vertices_world_exact'][0][0]=str(F(bad['vertices_world_exact'][0][0])+F(1,2**20))
        with self.assertRaises(AssertionError):audit_sliding_world_geometry(bad,frame,ids,frame['rho'],.02,[0])
        bad=copy.deepcopy(world);bad['conductances_exact'][0]=str(F(bad['conductances_exact'][0])*F(100001,100000))
        with self.assertRaises(AssertionError):audit_sliding_world_geometry(bad,frame,ids,frame['rho'],.02,[0])
        bad=copy.deepcopy(world);bad['basis_columns_exact'][1][0]='0'
        with self.assertRaises(AssertionError):audit_sliding_world_geometry(bad,frame,ids,frame['rho'],.02,[0])

    def test_independent_physical_clock_quadrature_cell_means_and_bridge(self):
        z,w=np.polynomial.legendre.leggauss(12)
        for t in (.025,.05):
            tt=(z+1)*t/2;s=5*tt
            q=[float(w@(.02*(1+(eta-s)**2))*t/2) for eta in (1,-1)]
            for a,b in zip(map(float,physical_clocks(F(t))),q):self.assertAlmostEqual(a,b,places=16)
        N=4;labels=pure_labels(N);mass=np.full(N*N,1/(N*N));cp=[]
        for x,y in labels:
            xx=x+z[:,None]/(2*N);yy=y+z[None,:]/(2*N)
            cp.append(float(np.sum((.5+.2*np.sin(2*math.pi*xx)*np.cos(2*math.pi*yy))*w[:,None]*w[None,:])/4))
        cp=np.array(cp);world=pure_inputs(labels,np.arange(N*N));bundle=reference_bundle({'mass':mass,'c0':cp},labels,world,N)
        np.testing.assert_allclose(bundle['reference'][0],cp,atol=2e-16,rtol=0)
        changed=cp.copy();changed[0]=math.nextafter(changed[0],math.inf);before=changed.tobytes()
        bundle=reference_bundle({'mass':mass,'c0':changed},labels,world,N)
        self.assertGreater(float(np.linalg.norm(bundle['reference'][0]-changed)),0);self.assertEqual(changed.tobytes(),before)

    def test_raw_time_direction_observables_and_bad_full_sequence_rejection(self):
        args=fixture();data,labels,world,runs,paths,res=args
        arrays=list(data.values())+[a for p in paths.values() for a in p.values()];before=[a.tobytes() for a in arrays]
        for a in arrays:a.setflags(write=False)
        result,b=evaluate(*args,N=12)
        self.assertTrue(result['all_frozen_symbol_and_sequence_controls_passed']);self.assertTrue(result['all_reference_decompositions_passed'])
        self.assertTrue(result['time_decreases_resolved'])
        self.assertGreater(result['direction_observables'][-1]['actual_direction_contrast'],0)
        for row in result['direction_observables']:
            self.assertFalse(row['state_or_observables_rescaled_or_projected']);self.assertFalse(row['signed_error_cancellation_used_for_acceptance'])
        self.assertEqual(before,[a.tobytes() for a in arrays])
        bad={K:{key:value.copy() for key,value in p.items()} for K,p in paths.items()};bad[16]['independent_states'][8,0]+=.01
        rejected,_=evaluate(data,labels,world,runs,bad,res,N=12)
        self.assertFalse(rejected['all_frozen_symbol_and_sequence_controls_passed']);self.assertFalse(rejected['registered_affine_shear_numerical_control_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
