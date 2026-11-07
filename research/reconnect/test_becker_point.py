"""Four pre-scientific boundary, duration, material and adverse-control fixtures."""
from fractions import Fraction as F
import argparse,copy,json,unittest
import numpy as np
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from voronoi_time_action import operators
from voronoi_time16_core import ordered_path
from explicit_duration_action import ordered_path_duration
from becker_point_core import extrude,observe_columns,evaluate


class Checks(unittest.TestCase):
    def test_bounded_periodic_world_faces_and_rejection(self):
        x=np.linspace(-.875,.875,8);y=(np.arange(8)+.5)/8;p=np.stack(np.meshgrid(x,y,indexing='xy'),axis=-1).reshape(-1,2)
        ids=np.arange(64);rho=np.where(p[:,0]<0,1.,4.);packet=build_bounded_faces(ids,p,rho,.02,(-1,1))
        edges,g,deg,cert=audit_geometry(packet,p,rho,ids,.02,(-1,1))
        self.assertEqual(len(edges),120);self.assertEqual(cert['left_external_faces'],8);self.assertEqual(cert['right_external_faces'],8)
        self.assertTrue(cert['all_cell_world_halfplanes_verified']);self.assertFalse(cert['x_periodic'])
        self.assertFalse(any(abs(i%8-j%8)==7 and i//8==j//8 for i,j in edges))
        bad=copy.deepcopy(packet);bad['conductances_exact'][0]=str(2*F(bad['conductances_exact'][0]))
        with self.assertRaises(AssertionError):audit_geometry(bad,p,rho,ids,.02,(-1,1))
        duplicate=p.copy();duplicate[0]=duplicate[1]
        with self.assertRaises(ValueError):build_bounded_faces(ids,duplicate,rho,.02,(-1,1))
        with self.assertRaises(ValueError):build_bounded_faces(ids,p,rho,.02,(0,1))

    def test_duration_old_path_bitwise_equivalence_and_explicit_time(self):
        m=np.array([.1,.2,.3,.4]);c0=np.array([.2,.4,.7,.5]);edges=np.array([[0,1],[1,2],[2,3],[0,3]])
        a=operators(m,edges,[F(1,50)]*4);b=operators(m,edges,[F(1,20),F(1,90),F(1,40),F(1,60)])
        load=lambda i:a if i%2==0 else b;np.random.seed(20261101);old,po=ordered_path(m,c0,load,4)
        np.random.seed(20261101);new,pn=ordered_path_duration(m,c0,load,4,T=F(1,20));self.assertEqual(old,new)
        for key in po:np.testing.assert_array_equal(po[key],pn[key])
        np.random.seed(20261101);long,pl=ordered_path_duration(m,c0,load,4,T=F(1,10))
        self.assertEqual(long['steps'][0]['dt_exact'],'1/40');self.assertGreater(np.linalg.norm(pl['states'][-1]-po['states'][-1]),1e-4)
        zero=operators(m,edges,[F(0)]*4);rz,pz=ordered_path_duration(m,c0,lambda i:zero,4,T=F(1,10))
        np.testing.assert_array_equal(pz['states'],np.tile(c0,(5,1)))
        with self.assertRaises(ValueError):ordered_path_duration(m,c0,load,4,T=F(0))

    def test_true_column_mass_means_and_transverse_component_integrals(self):
        mc=np.arange(1,33,dtype=float)/128;cc=.5+.2*np.cos((np.arange(32)+.5)*np.pi/32);ref=.5+.9*(cc-.5)
        col={'mass':mc,'c0':cc,'reference':ref};before=[v.tobytes() for v in col.values()]
        for v in col.values():v.setflags(write=False)
        d=extrude(col);np.testing.assert_array_equal(d['mass'].reshape(8,32).sum(axis=0),mc)
        np.testing.assert_array_equal(d['c0'].reshape(8,32),np.tile(cc,(8,1)))
        u=d['c0']+np.arange(256)/4096;observed=observe_columns(u,mc)
        for i in range(32):
            exact=sum((F(float(d['mass'][j*32+i]))*F(float(u[j*32+i])) for j in range(8)),F(0))/F(float(mc[i]))
            self.assertLessEqual(abs(observed[i]-float(exact)),3*np.finfo(float).eps)
        self.assertEqual(before,[v.tobytes() for v in col.values()])

    def test_original_positive_time_H_budget_and_bad_full_sequence_rejection(self):
        mc=np.full(32,1/32);cc=.5+.15*np.cos((np.arange(32)+.5)*np.pi/32);col={'mass':mc,'c0':cc,'reference':.5+(cc-.5)*np.exp(-.1)}
        d=extrude(col);runs={};paths={}
        for K in (4,8,16):
            state=np.array([.5+(d['c0']-.5)*np.exp(-(1+1/K**2)*i/(10*K)) for i in range(K+1)])
            paths[K]={'states':state,'independent_states':state.copy()}
            runs[K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':0.,'primary_control_mass_norm_difference':0.,'minimum':float(c.min()),'maximum':float(c.max())} for i,c in enumerate(state[1:])]}
        r=evaluate(d,col,runs,paths,{'reference_space_time_indicator_absolute':1e-12})
        self.assertTrue(r['registered_moving_Becker_oracle_discriminator_passed']);self.assertTrue(r['time_control']['H_time_decrease_resolved'])
        self.assertEqual(r['time_control']['fixed_state_decrease_indicator_relative'],sum(r['time_control']['components_absolute'].values())/r['original_column_initial_fluctuation_norm'])
        bad={K:{key:v.copy() for key,v in p.items()} for K,p in paths.items()};bad[16]['independent_states'][-1,0]+=.01
        rejected=evaluate(d,col,runs,bad,{'reference_space_time_indicator_absolute':1e-12});self.assertFalse(rejected['registered_moving_Becker_oracle_discriminator_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
