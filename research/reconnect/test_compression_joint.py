"""Two bounded software groups, locked before any generated test arrays."""
from pathlib import Path
import argparse,copy,signal,tempfile,time,unittest
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from compression_joint_core import nested_samples,save_npz,compare
from compression_voronoi_core import sample_points
from compression_density_sources import periodic_winding
from compression_joint_reference import recorded_reference_path
from compression_reference import reference_path,material_columns,project_to_saved_cells
from material_union import observe_path,group_sum
from compression_density_compare import reference_parts,classification


class Controls(unittest.TestCase):
    def test_01_nested_clock_lifts_immutability(self):
        ids=np.arange(16);axis=(np.arange(4)+.5)/4
        labels=np.stack(np.meshgrid(axis,axis),axis=-1).reshape(-1,2);rho=1+.25*np.cos(2*np.pi*labels[:,0])
        old=sample_points(labels,ids,rho,np.linspace(0,.05,3));before={k:v.copy() for k,v in old.items()}
        new=nested_samples(ids,labels,rho,old['time'],old['positions'])
        np.testing.assert_array_equal(new['time'][::2],old['time'])
        np.testing.assert_array_equal(new['positions'][::2],old['positions'])
        np.testing.assert_array_equal(new['positions'],sample_points(labels,ids,rho,new['time'])['positions'])
        np.testing.assert_array_equal(new['winding'][::2],periodic_winding(ids,old['time'],old['positions'],old['time']))
        for k in before:np.testing.assert_array_equal(old[k],before[k])
        # Explicit positive/negative wraps and half-period ambiguity, five frames.
        t=np.linspace(0,.05,5);x=np.array([[[v%1,.2],[(1-v)%1,.8]] for v in (.9,.95,1.,1.05,1.1)])
        w=periodic_winding(np.arange(2),t,x,t)
        np.testing.assert_array_equal(w[:,0,0],[0,0,1,1,1]);np.testing.assert_array_equal(w[:,1,0],[0,0,0,-1,-1])
        half=x.copy();half[1,0,0]=.4
        with self.assertRaises(ValueError):periodic_winding(np.arange(2),t,half,t)
        with self.assertRaises(ValueError):nested_samples(ids,labels,rho,old['time'][:-1],old['positions'][:-1])
        with self.assertRaises(ValueError):nested_samples(ids,labels,rho,old['time'],old['positions'][:-1])
        with tempfile.TemporaryDirectory() as q:
            p=Path(q)/'old.npz';save_npz(p,c0=np.arange(4));h=digest(p)
            with self.assertRaises(FileExistsError):save_npz(p,c0=np.zeros(4))
            self.assertEqual(h,digest(p))

    def test_02_reference_observation_budgets(self):
        # 8 material columns, 16 observed cells, 5 stored time frames.
        _,m4,_=material_columns(4,.25);pm=np.tile(m4/4,4)
        old,a=reference_path(8,4,4,pm);new,b=recorded_reference_path(8,4,4,pm)
        self.assertEqual(a,b)
        for key in old:np.testing.assert_array_equal(old[key],new[key])
        np.testing.assert_array_equal(new['B_history'][::2],new['B_snapshots'])
        m=group_sum(pm,2,2);coarse=observe_path(new['projected_states'],pm,m,side=2)
        for k,t in enumerate(new['snapshot_time']):
            parent,_=project_to_saved_cells(new['B_snapshots'][k],new['mass_1d'],t,.02,4,pm)
            expected=np.array([sum(pm[j]*parent[j] for j in js)/m[i] for i,js in enumerate(
                ([0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]))])
            np.testing.assert_allclose(coarse[k],expected,rtol=0,atol=2e-16)
        c=np.array([.2,.4,.6,.8]);base=np.array([c,.5+.9*(c-.5),.5+.8*(c-.5)])
        d={'mass':np.ones(4)/4,'c0':c,'reference':base.copy(),'reference_space_coarse':base+.002,
           'reference_space_fine':base+.001,'reference_time_check':base+.001,'space':np.full_like(base,.001),
           'time':np.full_like(base,.001)}
        d['reference'][0,0]+=1e-4
        states={(s,K):base.copy() for s in ('oracle','ledger') for K in (16,32,64)}
        independent={key:v.copy() for key,v in states.items()};mt={key:[0.,1e-6,2e-6] for key in states}
        before=copy.deepcopy(d);r=compare(d,states,independent,mt)
        for key in d:np.testing.assert_array_equal(d[key],before[key])
        for row in r['source_comparisons']:
            cp,hp=row['concentration_parts_absolute'],row['H_parts']
            self.assertAlmostEqual(cp['four_reference_space'],.004)
            self.assertAlmostEqual(cp['four_reference_time'],.004)
            self.assertGreater(cp['two_initial_bridge'],0)
            self.assertGreater(hp['two_initial_bridge'],0)
            self.assertEqual(row['concentration_decision'],'unresolved');self.assertEqual(row['H_decision'],'unresolved')
        for row in r['time_differences']:
            k=round(row['time']/.025)
            self.assertEqual(row['concentration_decrease_indicator'],4*k*1e-6)
            self.assertFalse(row['concentration_decrease_resolved']);self.assertFalse(row['H_decrease_resolved'])
        self.assertTrue(r['stop_expansion']);self.assertFalse(r['old_rows_rejudged'])
        self.assertEqual(classification(1.,1.),'unresolved');self.assertEqual(classification(-1.,1.),'unresolved')


def main(directory):
    out=Path(directory);start=time.monotonic()
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    def alarm(*_):raise TimeoutError('frozen120second software cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
        save(out/'tests.json',{'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),
            'elapsed_seconds':time.monotonic()-start,'maximum_nodes':16,'maximum_frames':5,
            'recorded_checks_total':508 if r.wasSuccessful() else 506,'old_checks_rerun':0})
        if not r.wasSuccessful():raise SystemExit(1)
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
