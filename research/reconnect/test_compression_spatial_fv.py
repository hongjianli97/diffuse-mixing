"""Analytic two-grid comparison and a coarse-projection masking counterexample."""
import argparse,json,unittest
import numpy as np
from compression_spatial_fv_core import compare
from voronoi_spatial_projection import project_path,validate_record


def fixture(hidden_error=0.):
    mp=np.full(4,.25);mf=np.full(16,1/16);cp=np.array([.25,.75,.75,.25]);idx=np.array([0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3])
    pids=np.array([8,3,11,6]);ids=np.arange(16);cf=cp[idx];pp=cp-.5;pf=cf-.5
    times=np.array([0.,.025,.05]);refp=np.array([.5+(1-.25*t/.05)*pp for t in times]);reff=refp[:,idx]
    d={'mass':mf,'c0':cf,'ids':ids,'parent_mass':mp,'parent_c0':cp,'parent_ids':pids,
      'parent_index':idx,'parent_id_per_child':pids[idx], 'reference_n1024-k1024_states':reff.copy(),
      'reference_n512-k256_states':reff.copy(),'reference_n1024-k256_states':reff.copy(),'reference_n1024-k512_states':reff.copy()}
    ref={'best_projected_states':refp.copy(),'snapshot_time':times,'space_difference':np.zeros((3,4)),
      'time_fine_difference':np.zeros((3,4)),'n512-k256_projected_states':refp.copy(),
      'n1024-k256_projected_states':refp.copy(),'n1024-k512_projected_states':refp.copy()}
    runs={};paths={};pattern=np.tile([1.,-1.,1.,-1.],(4,1));pattern[1::2]*=-1;pattern=pattern.ravel()
    for N,phi in [(128,pp),(256,pf)]:
        for K in (16,32):
            offset=(1/16 if N==128 else 1/256)+((1/1024 if N==128 else 1/2048) if K==16 else 0.)
            f=np.arange(K+1)/K;states=np.array([.5+(1-.25*t+offset*t)*phi for t in f])
            if N==256:states+=np.outer(f*hidden_error,pattern)
            paths[N,K]={'states':states,'independent_states':states.copy()}
            runs[N,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-16,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(K)]}
    return d,ref,runs,paths


class Checks(unittest.TestCase):
    def test_analytic_common_native_H_and_inputs_unchanged(self):
        d,ref,runs,paths=fixture();all_arrays=list(d.values())+[v for v in ref.values() if isinstance(v,np.ndarray)]+[v for p in paths.values() for v in p.values()]
        before=[a.tobytes() for a in all_arrays]
        for a in all_arrays:a.setflags(write=False)
        r,c=compare(d,ref,runs,paths);row=r['comparison_observations'][-1]
        self.assertAlmostEqual(row['coarse32_error_common'],1/16,places=15);self.assertAlmostEqual(row['fine32_error_common'],1/256,places=15)
        self.assertAlmostEqual(row['native_fine32_error'],1/256,places=15)
        self.assertAlmostEqual(row['coarse32_H_common'],1-(.75+1/16)**2,places=15)
        self.assertAlmostEqual(row['fine32_H_common'],1-(.75+1/256)**2,places=15)
        self.assertEqual(r['comparison_decision'],'compression_spatial_improvement_resolved_with_native_and_time_controls')
        self.assertEqual(before,[a.tobytes() for a in all_arrays]);self.assertIn('reference_projection_difference',c)

    def test_native_error_blocks_masked_common_improvement(self):
        r,c=compare(*fixture(hidden_error=.05));row=r['comparison_observations'][-1]
        self.assertEqual(row['common_decision'],'fine_better_resolvably');self.assertEqual(row['H_decision'],'fine_better_resolvably')
        self.assertEqual(row['native_decision'],'fine_worse_resolvably');self.assertEqual(r['comparison_decision'],'adverse_at_a_checkpoint')
        self.assertGreater(row['native_fine32_error'],row['native_coarse32_error'])

    def test_original_mass_denominator_and_missing_inputs(self):
        d,ref,runs,paths=fixture();d['parent_mass'][0]=np.nextafter(.25,np.inf)
        observed=project_path(np.ones((1,16)),d);self.assertLess(observed[0,0],1.)
        self.assertEqual(d['mass'][0],1/16);self.assertEqual(d['parent_mass'][0],np.nextafter(.25,np.inf))
        bad=dict(d);bad['parent_id_per_child']=d['parent_id_per_child'].copy();bad['parent_id_per_child'][0]=-1
        with self.assertRaises(ValueError):validate_record(bad)
        d,ref,runs,paths=fixture();del paths[256,32]['independent_states']
        with self.assertRaises(KeyError):compare(d,ref,runs,paths)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
