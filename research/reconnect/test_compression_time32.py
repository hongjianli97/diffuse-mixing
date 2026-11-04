"""Independent two-cell analytic norm/H and no missing-control fallback checks."""
import argparse,json,math,unittest
import numpy as np
from compression_time32_study import compare


def fixture(kernel4_offset=1/8):
    m=np.array([1.,3.]);c0=np.array([.25,.75]);mean=.625;phi=c0-mean
    times=np.array([0.,.025,.05]);reference=np.array([mean+(1-.25*t/.05)*phi for t in times]);reference[0]+=2**-20*phi
    ref={'mass':m,'c0':c0,'snapshot_time':times,'best_projected_states':reference,
      'space_difference':np.array([np.zeros(2),2**-18*phi,2**-18*phi]),
      'time_fine_difference':np.array([np.zeros(2),2**-19*phi,2**-19*phi])}
    ref['n1024-k256_projected_states']=reference+np.array([np.zeros(2),2**-18*phi,2**-18*phi])
    ref['n512-k256_projected_states']=ref['n1024-k256_projected_states']+ref['space_difference']
    ref['n1024-k512_projected_states']=reference+ref['time_fine_difference']
    paths={};runs={}
    for K,offset in [(4,1/16),(8,1/64),(16,1/256),(32,1/1024)]:
        fraction=np.arange(K+1)/K;states=np.array([mean+(1-.25*t+offset*t)*phi for t in fraction])
        delta={4:2**-43,8:2**-42,16:2**-39,32:2**-40}[K]
        controls=states-np.outer(fraction*delta,phi)
        paths[K]={'states':states,'independent_states':controls}
        runs[K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-12,'cumulative_tail_indicator_absolute':(i+1)*1e-16} for i in range(K)]}
    old={};result={'cases':{}}
    for K,offset in [(4,kernel4_offset),(8,1/16)]:
        states=np.array([mean+(1-.25*t+offset*t)*phi for t in np.arange(K+1)/K]);old[f'k{K}_states']=states
        for form in ('direct_endpoints','two_half_endpoints'):old[f'k{K}_{form}']=states[1:].copy()
        result['cases'][str(K)]={'steps':[{'direct_vs_sampled_relative_difference':0.,'two_half_vs_sampled_relative_difference':0.,
          'sparse_graph_audit':{'graph_algebra_eligible':True,'linear_algebra_tolerance':1e-10,'clipping_or_renormalization':False},
          'operator_diagnostics':{'clipping_fitting_or_normalization':False}} for _ in range(K)]}
    return ref,old,result,runs,paths


class Checks(unittest.TestCase):
    def test_exact_norm_H_time_index_and_unchanged_inputs(self):
        ref,old,result,runs,paths=fixture();all_arrays=[v for v in ref.values() if isinstance(v,np.ndarray)]+list(old.values())+[v for p in paths.values() for v in p.values()]
        before=[a.tobytes() for a in all_arrays]
        for a in all_arrays:a.setflags(write=False)
        r,v=compare(ref,old,result,runs,paths);row=r['comparison_observations'][-1];time=r['time_controls'][-1]
        sigma=math.sqrt(3/16)
        self.assertAlmostEqual(row['FV32_error'],1/1024,places=15)
        self.assertAlmostEqual(row['saved_kernel8_error'],1/16,places=15)
        expected=3/1024+1/16+2*(2**-18+2**-19)+2**-20+48*(1e-12+1e-16)/sigma+2**-39+2**-40
        self.assertAlmostEqual(row['state_indicator'],expected,places=15)
        self.assertAlmostEqual(row['FV32_H'],1-(.75+1/1024)**2,places=15)
        self.assertAlmostEqual(row['reference_H'],1-.75**2,places=15)
        self.assertAlmostEqual(time['FV16_32_state_difference'],3/1024,places=15)
        self.assertAlmostEqual(time['FV8_16_state_difference'],3/256,places=15)
        self.assertAlmostEqual(time['state_new_difference_ratio'],4,places=13)
        self.assertTrue(r['both_state_and_H_time_differences_decreased'])
        self.assertEqual(r['comparison_decision'],'compression32_comparison_inconclusive')
        self.assertEqual(before,[a.tobytes() for a in all_arrays]);self.assertIn('FV16_32_difference',v)
        positive,_=compare(*fixture(kernel4_offset=5/64))
        self.assertEqual(positive['comparison_decision'],'compression32_time_control_decreased_and_improvement_resolved')

    def test_missing_new_or_previous_controls_rejected(self):
        for which in ('new_path','previous_run','old_local'):
            ref,old,result,runs,paths=fixture()
            if which=='new_path':del paths[32]['independent_states']
            elif which=='previous_run':del runs[16]['steps'][7]['cumulative_tail_indicator_absolute']
            else:del old['k4_two_half_endpoints']
            with self.assertRaises(KeyError):compare(ref,old,result,runs,paths)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
