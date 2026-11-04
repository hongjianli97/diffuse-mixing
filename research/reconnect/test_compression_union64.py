"""Analytic three-grid checkpoint, H reference base, and hidden-native-error checks."""
import argparse,json,unittest
import numpy as np
from compression_union64_core import compare
from material_union import block_index


def fixture(hidden=0.):
    cp=np.tile([.25,.75],2048);phi=cp-.5;data={};runs={};paths={}
    for N in (64,128,256):
        idx=block_index(64,N//64);c0=cp[idx];m=np.full(N*N,1/(N*N));p=c0-.5
        refs=np.array([.5+(1-.25*t)*p for t in (0.,.5,1.)])
        data[N]={'mass':m,'c0':c0,'reference':refs,'space':np.zeros_like(refs),'time':np.zeros_like(refs)}
        for K in (16,32):
            offset={64:1/16,128:1/64,256:1/256}[N]+(1/4096 if K==16 else 0.)
            states=np.array([.5+(1-.25*t+offset*t)*p for t in np.arange(K+1)/K])
            if N==256:
                # Four-column oscillation has zero raw component sum in every common block.
                states+=np.outer(np.arange(K+1)/K*hidden,np.tile([1.,-1.,1.,-1.],N*N//4))
            paths[N,K]={'states':states,'independent_states':states.copy()}
            runs[N,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-16,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(K)]}
    r=data[64]['reference'];p=phi[None,:]
    # Distinct reference bases: an incorrect reconstructed H difference is detectable.
    data[64].update(reference_space_coarse=r+.0003*p,reference_space_fine=r+.0001*p,reference_time_check=r+.00002*p)
    data[64]['space']=.0002*np.broadcast_to(p,r.shape)
    data[64]['time']=.00002*np.broadcast_to(p,r.shape)
    return data,runs,paths


class Checks(unittest.TestCase):
    def test_analytic_three_level_time_indices_reference_H_and_readonly(self):
        data,runs,paths=fixture();values=[v for d in data.values() for v in d.values()]+[v for d in paths.values() for v in d.values()]
        before=[v.tobytes() for v in values]
        for v in values:v.setflags(write=False)
        result,common=compare(data,runs,paths)
        self.assertTrue(result['all_adjacent_times_common_native_H_resolved'])
        for row in result['adjacent_comparisons']:
            f=row['time']/.05;C,N=row['coarse_N'],row['fine_N']
            self.assertAlmostEqual(row['coarse_common_error'],f*{64:1/16,128:1/64}[C],places=14)
            self.assertAlmostEqual(row['native_fine_error'],f*{128:1/64,256:1/256}[N],places=14)
            self.assertAlmostEqual(row['common_indicator_components']['coarse16_32'],f/4096,places=14)
            amplitude=1-.25*f
            expected=2*abs((amplitude+.0003)**2-(amplitude+.0001)**2)
            self.assertAlmostEqual(row['H_indicator_components']['twice_reference_space'],expected,places=14)
        self.assertEqual(before,[v.tobytes() for v in values])
        self.assertEqual(common['n256_k32_states'].shape,(33,4096))

    def test_native_error_blocks_three_level_projection_mask(self):
        result,common=compare(*fixture(hidden=.025))
        row=result['adjacent_comparisons'][-1]
        self.assertEqual(row['common_decision'],'fine_better_resolvably')
        self.assertEqual(row['H_decision'],'fine_better_resolvably')
        self.assertEqual(row['native_decision'],'fine_worse_resolvably')
        self.assertFalse(result['all_adjacent_times_common_native_H_resolved'])
        self.assertEqual(result['comparison_decision'],'adverse_at_a_checkpoint')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
