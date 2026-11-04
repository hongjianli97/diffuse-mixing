"""Unequal-mass semigroup bound and false-objectivity rejection fixtures."""
from fractions import Fraction as F
import argparse,json,math,unittest
import numpy as np
from scipy.linalg import expm
from t2_translation_core import coefficient_perturbation,exact_fluctuation_upper,compare


class Checks(unittest.TestCase):
    def test_exact_union_bound_for_unequal_mass_semigroups(self):
        m=np.array([.125,.25,.5]);c=np.array([.2,.4,.8]);before=m.tobytes(),c.tobytes();m.setflags(write=False);c.setflags(write=False)
        old={'ids':[7,3,9],'edges':[[0,1],[1,2]],'conductances_exact':['1/10','1/5']}
        new={'ids':[7,3,9],'edges':[[0,1],[1,2],[0,2]],'conductances_exact':['11/100','1/5','1/50']}
        r=coefficient_perturbation(m,old,new);eta=F(r['delta_generator_mass_norm_upper_exact'])
        self.assertEqual(eta,F(12,25));self.assertEqual(r['introduced_shared_edges'],1);self.assertFalse(r['original_algebra_scale_passed'])
        def generator(packet):
            Q=np.zeros((3,3))
            for (i,j),g in zip(packet['edges'],packet['conductances_exact']):
                g=float(F(g));Q[i,j]+=g/m[i];Q[i,i]-=g/m[i];Q[j,i]+=g/m[j];Q[j,j]-=g/m[j]
            return Q
        A,B=generator(old),generator(new);root=np.sqrt(m)
        self.assertLessEqual(float(np.linalg.norm(root[:,None]*(B-A)/root[None,:],2)),float(eta))
        variance,upper=exact_fluctuation_upper(m,c);self.assertGreaterEqual(F(upper)**2,variance)
        t=.125;delta=expm(t*B)@c-expm(t*A)@c
        self.assertLess(float(np.sqrt(m@(delta*delta))),t*float(eta)*upper)
        self.assertEqual(before,(m.tobytes(),c.tobytes()))

    def test_large_coefficient_budget_cannot_masquerade_as_objectivity(self):
        m=np.array([.5,.5]);c0=np.array([.25,.75]);times=np.arange(17)/320
        states=np.array([.5+(c0-.5)*math.exp(-2*t) for t in times])
        path={'states':states,'independent_states':states.copy()}
        run={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-16,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(16)]}
        d={'mass':m,'c0':c0,'reference':states[[0,8,16]],'reference_evaluation_indicator_absolute':1e-14}
        old={'ids':[0,1],'edges':[[0,1]],'conductances_exact':['1/2']}
        zero=coefficient_perturbation(m,old,old)
        result=compare(d,run,path,run,path,[zero]*16);self.assertTrue(result['T2_registered_translation_case_passed'])
        changed=dict(old,conductances_exact=['3/5']);large=coefficient_perturbation(m,old,changed)
        bad=compare(d,run,path,run,path,[large]*16)
        self.assertTrue(bad['objectivity_within_fixed_indicators']);self.assertFalse(bad['geometry_within_original_algebra_scale'])
        self.assertFalse(bad['T2_registered_translation_case_passed'])
        perturbed={k:v.copy() for k,v in path.items()};perturbed['states'][8:]+=np.array([.01,-.01])
        bad=compare(d,run,path,run,perturbed,[zero]*16)
        self.assertFalse(bad['objectivity_within_fixed_indicators']);self.assertFalse(bad['T2_registered_translation_case_passed'])
        self.assertFalse(bad['observations'][-1]['moving_sequence_within_original_allowance'])
        perturbed['independent_states'][8:]+=np.array([.01,-.01])
        bad=compare(d,run,path,run,perturbed,[zero]*16)
        self.assertTrue(bad['observations'][-1]['moving_sequence_within_original_allowance'])
        self.assertFalse(bad['objectivity_within_fixed_indicators'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
