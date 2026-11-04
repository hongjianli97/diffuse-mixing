"""Three necessary checks of the new common-material observation."""
from copy import deepcopy
from decimal import Decimal as Dec,localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import unittest

from nested_material_observation import restrict_material_means
from contact_spectral_reference import TrigPi,dec,material_factors


class NestedObservationChecks(unittest.TestCase):
    def test_unequal_child_masses_conserve_inventory_and_contract_mass_norm(self):
        mass=[F(i+1) for i in range(16)];values=[F((i*7)%11,10) for i in range(16)]
        coarse=[sum(mass[(2*j+b)*4+2*i+a] for b in range(2) for a in range(2)) for j in range(2) for i in range(2)]
        before=deepcopy((mass,values,coarse));result=restrict_material_means(mass,values,coarse,2)
        first=[0,1,4,5]
        self.assertEqual(result.means[0],sum(mass[k]*values[k] for k in first)/sum(mass[k] for k in first))
        self.assertNotEqual(result.means[0],sum(values[k] for k in first)/4)
        self.assertEqual(result.exact_inventory_residual,0);self.assertGreaterEqual(result.jensen_square_gap,0)
        self.assertEqual((mass,values,coarse),before)

    def test_true_initial_and_transmitted_modal_averages_nest_exactly_in_theory(self):
        with localcontext() as ctx:
            ctx.prec=80;t=TrigPi();a=[dec(F(2,7)),Dec(0),dec(F(1,3))];b=[Dec(0),dec(F(1,5)),Dec(0)]
            factors=[material_factors(t,a,b,N,dec(F(1,50)),dec(F(1,20))) for N in [4,8]]
            for which in [0,2]: # endpoint and original continuous initial formula
                coarse=[Dec('.5')+x*y for y in factors[0][1] for x in factors[0][which]]
                fine=[Dec('.5')+x*y for y in factors[1][1] for x in factors[1][which]]
                mf=[F(1 if i<4 else 4,64) for j in range(8) for i in range(8)]
                mc=[F(1 if i<2 else 4,16) for j in range(4) for i in range(4)]
                observed=restrict_material_means(mf,list(map(F,fine)),mc,4)
                self.assertLess(max(abs(dec(x)-y) for x,y in zip(observed.means,coarse)),Dec('1e-65'))

    def test_inconsistent_mass_or_nesting_is_rejected_without_renormalization(self):
        mass=[F(1,16)]*16;coarse=[F(1,4)]*4;values=[F(1,3)]*16
        bad=list(coarse);bad[0]+=F(1,10**30)
        for args in [(mass,values,bad,2),(mass[:-1],values,coarse,2),
                     ([F(-1),*mass[1:]],values,coarse,2)]:
            with self.assertRaises(ValueError):restrict_material_means(*args)
        with self.assertRaises(ValueError):restrict_material_means(mass,values,coarse,2,ratio=0)
        with self.assertRaises(TypeError):restrict_material_means(mass,values,coarse,2,normalize=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(NestedObservationChecks))
    summary={'checks_run':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),'previous_checks_rerun':0}
    if args.json:
        with Path(args.json).open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
