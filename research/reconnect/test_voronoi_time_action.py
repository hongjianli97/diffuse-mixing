"""Small independent analytic and dense controls, not extra fluid cases."""
from fractions import Fraction as F
from decimal import Decimal, localcontext
import argparse
import json
import math
import unittest
import numpy as np
from scipy.linalg import expm
from voronoi_time_action import operators, exponential, uniformization, poisson_plan


class Checks(unittest.TestCase):
    def test_unequal_mass_analytic(self):
        m = np.array([1., 3.]); c = np.array([.2, .8]); g = F(2, 5); dt = F(3, 7)
        op = operators(m, np.array([[0, 1]]), [g])
        mean = m @ c / m.sum()
        expected = mean + (c - mean) * math.exp(-float(g) * (1/m[0] + 1/m[1]) * float(dt))
        primary, _ = exponential(op, c, dt)
        control, _, info = uniformization(op, c, dt)
        np.testing.assert_allclose(primary, expected, atol=4e-15, rtol=0)
        np.testing.assert_allclose(control, expected, atol=4e-15, rtol=0)
        self.assertLessEqual(F(info['tail_upper_exact']), F(1, 10**16))

    def test_noncommuting_chronology(self):
        m = np.array([1., 2., 3.]); c = np.array([.1, .4, .9]); dt = F(1, 3)
        edges = np.array([[0, 1], [1, 2]])
        a = operators(m, edges, [F(1), F(0)])
        b = operators(m, edges, [F(0), F(2)])
        Qa = np.array([[-1,1,0],[.5,-.5,0],[0,0,0.]])
        Qb = np.array([[0.,0,0],[0,-1,1],[0,2/3,-2/3]])
        expected = expm(float(dt)*Qb) @ expm(float(dt)*Qa) @ c
        primary, control = c.copy(), c.copy()
        for op in (a, b):
            primary, _ = exponential(op, primary, dt)
            control, _, _ = uniformization(op, control, dt)
        np.testing.assert_allclose(primary, expected, atol=6e-15, rtol=0)
        np.testing.assert_allclose(control, expected, atol=6e-15, rtol=0)
        self.assertGreater(np.max(abs(expected-expm(float(dt)*Qa)@expm(float(dt)*Qb)@c)), .001)

    def test_poisson_tail_and_no_renormalization(self):
        for z in (F(0), F(1, 4), F(10), F(25)):
            M, upper = poisson_plan(z)
            with localcontext() as ctx:
                ctx.prec = 90
                x = Decimal(z.numerator)/Decimal(z.denominator)
                weight = (-x).exp(); total = weight
                for k in range(1, M+1):
                    weight *= x/k; total += weight
                tail = 1-total
                bound = Decimal(upper.numerator)/Decimal(upper.denominator)
                self.assertGreaterEqual(tail, Decimal('-1e-85'))
                self.assertLessEqual(tail, bound+Decimal('1e-85'))
                if M:
                    previous = z**M / math.factorial(M)
                    ratio = z/(M+1)
                    self.assertTrue(ratio >= 1 or previous/(1-ratio)>F(1,10**16))
        op = operators(np.ones(2), np.array([[0,1]]), [F(1)])
        result, weights, info = uniformization(op, np.ones(2), F(1), budget=F(1,100))
        self.assertLess(float(result[0]), .9999)
        np.testing.assert_allclose(result, np.full(2, math.fsum(weights)), atol=2e-15, rtol=0)
        self.assertFalse(info['weights_or_state_normalized'])

    def test_zero_diffusion_and_constant(self):
        m = np.array([.7, 1.1, 2.3]); ij = np.array([[0,1],[1,2]])
        zero = operators(m, ij, [F(0),F(0)])
        c = np.array([.2,.7,.4])
        np.testing.assert_array_equal(exponential(zero,c,F(1))[0], c)
        np.testing.assert_array_equal(uniformization(zero,c,F(1))[0], c)
        op = operators(m, ij, [F(1,2),F(1,3)])
        c = np.full(3,.6)
        np.testing.assert_allclose(exponential(op,c,F(1,4))[0], c, atol=4e-15, rtol=0)
        np.testing.assert_allclose(uniformization(op,c,F(1,4))[0], c, atol=4e-15, rtol=0)

    def test_permutation_mass_and_energy(self):
        m=np.array([1.,2.,4.]);c=np.array([.1,.9,.4]);ij=np.array([[0,1],[1,2],[0,2]])
        g=[F(1,3),F(2,3),F(1,7)];op=operators(m,ij,g)
        p=np.array([2,0,1]);inverse=np.argsort(p)
        other=operators(m[p],inverse[ij],g)
        after=exponential(op,c,F(2,5))[0]
        np.testing.assert_allclose(exponential(other,c[p],F(2,5))[0],after[p],atol=6e-15,rtol=0)
        self.assertLess(abs(float(m@(after-c))),1e-14)
        self.assertGreaterEqual(float(after.min()),float(c.min()))
        self.assertLessEqual(float(after.max()),float(c.max()))
        mu=float(m@c/m.sum())
        self.assertLess(float(m@((after-mu)**2)),float(m@((c-mu)**2)))

    def test_invalid_input_rejected(self):
        for m,ij,g in (([1,0],[[0,1]],[F(1)]),([1,1],[[0,0]],[F(1)]),
                       ([1,1],[[0,2]],[F(1)]),([1,1],[[0,1]],[F(-1)])):
            with self.assertRaises(ValueError):operators(m,np.array(ij),g)
        op=operators([1.,2.],np.array([[0,1]]),[F(1)])
        for state,dt in (([.1,float('nan')],F(1)),([.1],F(1)),([.1,.2],F(0))):
            with self.assertRaises(ValueError):exponential(op,state,dt)
        with self.assertRaises(ValueError):poisson_plan(F(-1))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--json',required=True);args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    with open(args.json,'x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,
                   'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2)
        f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
