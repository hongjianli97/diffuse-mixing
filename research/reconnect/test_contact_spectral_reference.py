"""Four necessary software checks for the independent contact reference."""
from decimal import Decimal as Dec, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import unittest

from contact_spectral_reference import (TrigPi,PIECES,dec,projection_products,projection_boundary,
    material_factors,tail_certificate,reference_case)


class SpectralReferenceChecks(unittest.TestCase):
    def test_weighted_modes_match_periodic_transmission_and_Gram(self):
        with localcontext() as ctx:
            ctx.prec=80;t=TrigPi()
            for n in [1,2,3]:
                for x in [F(0),F(1,2),F(1)]:
                    s,c=t.sc(2*n*x)
                    self.assertEqual(s,0)
                    self.assertEqual(c,Dec((-1)**(n*int(2*x))))
                    # cos normal derivative is zero; rho*(sin/rho)'=k*cos.
                    self.assertEqual(dec(1)*(2*t.pi*n*c/dec(1)),dec(4)*(2*t.pi*n*c/dec(4)))
                for m in [1,2,3]:
                    cc=ss=cs=Dec(0)
                    for lo,hi,rho,_,_ in PIECES:
                        cc+=dec(rho)/2*(t.integral_cos(2*(n-m),F(0),lo,hi)+t.integral_cos(2*(n+m),F(0),lo,hi))
                        ss+=1/(2*dec(rho))*(t.integral_cos(2*(n-m),F(0),lo,hi)-t.integral_cos(2*(n+m),F(0),lo,hi))
                        cs+=(t.integral_sin(2*(n+m),F(0),lo,hi)+t.integral_sin(2*(m-n),F(0),lo,hi))/2
                    self.assertEqual(cc,dec(F(5,4)) if n==m else 0)
                    self.assertEqual(ss,dec(F(5,16)) if n==m else 0)
                    self.assertLess(abs(cs),Dec('1e-65'))

    def test_original_closed_projections_match_boundary_identity_and_keep_zero_mode(self):
        with localcontext() as ctx:
            ctx.prec=80;t=TrigPi();a,b=projection_products(t,128)
            self.assertGreater(a[0],0)
            self.assertLess(abs(a[0]-3*t.sc(F(1,5))[0]/(4*t.pi)),Dec('1e-65'))
            for n in [1,2,64,128]:
                x,y=projection_boundary(t,n)
                self.assertLess(max(abs(a[n]-x),abs(b[n]-y)),Dec('1e-65'))
            self.assertTrue(all(a[n]==0 for n in range(1,129,2)))
            self.assertTrue(all(b[n]==0 for n in range(2,129,2)))

    def test_true_cell_averages_of_modes_and_time_decay_not_point_values(self):
        with localcontext() as ctx:
            ctx.prec=80;t=TrigPi();D=dec(F(1,50));T=dec(F(1,20));N=8
            alpha=[dec(F(2,7)),Dec(0),dec(F(1,3))];beta=[Dec(0),dec(F(1,5)),Dec(0)]
            normal,y,_=material_factors(t,alpha,beta,N,D,T)
            tau=4*t.pi*t.pi*D*T
            for i in range(N):
                lo,hi=F(i,N),F(i+1,N);rho=F(1) if i<N//2 else F(4)
                expected=(-tau).exp()*(alpha[0]+(-4*tau).exp()*alpha[2]*N*t.integral_cos(F(4),F(0),lo,hi)
                    +(-tau).exp()*beta[1]*N*t.integral_sin(F(2),F(0),lo,hi)/dec(rho))
                self.assertLess(abs(normal[i]-expected),Dec('1e-65'))
                self.assertLess(abs(y[i]-N*t.integral_cos(F(2),F(0),lo,hi)),Dec('1e-65'))
            alias=[Dec(0)]*(N+1);alias[N]=Dec(1)
            vanished,_,_=material_factors(t,alias,[Dec(0)]*(N+1),N,D,T)
            self.assertEqual(vanished,[Dec(0)]*N)
            self.assertNotEqual(t.sc(F(1,N))[1],y[0])

    def test_rational_tail_bound_and_declared_problem_rejection(self):
        with localcontext() as ctx:
            ctx.prec=80;t=TrigPi();D=F(5764607523034235,288230376151711744);T=F(1,20)
            for K in [64,128]:
                cert=tail_certificate(K,D,T);bound=dec(cert['material_mass_norm_tail_bound_exact'])
                self.assertLess(Dec('.25')*(-4*t.pi*t.pi*dec(D)*dec(T)*((K+1)**2+1)).exp(),bound)
                self.assertGreater(F(cert['exp_step_partial_sum_lower_verified']),10)
            with self.assertRaises(ValueError):material_factors(t,[Dec(1)],[Dec(0)],7,dec(D),dec(T))
            with self.assertRaises(ValueError):material_factors(t,[Dec(1)],[Dec(0)],8,-dec(D),dec(T))
            with self.assertRaises(ValueError):reference_case({'N':127,'rho_left':'1','rho_right':'4'},64,60)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpectralReferenceChecks))
    summary={'checks_run':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),'previous_checks_rerun':0}
    if args.json:
        with Path(args.json).open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
