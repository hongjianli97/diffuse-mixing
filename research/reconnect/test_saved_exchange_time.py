"""Five new time-action checks on supplied small graphs only."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import copy
import json
import math
import unittest
import numpy as np
from scipy.linalg import expm

from saved_exchange_time import load_stencil,build_modal_blocks,modal_actions,stencil_action,uniformization,poisson_tail_bound


def fixture():
    N=4;offsets=[(0,1),(1,-1),(1,0),(1,1)];g=[];mapping=[]
    for c in range(N):
        row=[]
        for dx,dy in offsets:
            row.append(len(g));g.append(F(c+2,17) if dx else F(c+1,13))
        mapping.append(row)
    cache={'N':N,'offsets_positive_half':offsets,'conductances_exact':list(map(str,g)),'coefficient_map':mapping}
    mass=np.tile([1.,4.,9.,4.],N)
    positions=np.array([((c+.5)/N,(r+.5)/N) for r in range(N) for c in range(N)])
    ids=np.arange(N*N);state=np.array([.2+.03*((i*i+3*i)%17) for i in range(N*N)])
    return cache,mass,ids,positions,state


def dense_generator(cache,mass):
    N=cache['N'];Q=np.zeros((N*N,N*N))
    for r in range(N):
        for c in range(N):
            i=r*N+c
            for k,(dx,dy) in enumerate(cache['offsets_positive_half']):
                j=((r+dy)%N)*N+(c+dx)%N;g=float(F(cache['conductances_exact'][cache['coefficient_map'][c][k]]))
                Q[i,j]+=g/mass[i];Q[i,i]-=g/mass[i]
                Q[j,i]+=g/mass[j];Q[j,j]-=g/mass[j]
    return Q


class SavedExchangeTimeChecks(unittest.TestCase):
    def test_modal_and_stencil_action_match_independent_dense_shared_graph(self):
        cache,m,ids,x,c=fixture();s=load_stencil(cache,m,ids,x);modes=build_modal_blocks(s)
        direct=dense_generator(cache,m)@c
        np.testing.assert_allclose(stencil_action(s,c),direct,rtol=2e-14,atol=2e-15)
        np.testing.assert_allclose(modal_actions(modes,c,0)['initial_action'],direct,rtol=3e-14,atol=3e-15)
        self.assertTrue(all(np.array_equal(H,H.T) for H in modes.blocks))

    def test_two_time_actions_match_independent_dense_exponential(self):
        cache,m,ids,x,c=fixture();s=load_stencil(cache,m,ids,x);modes=build_modal_blocks(s)
        T=.05;Lam=math.ceil(float(max(s.outgoing)));M=24
        modal=modal_actions(modes,c,T)['state'];series=uniformization(s,c,T,Lam,M)
        truth=expm(T*dense_generator(cache,m))@c
        np.testing.assert_allclose(modal,truth,rtol=3e-14,atol=3e-15)
        np.testing.assert_allclose(series['state'],truth,rtol=3e-14,atol=3e-15)
        self.assertGreaterEqual(series['raw_minimum_embedded_diagonal'],0)

    def test_all_modes_roundtrip_and_raw_eigenvalues_are_retained(self):
        cache,m,ids,x,c=fixture();s=load_stencil(cache,m,ids,x);modes=build_modal_blocks(s)
        result=modal_actions(modes,c,0)
        self.assertEqual(result['full_eigen_coefficients'].shape,(4,4))
        self.assertGreater(np.linalg.norm(result['full_Fourier_coefficients'][2]),0)
        np.testing.assert_allclose(result['initial_roundtrip'],c,rtol=2e-14,atol=2e-15)
        for k,H in enumerate(modes.blocks):np.testing.assert_array_equal(modes.eigenvalues[k],np.linalg.eigh(H)[0])
        constant=modal_actions(modes,np.ones(16)*.37,.05)['state']
        np.testing.assert_allclose(constant,.37,rtol=2e-14,atol=2e-15)

    def test_invalid_cache_geometry_mass_and_time_are_rejected(self):
        cache,m,ids,x,c=fixture()
        bad=copy.deepcopy(cache);bad['conductances_exact'][0]='-1'
        with self.assertRaises(ValueError):load_stencil(bad,m,ids,x)
        bad=copy.deepcopy(cache);bad['conductances_exact'][1]='1/3'
        with self.assertRaises(ValueError):load_stencil(bad,m,ids,x)
        badm=m.copy();badm[-1]=0
        with self.assertRaises(ValueError):load_stencil(cache,badm,ids,x)
        badx=x.copy();badx[0,0]+=.001
        with self.assertRaises(ValueError):load_stencil(cache,m,ids,badx)
        s=load_stencil(cache,m,ids,x)
        with self.assertRaises(ValueError):uniformization(s,c,.05,0,47)
        with self.assertRaises(ValueError):modal_actions(build_modal_blocks(s),c,-1)

    def test_tail_bound_and_raw_weights_not_forced_to_sum_one(self):
        z=F(42,5);bound=poisson_tail_bound(z,47)
        self.assertLessEqual(bound,F(1,10**16));self.assertGreater(poisson_tail_bound(z,46),F(1,10**16))
        self.assertEqual(bound,z**48/math.factorial(48)/(1-z/F(49)))
        cache,m,ids,x,c=fixture();s=load_stencil(cache,m,ids,x);Lam=math.ceil(float(max(s.outgoing)))
        result=uniformization(s,c,.05,Lam,2)
        a=math.exp(-Lam*.05);raw=[a,a*(Lam*.05),a*(Lam*.05)**2/2]
        np.testing.assert_allclose(result['weights'],raw,rtol=2e-16,atol=0)
        self.assertEqual(result['raw_weight_sum'],math.fsum(map(float,result['weights'])))
        self.assertLess(result['raw_weight_sum'],1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json',required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SavedExchangeTimeChecks))
    with Path(args.json).open('x') as f:
        json.dump({'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(not result.wasSuccessful())
