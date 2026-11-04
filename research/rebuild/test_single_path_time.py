"""Checks that endpoint states belong to one uninterrupted BE path."""
import unittest
import numpy as np
from scipy.linalg import eigh,expm
from contact import laplacian
from fixed_be_time import integrate
from single_path_time import advance_path,modal_path


class SinglePathChecks(unittest.TestCase):
    def test_two_cell_checkpoints_match_exact_product_of_segment_factors(self):
        m=np.array([1.,3.]);g=np.array([2.]);c0=np.array([.2,.8]);times=np.array([.001,.01,.1]);counts=np.array([10,20,40])
        states,audits=advance_path(c0,m,g,times,counts);mean=m@c0/m.sum();rate=g[0]*(1/m[0]+1/m[1]);d=.6;last=0.
        for i,(t,n) in enumerate(zip(times,counts)):
            d*=(1+(t-last)/n*rate)**(-n)
            np.testing.assert_allclose(states[i],mean+d*np.array([-.75,.25]),atol=2e-14)
            self.assertEqual(audits[i]['cumulative_steps'],int(counts[:i+1].sum()));last=t

    def test_modal_path_and_accumulated_time_difference_match_actual_states(self):
        m=np.array([.1,.3,.6]);g=np.array([2.,.4]);c0=np.array([1.,.4,0.]);T=np.array([.001,.01,.1]);counts=[8,16,64]
        L=laplacian(g);root=np.sqrt(m);vals,U=eigh(L/root[:,None]/root[None,:])
        forecast=modal_path(vals,U,m,c0,T,counts);states,logs=advance_path(c0,m,g,T,counts)
        for t,(pred,delta),actual in zip(T,forecast,states):
            np.testing.assert_allclose(actual,pred,atol=5e-14)
            continuous=expm(-t*L/m[:,None])@c0
            np.testing.assert_allclose(actual-continuous,delta,atol=5e-14)

    def test_segments_with_identical_dt_equal_uniform_run_without_reset(self):
        m=np.array([.2,.3,.5]);g=np.array([1.,.2]);c0=np.array([.1,.9,.3])
        path,logs=advance_path(c0,m,g,[.02,.1],[20,80]);uniform,d=integrate(c0,m,g,.1,100)
        np.testing.assert_allclose(path[-1],uniform,atol=3e-14)
        reset,d=integrate(c0,m,g,.08,80)
        self.assertGreater(np.linalg.norm(path[-1]-reset),.001)

    def test_constants_and_zero_exchange_survive_step_size_changes(self):
        m=np.array([.1,.9]);times=[.001,.1];counts=[10,13]
        for c0,g in ((np.zeros(2),np.ones(1)),(np.ones(2),np.ones(1)),(np.array([.2,.8]),np.zeros(1))):
            states,logs=advance_path(c0,m,g,times,counts)
            np.testing.assert_allclose(states,np.tile(c0,(2,1)),atol=2e-14)

    def test_invalid_time_paths_and_segment_counts_are_rejected(self):
        m=np.ones(2);g=np.ones(1);c0=np.array([0.,1.])
        for times,counts in (([.1,.05],[4,4]),([0.,.1],[4,4]),([.1],[0]),([.1],[1.5]),([.1,.2],[2])):
            with self.assertRaises(ValueError):advance_path(c0,m,g,times,counts)


if __name__=='__main__':unittest.main(verbosity=2)
