"""Graph-level tests of mainline interfaces; no claim of fluid PDE acceptance."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.linalg import expm
from trajectory_chain import TrajectorySamples,SI_UNITS,laplacian,reference_distances,propagate_supplied,operator_audit,mixing_metrics
from chain_study import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'material_fv'))
from material_fv import step


class ChainChecks(unittest.TestCase):
    def test_matches_existing_unequal_mass_chain_solver(self):
        d,p,g=fixture();r=propagate_supplied(d,p,g);c=d.c0.copy()
        for i,dt in enumerate(np.diff(d.time)):
            c=step(c,d.mass,g[i],dt);np.testing.assert_allclose(r['states'][i+1],c,atol=2e-15)

    def test_ordered_noncommuting_propagation_and_exact_control(self):
        d,p,g=fixture();r=propagate_supplied(d,p,g);A0,A1=r['steps'];np.testing.assert_allclose(r['P'],A1@A0,atol=0,rtol=0)
        self.assertGreater(np.max(abs(r['P']-A0@A1)),1e-5)
        Q=-r['laplacians']/d.mass[None,:,None];dt=np.diff(d.time)
        ordered=expm(dt[1]*Q[1])@expm(dt[0]*Q[0]);averaged=expm(np.einsum('s,sij->ij',dt,Q))
        self.assertGreater(np.max(abs(ordered-averaged)),1e-5)
        np.testing.assert_allclose(expm(dt[1]*2*Q[0])@expm(dt[0]*Q[0]),expm((dt[0]+2*dt[1])*Q[0]),atol=2e-15)

    def test_probability_orientation_and_component_mass_map(self):
        d,p,g=fixture();r=propagate_supplied(d,p,g);P=r['P'];R=r['forward_row_mass_kernel'];S=r['forward_column_component_mass_map']
        np.testing.assert_allclose(P.sum(axis=1),1,atol=2e-15);np.testing.assert_allclose(d.mass@P,d.mass,atol=2e-15)
        np.testing.assert_allclose(R.sum(axis=1),1,atol=2e-15);np.testing.assert_allclose(S.sum(axis=0),1,atol=2e-15)
        np.testing.assert_allclose(S@(d.mass*d.c0),d.mass*r['states'][-1],atol=2e-15)
        np.testing.assert_allclose((d.mass*d.c0)@R,d.mass*r['states'][-1],atol=2e-15)
        self.assertGreater(np.max(abs(P-R)),1e-5);self.assertGreater(np.max(abs(P.sum(axis=0)-1)),1e-4)

    def test_zero_exchange_preserves_mass_average_under_stirring(self):
        d,p,g=fixture();r=propagate_supplied(d,p,np.zeros_like(g));np.testing.assert_array_equal(r['P'],np.eye(3))
        np.testing.assert_array_equal(r['states'],np.broadcast_to(d.c0,(3,3)))
        self.assertFalse(np.array_equal(d.positions[0],d.positions[-1]));self.assertEqual(r['metrics'][-1]['I'],1.);self.assertEqual(r['metrics'][-1]['H'],0.)

    def test_reference_graph_is_distinct_from_current_exchange(self):
        d,p,g=fixture();distance,length=reference_distances(d,p);self.assertEqual(g[1,0],0.)
        self.assertAlmostEqual(distance[0,2],1.5);self.assertEqual(distance[0,1],length[0]);self.assertNotAlmostEqual(distance[0,2],d.positions[-1,2,0]-d.positions[-1,0,0])
        disconnected,_=reference_distances(d,np.array([[0,1]]));self.assertTrue(np.isinf(disconnected[0,2]))

    def test_energy_identity_separates_numerical_dissipation(self):
        d,p,g=fixture();r=propagate_supplied(d,p,g);total=0.
        for i,b in enumerate(r['energy_balances']):
            self.assertLess(abs(b['energy_identity_residual_kg']),2e-15);self.assertGreater(b['backward_euler_extra_dissipation_kg'],0.)
            total+=b['dt_chi_endpoint_kg']+b['backward_euler_extra_dissipation_kg']
            self.assertGreaterEqual(r['metrics'][i]['energy_kg'],r['metrics'][i+1]['energy_kg'])
        self.assertAlmostEqual(total,r['metrics'][0]['energy_kg']-r['metrics'][-1]['energy_kg'])

    def test_constant_initial_state_has_defined_status_without_division(self):
        d,p,g=fixture();constant=TrajectorySamples(d.time,d.positions,d.ids,d.mass,np.full(3,.37),SI_UNITS)
        r=propagate_supplied(constant,p,g);np.testing.assert_allclose(r['states'],.37,atol=2e-15)
        self.assertIsNone(r['metrics'][-1]['I']);self.assertIsNone(r['metrics'][-1]['H']);self.assertEqual(r['metrics'][-1]['status'],'constant_initial_state')

    def test_permutation_and_common_mass_conductance_scaling(self):
        d,p,g=fixture();r=propagate_supplied(d,p,g);perm=np.array([2,0,1]);inverse=np.argsort(perm);pp=np.sort(inverse[p],axis=1)
        dp=TrajectorySamples(d.time,d.positions[:,perm],d.ids[perm],d.mass[perm],d.c0[perm],SI_UNITS)
        rp=propagate_supplied(dp,pp,g);np.testing.assert_allclose(rp['P'],r['P'][np.ix_(perm,perm)],atol=2e-15)
        scaled=TrajectorySamples(d.time,d.positions,d.ids,7*d.mass,d.c0,SI_UNITS);rs=propagate_supplied(scaled,p,7*g)
        np.testing.assert_allclose(rs['P'],r['P'],atol=2e-15)

    def test_inputs_refuse_identity_changes_bad_units_negative_weights(self):
        d,p,g=fixture()
        with self.assertRaises(ValueError):TrajectorySamples(d.time,d.positions,d.ids,d.mass,d.c0,SI_UNITS,frame_ids=[d.ids,d.ids[::-1],d.ids])
        with self.assertRaises(ValueError):TrajectorySamples(d.time,d.positions,d.ids,d.mass,d.c0,{})
        with self.assertRaises(ValueError):TrajectorySamples(d.time,d.positions,d.ids,-d.mass,d.c0,SI_UNITS)
        with self.assertRaises(ValueError):propagate_supplied(d,p,-g)
        with self.assertRaises(ValueError):laplacian(3,np.array([[0,1],[0,1]]),[1.,1.])

    def test_signed_conservative_operator_is_not_a_markov_generator(self):
        m=np.array([1.,2.,3.]);L=np.array([[-1.,1.,0.],[1.,0.,-1.],[0.,-1.,1.]])
        a=operator_audit(m,-L/m[:,None]);self.assertFalse(a['graph_algebra_eligible']);self.assertLess(a['minimum_raw_offdiagonal_per_s'],0.)
        good=operator_audit(m,-laplacian(3,np.array([[0,1],[1,2]]),[.8,.4])/m[:,None])
        self.assertTrue(good['graph_algebra_eligible']);self.assertFalse(good['physical_acceptance_inferred'])


if __name__=='__main__':unittest.main(verbosity=2)
