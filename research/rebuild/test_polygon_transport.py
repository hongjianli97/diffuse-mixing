"""2D orientation, exact geometric sweep and independent fluid mass checks."""
from pathlib import Path
import tempfile,unittest
import numpy as np
from becker import Becker
from becker_polygon_benchmark import quad_connectivity,true_vertices,actual_polygon_mass,direct_density_mass,physical_edge_mass_flux
from material_polygon_history import MaterialPolygonHistory,POLYGON_UNITS,signed_area,mesh_edge_area_flux


def inputs(nx=2):
    p=np.array([[float(i),float(j)] for j in range(2) for i in range(nx+1)])
    ids=np.array([f'v{i}' for i in range(len(p))]);cells=quad_connectivity(nx,1)
    return {'time':np.array([0.,.1]),'positions':np.stack((p,p+np.array([.03,-.01]))),
            'vertex_ids':ids,'cell_ids':np.array([f'c{i}' for i in range(nx)]),'cell_vertices':ids[cells],
            'mass':np.ones(nx),'c0':np.linspace(.2,.8,nx),'units':POLYGON_UNITS,
            'metadata':{'geometry_kind':'material_quadrilateral_cells'}}


class PolygonTransportChecks(unittest.TestCase):
    def test_signed_area_and_midpoint_edge_sweep_match_independent_dilation(self):
        kw=inputs(1);p=kw['positions'][0];kw['positions']=np.stack((p,1.2*p))
        h=MaterialPolygonHistory(**kw)
        np.testing.assert_allclose(signed_area(h.positions[:,h.cell_vertex_indices]).ravel(),[1.,1.44],atol=2e-16)
        g=h.geometric_conservation()
        self.assertLess(g['maximum_area_increment_residual'],2e-15)
        self.assertAlmostEqual(g['left_endpoint_sweep_residual_negative_control'],.04,places=13)

    def test_positive_endpoint_cells_can_collapse_inside_interval(self):
        kw=inputs(1);p=kw['positions'][0];kw['positions']=np.stack((p,-p))
        np.testing.assert_allclose(signed_area(kw['positions'][:,[0,1,3,2]]),1.)
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)

    def test_quadratic_minimum_resolves_valid_quarter_turn(self):
        kw=inputs(1);p=kw['positions'][0];rot=np.stack((-p[:,1],p[:,0]),axis=-1)
        kw['positions']=np.stack((p,rot));h=MaterialPolygonHistory(**kw)
        self.assertAlmostEqual(h.minimum_interval_area,.5,places=14)
        self.assertAlmostEqual(float(h.evaluate(.05)['area'][0]),.5,places=14)

    def test_shared_edge_is_single_and_exactly_oppositely_oriented(self):
        h=MaterialPolygonHistory(**inputs());internal=~h.boundary_edge_mask
        self.assertEqual(int(internal.sum()),1);np.testing.assert_array_equal(h.incidence[:,internal].sum(axis=0),0)
        g=h.evaluate([.025,.075]);flux=g['edge_area_flux'];sign=h.incidence.sum(axis=0)
        np.testing.assert_allclose(g['cell_area_rate'].sum(axis=1),flux@sign,atol=1e-15)

    def test_invalid_connectivity_identity_units_and_crossings_are_rejected(self):
        kw=inputs();kw['cell_vertices'][1]=kw['cell_vertices'][1,::-1]
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)
        kw=inputs();kw['cell_vertices'][0,0]='missing'
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)
        kw=inputs();kw['units']={**POLYGON_UNITS,'position':'cm'}
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)
        kw=inputs();kw['positions'][1,1]=kw['positions'][1,0]
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)
        kw=inputs();ids=np.tile(kw['vertex_ids'],(2,1));ids[1,[0,1]]=ids[1,[1,0]];kw['frame_vertex_ids']=ids
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)

    def test_snapshot_roundtrip_preserves_ids_edges_and_no_overwrite(self):
        kw=inputs();h=MaterialPolygonHistory(**kw);kw['positions'][:]=0.
        self.assertGreater(float(h.evaluate(.05)['area'].min()),0.)
        with self.assertRaises(ValueError):h.mass[0]=2.
        with tempfile.TemporaryDirectory(prefix='ltn-polygon-',dir='/tmp') as tmp:
            p=Path(tmp)/'mesh.npz';h.save(p);loaded=MaterialPolygonHistory.load(p)
            self.assertEqual(h.identity_digest,loaded.identity_digest)
            np.testing.assert_array_equal(h.cell_edge_signs,loaded.cell_edge_signs)
            np.testing.assert_array_equal(loaded.evaluate(loaded.time)['positions'],h.positions)
            with self.assertRaises(FileExistsError):h.save(p)

    def test_time_and_positive_mass_contract_rejects_invalid_queries(self):
        h=MaterialPolygonHistory(**inputs())
        for q in (-.01,.11,np.nan):
            with self.assertRaises(ValueError):h.evaluate(q)
        kw=inputs();kw['mass'][0]=0.
        with self.assertRaises(ValueError):MaterialPolygonHistory(**kw)

    def test_true_2D_transformed_cells_preserve_exact_mass_and_area(self):
        f=Becker();e=f.mu(np.linspace(-1.,1.,9));y=np.array([-.5,0.,.5]);t=np.array([0.,.03,.1])
        cells=quad_connectivity(8,2);p=true_vertices(f,e,y,t)
        area=signed_area(p[:,cells]);expected=np.tile(np.diff(f.x_mu(e[None,:],t[:,None]),axis=1)*.5,(1,2))
        np.testing.assert_allclose(area,expected,atol=1e-15,rtol=1e-13)
        mass=actual_polygon_mass(f,p,cells,t)
        np.testing.assert_allclose(mass,np.tile(np.diff(e)*.5,(len(t),2)),atol=1e-14,rtol=1e-12)

    def test_Reynolds_shared_edge_flux_matches_mass_derivative_and_density_integral(self):
        f=Becker();e=f.mu(np.linspace(-1.,1.,9));y=np.array([-.5,0.,.5]);t=np.array([0.,.05,.1])
        cells=quad_connectivity(8,2);p=true_vertices(f,e,y,t);ids=np.array([f'v{i}' for i in range(p.shape[1])])
        h=MaterialPolygonHistory(t,p,ids,np.array([f'c{i}' for i in range(len(cells))]),ids[cells],
                               np.tile(np.diff(e)*.5,2),np.full(len(cells),.5),POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
        q=np.array([.025]);eps=1e-6
        massplus=actual_polygon_mass(f,h.evaluate(q+eps)['positions'],cells,q+eps)
        massminus=actual_polygon_mass(f,h.evaluate(q-eps)['positions'],cells,q-eps)
        flux=physical_edge_mass_flux(h,f,q)
        np.testing.assert_allclose((massplus-massminus)/(2*eps),-flux@h.incidence.T,atol=1e-8,rtol=1e-5)
        modeled=h.evaluate(q)['positions'];m=actual_polygon_mass(f,modeled,cells,q)
        spot=direct_density_mass(f,modeled[0,cells[4]],q[0],order=64)
        self.assertAlmostEqual(spot,float(m[0,4]),places=10)

    def test_density_scales_with_fixed_mass_without_oracle_inputs(self):
        kw=inputs();h=MaterialPolygonHistory(**kw);kw['mass']=13*kw['mass'];scaled=MaterialPolygonHistory(**kw)
        a=h.evaluate([0.,.03,.1]);b=scaled.evaluate([0.,.03,.1])
        np.testing.assert_allclose(a['density']*a['area'],np.broadcast_to(h.mass,a['area'].shape),atol=1e-15)
        np.testing.assert_allclose(b['density'],13*a['density'],rtol=1e-14)
        np.testing.assert_array_equal(a['edge_area_flux'],b['edge_area_flux'])


if __name__=='__main__':unittest.main(verbosity=2)
