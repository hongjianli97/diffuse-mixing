"""Physical partition, true averaged moments and one refined interior context."""
import unittest
import numpy as np
from becker import Becker,initial
from becker_polygon_benchmark import quad_connectivity,true_vertices
from material_polygon_history import signed_area
from nonorthogonal_patch_study import oracle_mass_centroids
from mass_cone_study import stream_cell_moments,primitive_targets,independent_boundary_targets,build_support
from mass_moment_cone import transverse_drift_separator,require_bulk_context
from transverse_partition import split_transverse_parallelograms,logical_child_layout,neighbour_child_indices,polynomial_averages,direct_mass_polynomials


def example(nx=3,parent_rows=2):
    points=np.array([[i*.25+.6*(j*.5-.5),j*.5-.5] for j in range(parent_rows+1) for i in range(nx+1)])
    quads=points[quad_connectivity(nx,parent_rows)];mass=signed_area(quads);c0=np.linspace(.2,.8,len(quads))
    centers=quads.mean(axis=1);v=np.full(len(quads),.25**2/12)
    children=split_transverse_parallelograms(quads,mass,c0,centers,v,transverse_mass_uniform=True,transverse_initial_state_uniform=True)
    return quads,mass,c0,centers,v,children


class HalfConeChecks(unittest.TestCase):
    def test_exact_mass_component_and_identity_lineage(self):
        p,m,c,b,v,h=example();np.testing.assert_array_equal(h['mass'].reshape(-1,2).sum(axis=1),m)
        np.testing.assert_array_equal(h['c0'].reshape(-1,2),np.repeat(c[:,None],2,axis=1))
        np.testing.assert_allclose((h['mass']*h['c0']).reshape(-1,2).sum(axis=1),m*c,atol=1e-16)
        self.assertEqual(len(set(h['child_ids'])),2*len(m))
        np.testing.assert_array_equal(h['parent_index'],np.repeat(np.arange(len(m)),2))

    def test_law_of_total_covariance_requires_between_child_centroids(self):
        p,m,c,b,v,h=example();n=len(m);offset=h['centroid_offsets'].reshape(n,2,2)
        total=(h['covariance'].reshape(n,2,2,2)+offset[:,:,:,None]*offset[:,:,None,:]).mean(axis=1)
        np.testing.assert_allclose(total,h['parent_covariance'],atol=1e-16)
        self.assertGreater(float(np.max(abs(h['covariance'].reshape(n,2,2,2).mean(axis=1)-h['parent_covariance']))),.01)

    def test_linear_and_quadratic_mass_average_restriction(self):
        p,m,c,b,v,h=example();origin=np.array([.13,-.2]);axes=np.eye(2)
        parent=polynomial_averages(b,h['parent_covariance'],origin,axes)
        child=polynomial_averages(h['centroids'],h['covariance'],origin,axes)
        np.testing.assert_allclose(child.reshape(len(m),2,5).mean(axis=1),parent,atol=3e-16)
        self.assertGreater(float(np.min(parent[:,4]-(b[:,1]-origin[1])**2)),.02)

    def test_shared_split_edge_reversed_area_and_no_parent_mutation(self):
        p,m,c,b,v,h=example();before=[a.copy() for a in (p,m,c,b,v)]
        h=split_transverse_parallelograms(p,m,c,b,v,transverse_mass_uniform=True,transverse_initial_state_uniform=True)
        for j in range(len(m)):
            np.testing.assert_array_equal(h['quads'][2*j,[2,3]],h['quads'][2*j+1,[1,0]])
        np.testing.assert_allclose(signed_area(h['quads']).reshape(-1,2).sum(axis=1),signed_area(p),atol=3e-16)
        for old,now in zip(before,(p,m,c,b,v)):np.testing.assert_array_equal(old,now)

    def test_becker_half_mass_and_centroid_against_independent_density_integrals(self):
        f=Becker();edges=f.mu(np.linspace(-1.,1.,33));y=np.array([-.5,0.,.5]);t=.05
        p=true_vertices(f,edges,y,np.array([t]))[0];quads=p[quad_connectivity(32,2)]
        b=oracle_mass_centroids(f,edges,y,t);means,var=stream_cell_moments(f,edges,t);mass=np.tile(np.diff(edges)*.5,2)
        h=split_transverse_parallelograms(quads,mass,np.tile(initial(edges),2),b,np.tile(var,2),transverse_mass_uniform=True,transverse_initial_state_uniform=True)
        child=33;actual,poly=direct_mass_polynomials(f,h['quads'][child:child+1],h['mass'][child:child+1],t,h['centroids'][child],np.eye(2),order=64)
        np.testing.assert_allclose(actual,h['mass'][child:child+1],rtol=1e-12,atol=0.)
        np.testing.assert_allclose(poly[0,:2],0.,atol=3e-15)
        self.assertGreater(float(np.linalg.norm(h['centroids'][child]-h['quads'][child].mean(axis=0))),1e-5)

    def test_selected_child_is_interior_and_all_38_neighbours_are_actual_children(self):
        p,m,c,b,v,h=example(nx=32);column,row,boundary=logical_child_layout()
        self.assertEqual(int(np.count_nonzero(~boundary)),60);self.assertFalse(boundary[33]);require_bulk_context(bool(boundary[33]),'actual_interior')
        means=np.arange(32)*.25+.125;A,support=build_support(means,np.full(32,.25**2/12),16,range(10,23),hy=.25)
        indices=neighbour_child_indices(support['neighbour_column_and_virtual_row'],33)
        self.assertEqual(len(indices),38);self.assertTrue(np.all(indices<128));self.assertFalse(np.any(indices==33))
        np.testing.assert_allclose((h['centroids'][indices]-h['centroids'][33]).T,A[:2],atol=2e-15)
        self.assertEqual(set(row[indices]),{0,1,2})

    def test_invalid_assumption_geometry_and_neighbour_extension_rejected(self):
        p,m,c,b,v,h=example()
        with self.assertRaises(ValueError):split_transverse_parallelograms(p,m,c,b,v)
        with self.assertRaises(ValueError):split_transverse_parallelograms(p,m,c,b,v,transverse_mass_uniform=True)
        bad=p.copy();bad[0,2,0]+=.001
        with self.assertRaises(ValueError):split_transverse_parallelograms(bad,m,c,b,v,transverse_mass_uniform=True,transverse_initial_state_uniform=True)
        badb=b.copy();badb[0,0]+=10
        with self.assertRaises(ValueError):split_transverse_parallelograms(p,m,c,badb,v,transverse_mass_uniform=True,transverse_initial_state_uniform=True)
        with self.assertRaises(ValueError):neighbour_child_indices([[16,-1]],32)

    def test_half_spacing_removes_only_the_necessary_drift_separator(self):
        means=np.arange(15)/16;var=np.full(15,.001);target=np.array([0.,.09,.04,0.,.04])
        coarse,_=build_support(means,var,7,range(2,13),hy=.5);half,_=build_support(means,var,7,range(2,13),hy=.25)
        self.assertTrue(transverse_drift_separator(coarse,target)['verified'])
        certificate=transverse_drift_separator(half,target);self.assertFalse(certificate['verified'])
        self.assertGreater(certificate['exact_target_dot_functional']['float'],0.)

    def test_boundary_integral_target_is_invariant_under_transverse_halving(self):
        f=Becker();edges=f.mu(np.linspace(-1.,1.,33));means,var=stream_cell_moments(f,edges,.05)
        beta,Gamma,J=primitive_targets(f,edges,means,16,.05)
        for hy in (.5,.25):
            b,G=independent_boundary_targets(f,edges,16,means[16],.05,hy=hy,order=64)
            np.testing.assert_allclose(b,beta,atol=4e-13);np.testing.assert_allclose(G,Gamma,atol=4e-14)

    def test_rigid_motion_and_physical_mass_scale_preserve_partition(self):
        p,m,c,b,v,h=example();theta=.37;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);shift=np.array([.7,-.2])
        other=split_transverse_parallelograms(p@R.T+shift,7*m,c,b@R.T+shift,v,transverse_mass_uniform=True,transverse_initial_state_uniform=True)
        np.testing.assert_allclose(other['centroids'],h['centroids']@R.T+shift,atol=3e-16)
        np.testing.assert_allclose(other['covariance'],R@h['covariance']@R.T,atol=1e-16)
        np.testing.assert_allclose(other['mass'],7*h['mass'],atol=0.,rtol=0.)
        np.testing.assert_array_equal(other['c0'],h['c0'])


if __name__=='__main__':unittest.main(verbosity=2)
