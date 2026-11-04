"""Three small software fixtures, not additional scientific flow cases."""
from fractions import Fraction as F
import argparse,copy,json,math,unittest
import numpy as np
from affine_shear_core import COUNTS,pure_inputs,reference_bundle,frozen_coefficients,directional_certificate,modal_certificate
from affine_shear_spatial_core import compare
from sliding_periodic_voronoi import SlidingPeriodicDomain,build_sliding_faces
from voronoi_time_action import operators
from material_union import observe_path,group_sum
from t1_uniform_core import pure_labels


def fixture():
    data={};labels={};worlds={};runs={};paths={};res={}
    for N in (3,6,12):
        a=pure_labels(N);m=np.full(N*N,1/(N*N));c0=.5+.2*(math.sin(math.pi/N)/(math.pi/N))**2*np.sin(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1])
        data[N]={'mass':m,'c0':c0};labels[N]=a;worlds[N]=pure_inputs(a,np.arange(N*N));b=reference_bundle(data[N],a,worlds[N],N)
        u=math.sin(math.pi/N)**2;v=math.sin(2*math.pi/N)**2
        for K in COUNTS:
            states=[c0.copy()]
            for i in range(1,K+1):
                U,V=frozen_coefficients(worlds[N],K,i)
                states.append(.5+b['modes'][0]*math.exp(-4*N*N*u*float(U))+b['modes'][1]*math.exp(-4*N*N*(u*float(U)+v*float(V))))
            states=np.array(states);paths[N,K]={'states':states,'independent_states':states.copy()};res[N,K]=[0.]*K
            runs[N,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':0.} for i in range(K)]}
    return data,labels,worlds,runs,paths,res


class Checks(unittest.TestCase):
    def test_multi_N_world_directions_moments_and_mode_guard_rejection(self):
        for N in (4,8):
            labels=pure_labels(N);world=pure_inputs(labels,np.arange(N*N));frame=world['cases']['4']['frames'][3]
            packet,wp,cert=build_sliding_faces(world['ids'],frame['world_positions_exact'],frame['rho'],.02,SlidingPeriodicDomain(frame['shear_exact']))
            self.assertTrue(directional_certificate(packet,frame,N)['all_geometry_controls_passed'])
            m=np.full(N*N,1/(N*N));op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']));A=op.symmetric_generator
            action={'symmetric_data':A.data,'symmetric_indices':A.indices,'symmetric_indptr':A.indptr,'shape':A.shape}
            data={'mass':m,'c0':np.full(N*N,.5)}
            self.assertTrue(modal_certificate(data,labels,frame,action,N,0)['all_original_residue_guards_passed'])
            bad=copy.deepcopy(action);bad['symmetric_data']=bad['symmetric_data']*1.01
            with self.assertRaises(ArithmeticError):modal_certificate(data,labels,frame,bad,N,0)

    def test_original_parent_integrals_and_nonzero_bridges_not_repaired(self):
        side=3;ratio=2;mass=np.arange(1,37,dtype=float)/1024;values=np.arange(36,dtype=float)/64;parent=group_sum(mass,side,ratio)
        observed=observe_path(values[None,:],mass,parent,side,ratio)[0];expected=[]
        for y in range(side):
            for x in range(side):
                ids=[(2*y+dy)*6+2*x+dx for dy in (0,1) for dx in (0,1)]
                expected.append(float(sum((F(float(mass[i]))*F(float(values[i])) for i in ids),F(0))/F(float(parent[y*side+x]))))
        np.testing.assert_array_equal(observed,expected)
        modified=parent*2;raw=observe_path(np.ones((1,36)),mass,modified,side,ratio)[0];np.testing.assert_array_equal(raw,np.full(9,.5))
        args=fixture();args[0][6]['c0'][0]=math.nextafter(args[0][6]['c0'][0],math.inf);before=args[0][6]['c0'].tobytes()
        result,common,b=compare(*args,base=3)
        self.assertGreater(result['per_grid_controls']['6']['initial_two_mode_bridge_absolute'],0)
        self.assertEqual(before,args[0][6]['c0'].tobytes());self.assertFalse(result['original_parent_capacity_controls'][1]['masses_reweighted'])

    def test_fixed_positive_spatial_budgets_raw_paths_and_adverse_sequence_rejection(self):
        args=fixture();data,labels,worlds,runs,paths,res=args
        values=[v for d in data.values() for v in d.values()]+[v for p in paths.values() for v in p.values()];before=[a.tobytes() for a in values]
        for a in values:a.setflags(write=False)
        r,common,b=compare(*args,base=3)
        self.assertTrue(r['all_grid_symbol_reference_and_time_controls_passed']);self.assertTrue(r['all_native_common_and_H_spatial_gains_resolved'])
        self.assertTrue(r['fine_grid_direction_response_resolved']);self.assertTrue(r['registered_three_space_level_affine_shear_control_passed'])
        for row in r['adjacent_spatial_comparisons']:
            for name,field,den in [('native','native_indicator_components_absolute',r['common_initial_fluctuation_norm']),('common','common_indicator_components_absolute',r['common_initial_fluctuation_norm']),('native_H','native_H_indicator_components',1.),('common_H','common_H_indicator_components',1.)]:
                self.assertEqual(row['fixed_indicators'][name],sum(row[field].values())/den);self.assertGreaterEqual(min(row[field].values()),0)
            self.assertFalse(row['signed_cancellation_or_extrapolated_state_used'])
        self.assertEqual(before,[a.tobytes() for a in values])
        bad={key:{name:value.copy() for name,value in p.items()} for key,p in paths.items()};bad[12,16]['independent_states'][8,0]+=.01
        rejected,_,_=compare(data,labels,worlds,runs,bad,res,base=3)
        self.assertFalse(rejected['all_grid_symbol_reference_and_time_controls_passed']);self.assertFalse(rejected['registered_three_space_level_affine_shear_control_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
