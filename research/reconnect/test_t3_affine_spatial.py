"""Small independent direction, material integral and adverse-control fixtures; not new flow cases."""
from fractions import Fraction as F
import argparse,copy,json,math,unittest
import numpy as np
from affine_compression_inputs import MODELS,COUNTS,pure_inputs,directional_certificate
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from voronoi_time_action import operators
from t3_affine_spatial_core import modal_certificate,compare
from t1_uniform_core import pure_labels
from material_union import observe_path,group_sum


def fixture():
    data={};worlds={};runs={};paths={};res={}
    for N in (3,6,12):
        labels=pure_labels(N);m=np.full(N*N,1/(N*N));c0=.5+.2*(math.sin(math.pi/N)/(math.pi/N))**2*np.sin(2*math.pi*labels[:,0])*np.cos(2*math.pi*labels[:,1])
        data[N]={'mass':m,'c0':c0};worlds[N]=pure_inputs(labels,np.arange(N*N));qh=4*N*N*math.sin(math.pi/N)**2
        for model in MODELS:
            for K in COUNTS:
                clock=F(0);states=[c0.copy()]
                for step in range(K):
                    f=worlds[N]['models'][model][str(K)]['frames'][2*step+1];a=F(f['a_exact']);clock+=F(1,20*K)*F(f['Gamma_actual_exact'])*(a+1/a)
                    states.append(.5+(c0-.5)*math.exp(-qh*float(clock)))
                states=np.array(states);paths[N,model,K]={'states':states,'independent_states':states.copy()}
                runs[N,model,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(K)]};res[N,model,K]=[0.]*K
    return data,worlds,runs,paths,res


class Checks(unittest.TestCase):
    def test_multi_N_directional_geometry_and_modal_failure_guard(self):
        for N in (4,8):
            labels=pure_labels(N);ids=np.arange(N*N);world=pure_inputs(labels,ids);m=np.full(N*N,1/(N*N))
            c0=.5+.2*(math.sin(math.pi/N)/(math.pi/N))**2*np.sin(2*math.pi*labels[:,0])*np.cos(2*math.pi*labels[:,1])
            for model in MODELS:
                frame=world['models'][model]['4']['frames'][3]
                packet,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],OrientedPeriodicRectangle.from_packet(frame['boundary']))
                self.assertTrue(directional_certificate(packet,frame,ids,N)['all_directional_conductances_exact'])
                op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']));A=op.symmetric_generator
                action={'symmetric_data':A.data,'symmetric_indices':A.indices,'symmetric_indptr':A.indptr,'shape':A.shape}
                self.assertTrue(modal_certificate({'mass':m,'c0':c0},frame,action,N,0)['original_algebra_scale_passed'])
                bad=copy.deepcopy(action);bad['symmetric_data']=bad['symmetric_data']*1.01
                with self.assertRaises(ArithmeticError):modal_certificate({'mass':m,'c0':c0},frame,bad,N,0)

    def test_direct_fixed_parent_integrals_and_capacity_bridge_not_repaired(self):
        side=3;ratio=2;mass=np.arange(1,37,dtype=float)/1024;values=np.arange(36,dtype=float)/64
        parent=group_sum(mass,side,ratio);observed=observe_path(values[None,:],mass,parent,side,ratio)[0]
        expected=[]
        for y in range(side):
            for x in range(side):
                ids=[(2*y+dy)*6+2*x+dx for dy in (0,1) for dx in (0,1)]
                exact=sum((F(float(mass[i]))*F(float(values[i])) for i in ids),F(0))/F(float(parent[y*side+x]))
                expected.append(float(exact))
        np.testing.assert_array_equal(observed,expected)
        modified=parent*2;raw=observe_path(np.ones((1,36)),mass,modified,side,ratio)[0]
        np.testing.assert_array_equal(raw,np.full(9,.5));np.testing.assert_array_equal(modified,parent*2)

    def test_spatial_budget_uses_raw_inputs_and_rejects_bad_full_sequence(self):
        args=fixture();data,worlds,runs,paths,res=args
        snapshot=[v.tobytes() for d in data.values() for v in d.values()]+[v.tobytes() for p in paths.values() for v in p.values()]
        for d in data.values():
            for value in d.values():value.setflags(write=False)
        for p in paths.values():
            for value in p.values():value.setflags(write=False)
        result,common,bundles=compare(*args,base=3)
        self.assertTrue(result['original_parent_capacities_preserved']);self.assertTrue(result['all_grid_symbol_reference_and_time_controls_passed'])
        self.assertFalse(result['fine_grid_model_response_resolved']);self.assertFalse(result['T3_registered_three_space_level_affine_control_passed'])
        self.assertTrue(result['all_native_common_and_H_spatial_gains_resolved'])
        for row in result['adjacent_spatial_comparisons']:
            self.assertEqual(row['fixed_indicators']['common'],sum(row['common_indicator_components_absolute'].values())/result['common_initial_fluctuation_norm'])
            self.assertFalse(row['signed_cancellation_or_extrapolated_state_used'])
        self.assertEqual(snapshot,[v.tobytes() for d in data.values() for v in d.values()]+[v.tobytes() for p in paths.values() for v in p.values()])
        bad={key:{name:value.copy() for name,value in p.items()} for key,p in paths.items()};bad[12,'constant_D',16]['independent_states'][8,0]+=.01
        rejected,_,_=compare(data,worlds,runs,bad,res,base=3)
        self.assertFalse(rejected['all_grid_symbol_reference_and_time_controls_passed']);self.assertFalse(rejected['T3_registered_three_space_level_affine_control_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
