"""One four-field density experiment, with an audited seal before oracle loading."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gc, hashlib, resource, signal, time, traceback
import numpy as np
from audit_route_decision import ROOT, read, save, digest
from compression_density_sources import periodic_winding, anchor_rings, kernel_density, ledger_density
from density_snapshot_metrics import moments, evaluate
from audit_density_snapshot import preseal_audit

PARENT = ROOT/'runs/reconnect-density-snapshot-definition'
OLD = ROOT/'runs/reconnect-compression-density-calibration'
PROTOCOL = ROOT/'research/reconnect/density_snapshot_protocol.json'


def array_record(a):
    return {'shape': list(a.shape), 'dtype': str(a.dtype),
            'sha256_C_bytes': hashlib.sha256(a.tobytes()).hexdigest()}


def load(path, keys, phase, role, access, seal=None):
    allowed = {'trajectory': {'ids', 'labels', 'rho0_point'}, 'mass': {'mass'},
        'builder': {'ids', 'k32_time', 'k32_positions'}, 'oracle': {'k32_rho'}}
    assert role in allowed and set(keys) <= allowed[role]
    if role == 'oracle':
        assert phase == 'postseal_scoring' and seal is not None and seal.is_file()
        assert read(seal)['all_six_fields_independently_verified_before_oracle']
    else: assert phase in ('production', 'postseal_scoring')
    with np.load(path, allow_pickle=False) as z: data = {k: z[k] for k in keys}
    for a in data.values(): a.setflags(write=False)
    access.append({'file': str(path.relative_to(ROOT)), 'keys': list(keys), 'phase': phase,
        'role': role, 'density_seal_sha256': digest(seal) if seal else None})
    return data


def preserve():
    old = read(PARENT/'preservation.json'); p = PARENT/'sha256.json'
    formal = old['formal_manifests_verified'] + [{'manifest': str(p.relative_to(ROOT)),
        'manifest_sha256': digest(p), 'entries_verified': len(read(p))}]
    assert len(formal) == 124
    for item in formal:
        p = ROOT/item['manifest']; assert digest(p) == item['manifest_sha256']
        entries = read(p); assert len(entries) == item['entries_verified']
        for name, h in entries.items(): assert digest(ROOT/name) == h, name
    originals = read('/workspace/original-handoff-sha256.json')
    attachments = read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals) == 782 and len(attachments) == 15
    for name, h in originals.items(): assert digest(ROOT/name) == h, name
    for item in attachments:
        p = Path('/workspace')/item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p = ROOT/item['manifest']; assert digest(p) == item['manifest_sha256']
        diff = [{'path': k, 'saved': v, 'current': digest(ROOT/k)}
                for k, v in read(p).items() if digest(ROOT/k) != v]
        assert sorted(diff, key=lambda x: x['path']) == sorted(
            item['historical_paths_differing_from_later_formal_versions'], key=lambda x: x['path'])
    return {'formal_manifests_verified': formal, 'originals_verified': 782, 'attachments_verified': 15,
        'historical_diagnostic_differences_retained': old['historical_diagnostic_differences_retained'],
        'all_prior_scientific_files_unchanged': True}


def main(directory):
    out = Path(directory).resolve(); out.mkdir(exist_ok=False); start = time.monotonic()
    access = []; index = {}; new_fields = 0
    def guard(stage):
        elapsed = time.monotonic()-start; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed < 600 and rss < 1024, (stage, elapsed, rss)
        print(f'{stage}: {elapsed:.2f}s, peak{rss:.1f}MiB', flush=True)
    def alarm(*_): raise TimeoutError('frozen600second four-field main cap')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(600)
    try:
        p = read(PROTOCOL); gate = read(PARENT/'next-gate-spec.json')
        assert gate['type'] == 'single_saved_snapshot_four_density_fields'
        assert gate['new_density_fields'] == 4 and gate['new_N'] == [128,256]
        assert not gate['already_executed'] and p['metrics'] == gate['metrics']
        assert digest(PARENT/'next-gate-spec.json') == p['source_gate_sha256']
        for name, h in p['sources'].items(): assert digest(ROOT/name) == h, name
        save(out/'protocol.json', p); save(out/'source-gate-spec.json', gate)
        save(out/'implementation-lock.json', {'files': {name: digest(ROOT/name)
            for name in p['implementation_files']}, 'locked_before_fields': True})
        save(out/'preservation.json', preserve()); guard('prior files and new implementation sealed')
        inventory = read(PARENT/'input-inventory.json')
        for c, row in zip(p['cases'], inventory['cases']):
            N = c['N']; folder = out/f'n{N}'; folder.mkdir()
            pure = load(ROOT/c['trajectory'], ('ids','labels','rho0_point'), 'production', 'trajectory', access)
            m = load(ROOT/c['mass_file'], ('mass',), 'production', 'mass', access)['mass']
            b = load(ROOT/c['builder'], ('ids','k32_time','k32_positions'), 'production', 'builder', access)
            ids, labels, rho0 = pure['ids'], pure['labels'], pure['rho0_point']
            t, x = b['k32_time'], b['k32_positions']
            assert np.array_equal(ids,b['ids']) and np.array_equal(ids,np.arange(N*N))
            assert np.array_equal(t,np.linspace(0,.05,65)) and np.array_equal(x[0],labels)
            assert t[63].hex() == '0x1.9333333333334p-5'
            for key, a in [('ids',ids),('labels',labels),('rho0',rho0),('mass',m),
                           ('time',t),('snapshot_positions',x[63])]:
                assert array_record(a) == row['arrays'][key], (N,key)
            if N == 64:
                index[str(N)] = {'N':N, 'winding_file':str((OLD/'integer-winding.npz').relative_to(ROOT)),
                    'winding_key':'k32', 'anchor_file':str((OLD/'anchor-rings.json').relative_to(ROOT)),
                    'area_file':str((OLD/'ledger/k32/areas-step31.json').relative_to(ROOT)),
                    'fields':{s:{'file':inventory['N64_reused_fields'][s]['file'],
                        'sha256':inventory['N64_reused_fields'][s]['sha256'], 'new_production_field':False}
                        for s in ('kernel','ledger')}}
            else:
                guard(f'N{N} explicit winding begin')
                w = periodic_winding(ids,t,x,np.linspace(0,.05,65))
                np.savez_compressed(folder/'integer-winding.npz',k32=w)
                graph = read(ROOT/c['geometry'])
                rings,a0 = anchor_rings(ids,labels,graph['edges']); del graph
                assert all(a == F(2,N*N) for a in a0)
                save(folder/'anchor-rings.json', {'rings':rings, 'initial_areas_exact':list(map(str,a0)),
                    'images_fixed_from_t0_labels':True, 'physical_capacity_replaced':False})
                guard(f'N{N} fixed anchors ready')
                u, support = kernel_density(ids,x[63],ids,m,2/N)
                np.savez_compressed(folder/'kernel.npz',rho=u); new_fields += 1
                v, areas = ledger_density(x[63],w[63],rho0,rings,a0)
                np.savez_compressed(folder/'ledger.npz',rho=v); new_fields += 1
                save(folder/'areas.json',{'current_areas_exact':list(map(str,areas))})
                index[str(N)] = {'N':N, 'winding_file':str((folder/'integer-winding.npz').relative_to(ROOT)),
                    'winding_key':'k32', 'anchor_file':str((folder/'anchor-rings.json').relative_to(ROOT)),
                    'area_file':str((folder/'areas.json').relative_to(ROOT)), 'kernel_support':support,
                    'fields':{s:{'file':str((folder/f'{s}.npz').relative_to(ROOT)),
                        'sha256':digest(folder/f'{s}.npz'), 'new_production_field':True} for s in ('kernel','ledger')}}
                del w,rings,a0,u,v,areas
            guard(f'N{N} production inputs/fields ready')
            del pure,m,b,ids,labels,rho0,t,x
            gc.collect()
        assert new_fields == 4
        save(out/'density-index.json',index)
        preseal_audit(out,p,index,guard)
        guard('all six density fields independently checked')
        sealed = {}
        for item in index.values():
            for key in ('winding_file','anchor_file','area_file'):
                sealed[item[key]] = digest(ROOT/item[key])
            for f in item['fields'].values(): sealed[f['file']] = digest(ROOT/f['file'])
        for f in out.rglob('*'):
            if f.is_file() and (f.suffix == '.npz' or f.name in ('density-index.json','density-audit.json')):
                sealed[str(f.relative_to(ROOT))] = digest(f)
        save(out/'density-seal.json',{'files':sealed, 'new_production_fields':4, 'N64_fields_reused':2,
            'independent_kernel_evaluations':3, 'independent_exact_ledger_checks':3,
            'all_six_fields_independently_verified_before_oracle':True,
            'future_oracle_or_scalar_received_by_producers':False})
        guard('density seal complete; oracle scoring begins')
        exact = []
        for c,row in zip(p['cases'],inventory['cases']):
            N = c['N']
            m = load(ROOT/c['mass_file'],('mass',),'postseal_scoring','mass',access)['mass']
            o = load(ROOT/c['builder'],('k32_rho',),'postseal_scoring','oracle',access,
                     out/'density-seal.json')['k32_rho'][63]
            assert array_record(o) == row['arrays']['snapshot_oracle']
            for source in ('kernel','ledger'):
                with np.load(ROOT/index[str(N)]['fields'][source]['file'],allow_pickle=False) as z: u = z['rho']
                with np.load(out/f'n{N}/{source}-independent.npz',allow_pickle=False) as z: a = z['rho']
                exact.append({'N':N,'source':source,'moments':moments(m,o,u,a)})
            guard(f'N{N} exact scoring complete')
        save(out/'exact-moments.json',exact)
        result80,result100 = evaluate(exact,80),evaluate(exact,100)
        assert result80 == result100, 'precision comparison unresolved; stop without refinement'
        save(out/'precision-audit.json',{'digits':[80,100],'reported_significant_digits':65,
            'all_rows_and_decisions_identical':True})
        result80.update(new_density_fields=4,N64_fields_reused=2,new_geometry=0,new_coefficients=0,
            new_scalar_paths=0,new_reference_PDE=0,new_software_tests=0,recorded_checks_total=508,
            density_seal_preceded_oracle_scoring=True,old_budgets_and_decisions_unchanged=True,
            stage3='incomplete',fluid_orc_enabled=False,kappa_computed=False)
        save(out/'results.json',result80); save(out/'input-access.json',access)
        for name,h in p['sources'].items(): assert digest(ROOT/name) == h,name
        guard('fixed three-grid snapshot comparison complete')
        save(out/'execution.json',{'completed':True,'elapsed_seconds':time.monotonic()-start,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'new_density_fields':4,'N64_fields_reused':2,'new_scalar_paths':0,'new_geometry':0,
            'new_coefficients':0,'new_reference_PDE':0,'new_software_tests':0,'old_tests_rerun':0,
            'published_or_pushed':False})
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,
            'new_fields_written':new_fields,'partial_data_preserved':True,'automatic_retry':False})
        if not (out/'input-access.json').exists(): save(out/'input-access.json',access)
        raise
    finally: signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
