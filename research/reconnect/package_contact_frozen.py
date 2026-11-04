"""Current stationary contact frozen probe; retain five hashed historical bases."""
from pathlib import Path
import hashlib,json,zipfile
from package_compression_midpoint import digest,dump,verify_zip


def main():
    root=Path(__file__).resolve().parents[2];out=root/'runs/reconnect-contact-frozen';exports=root/'exports'
    source=root/'runs/reconnect-shear-transient-spatial';stem='ltn-contact-frozen'
    full=exports/(stem+'-full.zip');review=exports/(stem+'-review.zip');seal=out/'sha256.json'
    assert not seal.exists() and not full.exists() and not review.exists(),'never overwrite sealed evidence'
    assert json.loads((out/'tests.json').read_text())['exit_code']==0
    assert json.loads((out/'numerical-audit.json').read_text())['all_region_errors_and_contributions_recomputed']
    prior=json.loads((source/'sha256.json').read_text())
    for rel,h in prior.items():assert digest(root/rel)==h,rel
    bases=[]
    for previous in ['ltn-compression-spatial','ltn-shear-spatial','ltn-shear-reference','ltn-ordered-shear','ltn-shear-transient-spatial']:
        item=json.loads((exports/(previous+'-delivery.json')).read_text())['full']
        path=exports/item['file'];assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
        bases.append({**item,'download_index':previous+'-downloads.md'})
    basis={'archive_scope':'complete single stationary density-contact raw graph, all material-average inputs and independent frozen references; historical scientific closure uses five bases',
        'historical_bases':bases,'historical_base_SHA256_all_verified':True,
        'source_scientific_manifest':str((source/'sha256.json').relative_to(root)),
        'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'original_782_files_and_attachments_for_full_audit':'original handoff materials still required'}
    dump(out/'delivery-basis.json',basis)
    entries=dict(prior);entries[str((source/'sha256.json').relative_to(root))]=digest(source/'sha256.json')
    for p in sorted((root/'research/reconnect').iterdir()):
        if p.suffix in ['.py','.md']:entries[str(p.relative_to(root))]=digest(p)
    for p in sorted(out.iterdir()):
        if p.is_file():entries[str(p.relative_to(root))]=digest(p)
    dump(seal,dict(sorted(entries.items())))
    selected={rel for rel in entries if rel.startswith('research/reconnect/') or rel.startswith(str(out.relative_to(root))+'/')}
    selected.update([str(seal.relative_to(root)),str((source/'sha256.json').relative_to(root)),
        str((source/'stage3-status.json').relative_to(root)),'docs/RESEARCH_STATUS.md'])
    evidence={rel:digest(root/rel) for rel in sorted(selected)}
    note=('Complete current stationary density-contact frozen probe and raw shared edges/conductances/CSR. No transient admission.\n'
          'Extract compression-spatial, shear-spatial, shear-reference, ordered-shear and shear-transient-spatial historical archives in that order, then this archive.\n'
          'The scientific manifest covers combined history; this current archive is incremental.\n'
          'Original 782 imported files/attachments remain separately required for full provenance audit.\n'
          'docs/RESEARCH_STATUS.md is a mutable snapshot outside the scientific manifest. Stage 3 incomplete. No fluid ORC.\n')
    def write_archive(path,items,manifest):
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for rel in sorted(items):z.write(root/rel,rel,compress_type=zipfile.ZIP_STORED if rel.endswith('.npz') else zipfile.ZIP_DEFLATED)
            z.writestr(manifest,json.dumps(items,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
            z.writestr('historical-base-dependencies.json',json.dumps(basis,indent=2)+'\n')
    write_archive(full,evidence,'evidence-manifest.json');verify_zip(full,'evidence-manifest.json')
    small={rel:h for rel,h in evidence.items() if rel.startswith('research/reconnect/') or
        (rel.startswith(str(out.relative_to(root))+'/') and not rel.endswith('/contact-frozen.npz')) or rel=='docs/RESEARCH_STATUS.md'}
    write_archive(review,small,'review-manifest.json');verify_zip(review,'review-manifest.json')
    parts=[];joined=hashlib.sha256();full_sha=digest(full)
    with full.open('rb') as f:
        for index,block in enumerate(iter(lambda:f.read(31*1024*1024),b''),1):
            path=exports/(full.name+f'.part{index:02d}')
            with path.open('xb') as part:part.write(block)
            joined.update(block);parts.append({'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)})
    assert joined.hexdigest()==full_sha
    script=exports/'restore-contact-frozen.py';assert not script.exists()
    script.write_text((exports/'restore-compression-frozen.py').read_text().replace('ltn-compression-frozen-delivery.json',stem+'-delivery.json'))
    compile(script.read_text(),str(script),'exec')
    info={'full':{'file':full.name,'bytes':full.stat().st_size,'sha256':full_sha},'full_archive_is_incremental':True,
        'current_contact_frozen_probe_complete':True,'historical_bases':bases,'historical_base_SHA256_all_verified':True,
        'parts':parts,'parts_reassembled_SHA256_verified':True,
        'review':{'file':review.name,'bytes':review.stat().st_size,'sha256':digest(review)},
        'restore_script':{'file':script.name,'sha256':digest(script)},'scientific_manifest_entries':len(entries),
        'current_archive_evidence_entries':len(evidence),'CRC_and_all_current_archived_hashes_verified':True,
        'prior_formal_manifests_unchanged':41,'reused_checks_passed':0,'new_checks_passed':6,'recorded_checks_total':336,'new_candidate_graphs':1,'new_candidate_transient_runs':0,'physical_cases':1,'fluid_orc_enabled':False}
    dump(exports/(stem+'-delivery.json'),info)
    (exports/(stem+'-full.sha256')).write_text(full_sha+'  '+full.name+'\n')
    lines=['# 二维静止密度接触冻结图：本轮数据下载','',
        f'本轮完整冻结图包：[{full.name}]({full.name})，{full.stat().st_size/2**20:.2f} MiB。含单张原始共享边/电导/CSR、完整物质平均初态、解析面通量、Gauss8/16体积分和全部误差观测。',
        '',f'SHA-256：`{full_sha}`。','',
        '本轮采用增量交付。完整历史审计/CLI复现依次解压以下五份历史基包，再叠加本轮；原782文件审计还需原交接材料。','']
    for index,item in enumerate(bases,1):lines.append(f'- [历史基包{index}及分卷]({item["download_index"]})，SHA-256 `{item["sha256"]}`，已重新核验。')
    lines+=['','本轮分卷、重组脚本和交付JSON放在同一目录；先重组再解压。','']
    for index,item in enumerate(parts,1):lines.append(f'- [本轮分卷{index}]({item["file"]})，{item["bytes"]/2**20:.2f} MiB，SHA-256 `{item["sha256"]}`。')
    lines+=['',f'- [重组脚本]({script.name})',f'- [交付哈希清单]({stem}-delivery.json)',f'- [小型审阅包]({review.name})','',
        '```bash','python '+script.name,'```','']
    (exports/(stem+'-downloads.md')).write_text('\n'.join(lines))
    print(json.dumps({key:info[key] for key in ['full','review','full_archive_is_incremental','historical_base_SHA256_all_verified','scientific_manifest_entries','prior_formal_manifests_unchanged','new_checks_passed']},indent=2),flush=True)
    print('Current byte parts:',len(parts),flush=True)


if __name__=='__main__':main()
