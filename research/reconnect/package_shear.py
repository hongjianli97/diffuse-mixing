"""Current shear case archives; reuse the sealed historical base by hash."""
from pathlib import Path
import hashlib,json,zipfile
from package_compression_midpoint import digest,dump,verify_zip


def main():
    root=Path(__file__).resolve().parents[2];out=root/'runs/reconnect-shear-frozen';exports=root/'exports'
    source=root/'runs/reconnect-compression-spatial';stem='ltn-shear-frozen'
    full=exports/(stem+'-full.zip');review=exports/(stem+'-review.zip');seal=out/'sha256.json'
    assert not seal.exists() and not full.exists() and not review.exists(),'never overwrite sealed evidence'
    assert json.loads((out/'tests.json').read_text())['exit_code']==0
    assert json.loads((out/'numerical-audit.json').read_text())['raw_graph_action_area_weights_and_all_reported_norms_recomputed']
    prior=json.loads((source/'sha256.json').read_text())
    for rel,h in prior.items():assert digest(root/rel)==h,rel
    base=json.loads((exports/'ltn-compression-spatial-delivery.json').read_text())['full']
    base_file=exports/base['file'];assert base_file.stat().st_size==base['bytes'] and digest(base_file)==base['sha256']
    basis={'archive_scope':'complete current shear experiment and source code; historical scientific closure supplied by the unchanged base archive',
        'historical_base':base,'historical_base_download_index':'ltn-compression-spatial-downloads.md',
        'source_scientific_manifest':str((source/'sha256.json').relative_to(root)),
        'source_scientific_manifest_sha256':digest(source/'sha256.json'),
        'historical_base_archive_SHA256_verified':True,'original_782_files_and_attachments_for_full_audit':'still require original handoff materials'}
    dump(out/'delivery-basis.json',basis)
    entries=dict(prior);entries[str((source/'sha256.json').relative_to(root))]=digest(source/'sha256.json')
    for p in sorted((root/'research/reconnect').iterdir()):
        if p.suffix in ['.py','.md']:entries[str(p.relative_to(root))]=digest(p)
    for p in sorted(out.iterdir()):
        if p.is_file():entries[str(p.relative_to(root))]=digest(p)
    dump(seal,dict(sorted(entries.items())))
    state_source=json.loads((out/'config.json').read_text())['state_source']
    selected={rel for rel in entries if rel.startswith('research/reconnect/') or rel.startswith(str(out.relative_to(root))+'/')}
    selected.update([str(seal.relative_to(root)),str((source/'sha256.json').relative_to(root)),
        str((source/'stage3-status.json').relative_to(root)),str(Path(state_source).relative_to(root)),'docs/RESEARCH_STATUS.md'])
    evidence={rel:digest(root/rel) for rel in sorted(selected)}
    note=('Complete current shear experiment: raw shared edges/CSR, point inputs, saved state source, references, code and audit.\n'
          'Historical scientific files are in the separately hashed ltn-compression-spatial base archive.\n'
          'Extract this archive over that base for complete source verification and CLI reproduction.\n'
          'The current scientific sha256.json covers the combined history, not just this incremental archive.\n'
          'Original 782 imported files and historical attachments are audited but not all duplicated.\n'
          'docs/RESEARCH_STATUS.md is a mutable delivery snapshot outside the scientific manifest.\n'
          'No Stage 3 physical acceptance or fluid ORC.\n')
    def write_archive(path,items,manifest):
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for rel in sorted(items):z.write(root/rel,rel,compress_type=zipfile.ZIP_STORED if rel.endswith('.npz') else zipfile.ZIP_DEFLATED)
            z.writestr(manifest,json.dumps(items,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
            z.writestr('historical-base-dependency.json',json.dumps(basis,indent=2)+'\n')
    write_archive(full,evidence,'evidence-manifest.json');verify_zip(full,'evidence-manifest.json')
    small={rel:h for rel,h in evidence.items() if rel.startswith('research/reconnect/') or
        (rel.startswith(str(out.relative_to(root))+'/') and not rel.endswith('/shear-frozen.npz')) or rel=='docs/RESEARCH_STATUS.md'}
    write_archive(review,small,'review-manifest.json');verify_zip(review,'review-manifest.json')
    parts=[];joined=hashlib.sha256();full_sha=digest(full)
    with full.open('rb') as f:
        for index,block in enumerate(iter(lambda:f.read(31*1024*1024),b''),1):
            path=exports/(full.name+f'.part{index:02d}')
            with path.open('xb') as part:part.write(block)
            joined.update(block);parts.append({'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)})
    assert joined.hexdigest()==full_sha
    script=exports/'restore-shear-frozen.py';assert not script.exists()
    script.write_text((exports/'restore-compression-frozen.py').read_text().replace('ltn-compression-frozen-delivery.json',stem+'-delivery.json'))
    compile(script.read_text(),str(script),'exec')
    info={'full':{'file':full.name,'bytes':full.stat().st_size,'sha256':full_sha},'full_archive_is_incremental':True,
        'current_experiment_complete':True,'historical_base':base,'historical_base_archive_SHA256_verified':True,
        'parts':parts,'parts_reassembled_SHA256_verified':True,
        'review':{'file':review.name,'bytes':review.stat().st_size,'sha256':digest(review)},
        'restore_script':{'file':script.name,'sha256':digest(script)},'scientific_manifest_entries':len(entries),
        'current_archive_evidence_entries':len(evidence),'CRC_and_all_current_archived_hashes_verified':True,
        'prior_formal_manifests_unchanged':36,'new_checks_passed':7,'new_transient_runs':0,'fluid_orc_enabled':False}
    dump(exports/(stem+'-delivery.json'),info)
    (exports/(stem+'-full.sha256')).write_text(full_sha+'  '+full.name+'\n')
    lines=['# 二维周期剪切冻结检验：本轮数据下载','',
        f'本轮完整算例包：[{full.name}]({full.name})，{full.stat().st_size/2**20:.2f} MiB。包含本轮原始共享边/CSR及全部参考。',
        '',f'SHA-256：`{full_sha}`。','',
        '本轮采用增量交付；完整历史科学链保持在已封存的[历史基包及分卷](ltn-compression-spatial-downloads.md)。完整审计/CLI复现需先解压历史基包，再叠加本轮包；原782文件审计还需原交接材料。',
        '',f'历史基包SHA-256：`{base["sha256"]}`，已重新核验。','',
        '使用本轮分卷时，将全部分卷、重组脚本与交付JSON放在同一目录；先重组再解压。','']
    for index,item in enumerate(parts,1):lines.append(f'- [本轮分卷{index}]({item["file"]})，{item["bytes"]/2**20:.2f} MiB，SHA-256 `{item["sha256"]}`。')
    lines+=['',f'- [重组脚本]({script.name})',f'- [交付哈希清单]({stem}-delivery.json)',f'- [小型审阅包]({review.name})','',
        '```bash','python '+script.name,'```','']
    (exports/(stem+'-downloads.md')).write_text('\n'.join(lines))
    print(json.dumps({key:info[key] for key in ['full','review','full_archive_is_incremental','historical_base_archive_SHA256_verified','scientific_manifest_entries','prior_formal_manifests_unchanged','new_checks_passed']},indent=2),flush=True)
    print('Current byte parts:',len(parts),flush=True)


if __name__=='__main__':main()
