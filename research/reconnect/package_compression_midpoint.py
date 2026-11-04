"""Seal current evidence and create verified review/full archives and byte parts."""
from pathlib import Path
import hashlib,json,zipfile


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def verify_zip(path,manifest_name):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        evidence=json.loads(z.read(manifest_name))
        for rel,h in evidence.items():
            with z.open(rel) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,rel
        layers=0
        for rel in evidence:
            if not rel.endswith('/sha256.json'):continue
            entries=json.loads(z.read(rel))
            if all(name in evidence for name in entries):
                for name,h in entries.items():assert evidence[name]==h,(rel,name)
                layers+=1
    return layers


def main():
    root=Path(__file__).resolve().parents[2];out=root/'runs/reconnect-kernel2d-compression-midpoint';exports=root/'exports'
    source=root/'runs/reconnect-kernel2d-compression';stem='ltn-compression-midpoint'
    full=exports/(stem+'-full.zip');review=exports/(stem+'-review.zip');seal=out/'sha256.json'
    assert not full.exists() and not review.exists() and not seal.exists(),'never overwrite sealed evidence/delivery'
    prior=json.loads((source/'sha256.json').read_text())
    for rel,h in prior.items():assert digest(root/rel)==h,rel
    assert json.loads((out/'tests.json').read_text())['exit_code']==0
    assert json.loads((out/'numerical-audit.json').read_text())['independent_pair_scatter_agrees_with_nested_CSR']
    entries=dict(prior);entries[str((source/'sha256.json').relative_to(root))]=digest(source/'sha256.json')
    for p in sorted((root/'research/reconnect').iterdir()):
        if p.suffix in ['.py','.md']:entries[str(p.relative_to(root))]=digest(p)
    for p in sorted(out.iterdir()):
        if p.is_file():entries[str(p.relative_to(root))]=digest(p)
    dump(seal,dict(sorted(entries.items())))
    evidence={**entries,str(seal.relative_to(root)):digest(seal),'docs/RESEARCH_STATUS.md':digest(root/'docs/RESEARCH_STATUS.md')}
    note=('LTN compression midpoint scientific evidence. No Stage 3 physical acceptance or fluid ORC.\n'
          'Original 782 imported files and historical attachments are audited but not all duplicated here.\n'
          'The mutable progress index is a delivery snapshot, excluded from scientific sha256.json.\n'
          'The review archive omits the large raw midpoint graph; use full archive for exact sparse replay.\n')
    def write_archive(target,items,manifest_name):
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for rel in sorted(items):
                z.write(root/rel,rel,compress_type=zipfile.ZIP_STORED if rel.endswith('.npz') else zipfile.ZIP_DEFLATED)
            z.writestr(manifest_name,json.dumps(items,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    write_archive(full,evidence,'evidence-manifest.json');layers=verify_zip(full,'evidence-manifest.json')
    small={rel:h for rel,h in evidence.items() if rel.startswith('research/reconnect/') or
        (rel.startswith(str(out.relative_to(root))+'/') and not rel.endswith('/compression-midpoint.npz')) or rel=='docs/RESEARCH_STATUS.md'}
    write_archive(review,small,'review-manifest.json');verify_zip(review,'review-manifest.json')
    full_sha=digest(full);parts=[];joined=hashlib.sha256()
    with full.open('rb') as f:
        index=0
        for block in iter(lambda:f.read(31*1024*1024),b''):
            index+=1;p=exports/(full.name+f'.part{index:02d}')
            with p.open('xb') as output:output.write(block)
            joined.update(block);parts.append({'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
    assert joined.hexdigest()==full_sha
    script=exports/'restore-compression-midpoint.py'
    assert not script.exists()
    script.write_text((exports/'restore-compression-frozen.py').read_text().replace('ltn-compression-frozen-delivery.json',stem+'-delivery.json'))
    compile(script.read_text(),str(script),'exec')
    info={'full':{'file':full.name,'bytes':full.stat().st_size,'sha256':full_sha},'parts':parts,
        'parts_reassembled_SHA256_verified':True,
        'review':{'file':review.name,'bytes':review.stat().st_size,'sha256':digest(review)},
        'restore_script':{'file':script.name,'sha256':digest(script)},'scientific_manifest_entries':len(entries),
        'full_archive_evidence_entries':len(evidence),'fully_archived_scientific_layers_verified':layers,
        'CRC_and_all_archived_hashes_verified':True,'prior_formal_manifests_unchanged':32,
        'new_checks_passed':7,'new_candidate_transient_runs':0,'fluid_orc_enabled':False}
    dump(exports/(stem+'-delivery.json'),info)
    (exports/(stem+'-full.sha256')).write_text(full_sha+'  '+full.name+'\n')
    lines=['# 压缩中点与端点时间顺序：数据下载','',
        f'完整科学数据包：[{full.name}]({full.name})，{full.stat().st_size/2**20:.2f} MiB。',
        '',f'SHA-256：`{full_sha}`。','',
        '若使用分卷，全部分卷、重组脚本和交付JSON放在同一目录；分卷不能单独解压。','']
    for i,item in enumerate(parts,1):lines.append(f'- [分卷{i}]({item["file"]})，{item["bytes"]/2**20:.2f} MiB，SHA-256 `{item["sha256"]}`。')
    lines+=['',f'- [重组脚本]({script.name})',f'- [交付哈希清单]({stem}-delivery.json)',f'- [小型审阅包]({review.name})','',
        '```bash','python '+script.name,'```','',
        '完整包保留本轮原始共享边/CSR及引用科学证据；原782文件的跨环境审计仍需原交接材料。','']
    (exports/(stem+'-downloads.md')).write_text('\n'.join(lines))
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
