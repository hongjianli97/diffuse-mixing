"""Verify inherited immutable evidence without editing prior runs."""
from pathlib import Path
import argparse,hashlib,json


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(output):
    root=Path(__file__).resolve().parents[2];output=Path(output).resolve();source=root/'runs/rebuild-stage3-hybrid-face'
    old=json.loads((source/'preservation.json').read_text());originals=json.loads(Path('/workspace/original-handoff-sha256.json').read_text())
    for rel,h in originals.items():assert digest(root/rel)==h,rel
    attachments=json.loads(Path('/workspace/setup-ltn/attachment-manifest.json').read_text())
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    formal=old['formal_scientific_manifests_verified']+[{'manifest':str((source/'sha256.json').relative_to(root)),
        'manifest_sha256':digest(source/'sha256.json'),'entries_verified':len(json.loads((source/'sha256.json').read_text()))}]
    for item in formal:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(root/rel)==h,(item['manifest'],rel)
    historical=old['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    for item in historical:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries']
        differences=[{'path':rel,'saved':h,'current':digest(root/rel)} for rel,h in entries.items() if digest(root/rel)!=h]
        assert sorted(differences,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    conversations=json.loads((output/'source-conversation-sha256.json').read_text())
    for item in conversations:
        p=Path(item['source_path']);assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
        if 'archive_copy' in item:assert digest(root/item['archive_copy'])==item['sha256']
    result={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),'newly_consulted_material_files_verified':len(conversations),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'new_fluid_transients_or_propagators_computed':False,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (output/'preservation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),'new_material_files_verified':len(conversations),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
