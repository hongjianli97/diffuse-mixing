"""Freeze source/protocol before test or science arrays; stdlib only."""
from pathlib import Path
import ast,hashlib,json,subprocess

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    parent=ROOT/'runs/reconnect-compression-joint-definition'
    out=ROOT/'runs/reconnect-compression-joint-control'
    gate=read(parent/'next-gate-spec.json');assert not gate['already_executed']
    sources=dict(gate['sources'])
    for p in [parent/'next-gate-spec.json',parent/'sha256.json',
              ROOT/'runs/reconnect-compression-density-calibration/integer-winding.npz',
              ROOT/'runs/reconnect-compression-density-calibration/anchor-rings.json',
              ROOT/'runs/reconnect-compression-union64/trajectory-source.npz',
              ROOT/'runs/reconnect-compression-union64/union-definition.npz',
              ROOT/'runs/reconnect-compression-union64/original-parent-reference.npz']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    # Include all research Python dependencies without importing or running them.
    implementation={}
    for p in sorted((ROOT/'research/reconnect').glob('*.py')):
        ast.parse(p.read_text());implementation[str(p.relative_to(ROOT))]=sha(p)
    out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_gate_sha256':sha(parent/'next-gate-spec.json'),'sources':sources,
        'main_cap_seconds':1600,'audit_cap_seconds':400,'software_cap_seconds':120,'RSS_MiB_cap':1024,
        'threads':4,'recorded_checks_before':506,'new_checks':2,'old_checks_rerun':0})
    save(out/'source-gate-spec.json',gate)
    save(out/'implementation-lock.json',{'files':implementation,'fixed_before_test_or_physical_arrays':True,
        'automatic_retry_after_failure':False})
    print(f'Locked {len(implementation)} source files and {len(sources)} scientific inputs before tests.')

if __name__=='__main__':main()
