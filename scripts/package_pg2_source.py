"""Allowlist only necessary Part 1/2 code and configs; no local reference material."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    files=list((ROOT/'policy_gradient_part1').glob('*.py'))+list((ROOT/'policy_gradient_part2').glob('*.py'))
    files += [ROOT/'scripts'/name for name in ['train_pg2.py','evaluate_pg.py','validate_pg2.py',
        'verify_pg_resume.py','verify_source_snapshot.py','diagnose_pg2.py','pong_part2_pipeline.py','launch_pong_part2.py']]
    files += [ROOT/'configs'/name for name in ['cartpole_part2.json','pong_part2.json']]
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    output=ROOT/'tmp/pg2_source.tar.gz';output.parent.mkdir(exist_ok=True)
    with tarfile.open(output,'w:gz') as archive:
        for f in files: archive.add(f,arcname=f.relative_to(ROOT).as_posix())
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('source_manifest.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    (ROOT/'results/policy_gradient_part2/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Packaged {len(files)} files: {output}')
