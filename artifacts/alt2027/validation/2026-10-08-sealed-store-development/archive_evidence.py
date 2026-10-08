"""Copy selected evidence losslessly, preserve Run bytes, and verify every member.

Run from this repository root. Outputs are confined to this script's directory.
The original runs, scripts, stores, and their manifests are never modified.
"""
from pathlib import Path
import datetime,gzip,hashlib,io,json,subprocess,tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
WORKSPACE=REPO.parent.parent
INV=WORKSPACE/'work/fast-engine-investigation'
files=[]

def digest(data):return hashlib.sha256(data).hexdigest()
def relative(path):
 try:return str(path.relative_to(WORKSPACE))
 except ValueError:return str(path)
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing archive differs: {path}')
 else:path.write_bytes(data)
def archive_file(source,dest,compress=False):
 source=Path(source);before=source.read_bytes();payload=gzip.compress(before,mtime=0) if compress else before
 target=HERE/dest;write(target,payload)
 if source.read_bytes()!=before:raise RuntimeError(f'source changed while copied: {source}')
 record={'path':str(target.relative_to(HERE)),'bytes':len(payload),'sha256':digest(payload),'source':relative(source),'source_bytes':len(before),'source_sha256':digest(before)}
 if compress:record['encoding']='gzip';assert gzip.decompress(target.read_bytes())==before
 files.append(record)

def run_archive(name):
 source=REPO/'output/alt2027'/name;manifest=json.loads((source/'manifest.json').read_bytes())
 members={str(p.relative_to(source)):p.read_bytes() for p in sorted(source.rglob('*')) if p.is_file()}
 for path,record in manifest['outputs'].items():
  data=members[path]
  if len(data)!=record['bytes'] or digest(data)!=record['sha256']:raise RuntimeError(f'original Run output mismatch: {name}/{path}')
 recovery={'method':'Recovered after execution, only when exact SHA256 matches original Run source inventory. Original Run files are unmodified. This supplement is not evidence of a clean execution tree.','found':{},'unavailable':{}}
 if not (source/'source-capsule').exists():
  capsule=REPO/'output/alt2027/sealed-pilot-validation-20261008-001/source-capsule'
  wanted={p:h for p,h in manifest['source']['files'].items() if p.startswith(('src/','scripts/')) or p in ('pyproject.toml','requirements-alt.lock')}
  for path,want in wanted.items():
   found=None
   for candidate in [REPO/path,capsule/path]:
    if candidate.is_file():
     data=candidate.read_bytes()
     if digest(data)==want:found=(data,relative(candidate));break
   if found is None:
    for commit in ['83224dc756481ab8863855a8def26f1504447b42',manifest['source']['commit']]:
     result=subprocess.run(['git','show',f'{commit}:{path}'],cwd=REPO,capture_output=True)
     if result.returncode==0 and digest(result.stdout)==want:found=(result.stdout,f'git:{commit}:{path}');break
   if found:
    members['recovered-source-capsule/'+path]=found[0]
    recovery['found'][path]={'sha256':want,'recovered_from':found[1]}
   else:recovery['unavailable'][path]={'sha256':want,'reason':'No exact matching source bytes available in checked candidates.'}
  members['source-capsule-recovery.json']=(json.dumps(recovery,sort_keys=True,indent=2)+'\n').encode()
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w',format=tarfile.PAX_FORMAT) as tar:
  for path,data in sorted(members.items()):
   info=tarfile.TarInfo(path);info.size=len(data);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data))
 payload=gzip.compress(out.getvalue(),mtime=0);dest=HERE/'runs'/f'{name}.tar.gz';write(dest,payload)
 rec={'path':str(dest.relative_to(HERE)),'bytes':len(payload),'sha256':digest(payload),'source':relative(source),'members':[{'path':p,'bytes':len(d),'sha256':digest(d)} for p,d in sorted(members.items())],'original_run_status':manifest['status'],'original_outputs_verified':len(manifest['outputs'])}
 if recovery['found'] or recovery['unavailable']:rec['source_capsule_recovery']={'matched':len(recovery['found']),'unavailable':recovery['unavailable']}
 files.append(rec)
 for path,data in members.items():
  if not path.startswith('recovered-source-capsule/') and path!='source-capsule-recovery.json':
   if (source/path).read_bytes()!=data:raise RuntimeError(f'original Run changed: {name}/{path}')

for run in ['sealed-pilot-validation-20261008-001','sealed-profile-pilot-20261008-001','sealed-profile-pilot-20261008-002']:
 run_archive(run)

selected_dirs=['count-coordinate-002','count-coordinate-003','count-coordinate-004','count-hybrid-sweep-001','count-hybrid-sweep-003','count-precision-001','count-remaining-reference-001','count-error-decomposition-001','count-payload-decomposition-001','sealed-native-probe']
for name in selected_dirs:
 for source in sorted((INV/name).rglob('*')):
  if not source.is_file() or '__pycache__' in source.parts:continue
  compress=source.name=='rows.jsonl'
  archive_file(source,Path('investigation')/source.relative_to(INV).with_name(source.name+('.gz' if compress else '')),compress)
selected_scripts=['count_coordinate_probe_original.py','count_coordinate_probe_gridonly.py','count_coordinate_probe.py','count_hybrid_sweep_initial.py','count_hybrid_sweep.py','count_precision_probe.py','count_remaining_reference_probe.py','count_error_decomposition_probe.py','count_payload_decomposition_probe.py','count-interpolation-findings.txt','reader_initialization_probe.py','reader_initialization_probe.json','sealed_warm_profile_probe.py','sealed_warm_profile_probe.json']
for name in selected_scripts:archive_file(INV/name,Path('investigation')/name)
archive_file(WORKSPACE/'outputs/alt-sealed-native-evidence.txt',Path('investigation')/'alt-sealed-native-evidence.txt')
for name in ['sealed-reader-corrected-diagnostic-20261008-001.json','sealed-pilot-build-20261008-001.log','sealed-pilot-validation-20261008-001.log','sealed-profile-pilot-20261008-001.log','sealed-profile-pilot-20261008-002.log','sealed-profile-pilot-config-20261008-001.json','engine-sealed-pilot-20261008-001.json','sealed-full-pytest-20261008-001.log','sealed-full-appendix-20261008-001.log','sealed-shared-memory-tests-20261008-001.log','sealed-native-integration-20261008-001.log','sealed-native-distributed-20261008-002.log','sealed-native-full-appendix-20261008-002.log','sealed-native-full-pytest-20261008-002.log']:
 p=REPO/'output/alt2027'/name
 if p.exists():archive_file(p,Path('working-records')/name)
for name in ['sealed-full-store-83224dc-20261008-001','scitas-reference-depth-complete-20261008']:
 for p in sorted((REPO/'output/alt2027'/name).rglob('*')):
  if p.is_file() and p.stat().st_size<=10_000_000 and p.suffix!='.bin':archive_file(p,Path('cluster')/name/p.relative_to(REPO/'output/alt2027'/name))
# Record generated archive instructions and the archiving procedure itself too.
for p in [HERE/'README.md',Path(__file__)]:
 data=p.read_bytes();files.append({'path':str(p.relative_to(HERE)),'bytes':len(data),'sha256':digest(data),'source':'generated archival documentation/procedure'})
manifest={'schema_version':1,'purpose':'sealed-store development evidence; numerical admission remains separate','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'exclusions':['All numerical kernel-store .bin payloads','Python bytecode caches','Out-of-range count-hybrid-sweep-002 rows (excluded diagnostic described in README)','No current source substituted for an unavailable historical source hash']}
# Verify bytes and every archive member independently after writing.
for record in files:
 p=HERE/record['path'];data=p.read_bytes();assert len(data)==record['bytes'] and digest(data)==record['sha256']
 if 'members' in record:
  with tarfile.open(p,'r:gz') as tar:
   actual={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
  assert set(actual)=={m['path'] for m in record['members']}
  for member in record['members']:
   payload=actual[member['path']];assert len(payload)==member['bytes'] and digest(payload)==member['sha256']
 if record.get('encoding')=='gzip':
  payload=gzip.decompress(data);assert len(payload)==record['source_bytes'] and digest(payload)==record['source_sha256']
manifest['verification']={'status':'passed','files_verified':len(files),'archive_members_verified':sum(len(r.get('members',[])) for r in files),'lossless_gzip_rows_verified':sum(r.get('encoding')=='gzip' for r in files),'total_archived_file_bytes':sum(r['bytes'] for r in files)}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(manifest['verification'],indent=2))
for f in files:
 if 'source_capsule_recovery' in f:print(f['path'],json.dumps(f['source_capsule_recovery']))
