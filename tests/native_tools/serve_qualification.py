#!/usr/bin/env python3
"""One isolated serving integration run, owned by the shared GPU guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent


def main():
    if os.environ.get('QTIP_GPU_OWNED')!='1':raise RuntimeError('Use the shared GPU/service guard')
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=18755)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);bundle=a.bundle.resolve()
    build=json.loads((bundle/'BUILD.json').read_text())
    for name,h in build['files'].items():assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==h,name
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
    command=['bash',str(bundle/'serve.sh'),'--host','127.0.0.1','--port',str(a.port),'--served-name','ornith-unified']
    env=dict(os.environ);env.pop('VLLM_PLUGINS',None)
    save('settings.json',dict(command=command,bundle=str(bundle),build_sha256=hashlib.sha256((bundle/'BUILD.json').read_bytes()).hexdigest(),
        full_parent_settings=True,tool_replay_non_thinking=True,model_quality_benchmark=False,no_speed_claim=True))
    server=None
    with (out/'server.log').open('x') as log:
        try:
            server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            base=f'http://127.0.0.1:{a.port}';deadline=time.monotonic()+480
            while time.monotonic()<deadline:
                if server.poll() is not None:raise RuntimeError(f'Server exited {server.returncode}; inspect server.log')
                try:
                    with urllib.request.urlopen(base+'/v1/models',timeout=3) as response:models=json.load(response)
                    if models.get('data'):break
                except Exception:time.sleep(1)
            else:raise TimeoutError('Server readiness timeout')
            save('models.json',models)
            with (out/'client.log').open('x') as client:
                subprocess.run([sys.executable,str(HERE/'live_tools.py'),'--base-url',base,'--model','ornith-unified','--out',str(out/'calls')],
                    stdout=client,stderr=subprocess.STDOUT,check=True,timeout=1800)
            maps=[];group=os.getpgrp();native=str(bundle/'plugin-site/ornith_g256/_xgrammar_native/libxgrammar-candidate.so')
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    if os.getpgid(int(proc.name))!=group:continue
                    libs=sorted(set(line.split()[-1] for line in (proc/'maps').read_text().splitlines() if 'libxgrammar' in line))
                    if libs:
                        maps.append(dict(pid=int(proc.name),command=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace'),libraries=libs))
                        assert libs==[native],maps[-1]
                except (ProcessLookupError,FileNotFoundError,PermissionError):continue
            save('native-process-mappings.json',maps)
            assert len(maps)>=2,'Require observed API and engine native mappings'
            assert server.poll() is None,'Serving process exited during qualification'
            save('summary.json',dict(status='PASS_UNIFIED_SERVING_INTEGRATION',
                build_sha256=hashlib.sha256((bundle/'BUILD.json').read_bytes()).hexdigest(),
                calls=json.loads((out/'calls/summary.json').read_text()),native_processes=maps,
                pending_full_final_comparison=True,no_speed_claim=True))
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:server.wait(timeout=60)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':main()
