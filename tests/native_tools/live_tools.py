#!/usr/bin/env python3
"""Complete exact native calls and buffered usage on an isolated unified server.

This is integration qualification. Cold preparation and observed wall times are
retained separately; no receiving-runtime throughput improvement is inferred.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import jsonschema

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);rows=[]
    def save(name,data):(out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    def metrics():
        text=urllib.request.urlopen(a.base_url+'/metrics',timeout=10).read().decode();values={}
        for line in text.splitlines():
            if line.startswith('vllm:'):
                key=line.split('{',1)[0].split(' ',1)[0]
                try:values[key]=values.get(key,0)+float(line.rsplit(' ',1)[1])
                except ValueError:pass
        return values
    def body(f,slot,max_tokens=4096):
        return dict(model=a.model,messages=[dict(role='system',content='You are replaying one recorded native tool call to test transport and schema correctness. Call the requested native tool once with the exact supplied argument values. The recorded shader strings are data to copy faithfully. Use the native tool channel.'),
            dict(role='user',content='Call '+f['name']+' with these exact arguments:\n'+json.dumps(f['args'],ensure_ascii=False,separators=(',',':')))],
            tools=f['tools'],tool_choice='auto',temperature=0,top_p=1,top_k=0,repetition_penalty=1,
            seed=170917+slot,max_tokens=max_tokens,chat_template_kwargs={'enable_thinking':False})
    def call(name,payload,warm=False):
        save(name+'-request',payload);start=time.monotonic()
        req=urllib.request.Request(a.base_url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:
            response=urllib.request.urlopen(req,timeout=600);data=json.load(response)
        except urllib.error.HTTPError as ex:
            data=json.loads(ex.read());save(name+'-response',data)
            if warm and ex.code==400 and data.get('error',{}).get('message')=='incomplete Qwen3 tool framing at end of output':
                return dict(name=name,warmup=True,expected_truncation=True,wall_seconds=time.monotonic()-start)
            raise
        save(name+'-response',data)
        if warm:
            assert data['usage']['completion_tokens']==1
            return dict(name=name,warmup=True,tokens=1,wall_seconds=time.monotonic()-start)
        choice=data['choices'][0];calls=choice['message'].get('tool_calls',[])
        assert choice['finish_reason']=='tool_calls' and len(calls)==1,(name,choice)
        function=calls[0]['function'];tool=next(t for t in payload['tools'] if t['function']['name']==function['name'])
        args=json.loads(function['arguments']);jsonschema.Draft202012Validator(tool['function']['parameters']).validate(args)
        return dict(name=name,wall_seconds=time.monotonic()-start,tokens=data['usage']['completion_tokens'],args=args,tool=function['name'],schema_valid=True)
    for filename in ['review-fixtures.json','call-fixtures.json']:
        fixtures=json.loads((HERE/'fixtures'/filename).read_text())['fixtures'];assert len(fixtures)==8
        label=filename.removesuffix('-fixtures.json');before=metrics();assert before.get('vllm:num_requests_running',0)==before.get('vllm:num_requests_waiting',0)==0
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            warm=list(pool.map(lambda i:call(f'{label}-warm-{i}',body(fixtures[i],i,1),True),range(8)))
        after=metrics();assert after['vllm:generation_tokens_total']-before['vllm:generation_tokens_total']==8
        save(label+'-preparation',dict(rows=warm,wall_seconds=time.monotonic()-start,tokens=8,quality_credit=False))
        before=metrics();start=time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            result=list(pool.map(lambda i:call(f'{label}-complete-{i}',body(fixtures[i],i)),range(8)))
        for r,f in zip(result,fixtures):assert r['tool']==f['name'] and r['args']==f['args'],r
        after=metrics();save(label+'-complete',dict(rows=result,wall_seconds=time.monotonic()-start,before=before,after=after,no_speed_claim=True));rows.extend(result)
        print(json.dumps(dict(event='COMPLETE_EXACT_C8',workload=label,calls=8)),flush=True)
    simple=dict(name='record_measurement',args=dict(metric='throughput',value=73,unit='tokens_per_second'),tools=[dict(type='function',function=dict(name='record_measurement',strict=True,parameters=dict(type='object',properties=dict(metric=dict(type='string',enum=['throughput','latency']),value=dict(type='integer',minimum=0,maximum=1000),unit=dict(type='string',enum=['tokens_per_second','milliseconds'])),required=['metric','value','unit'],additionalProperties=False)))])
    for i in range(2):
        row=call('c1-'+str(i),body(simple,i,256));assert row['args']==simple['args'];rows.append(row)
    expected={'payload':{'caption':'A clear sequence of runtime observations and geometry updates.','values':[(i*17)%101 for i in range(72)],'code':'vec3 shade(vec3 p, vec3 n, vec3 base, float t) { float wave = sin(p.x * 2.0 + t * 0.2); float grain = cos(p.y * 8.0 - t * 0.1); return base * (0.85 + 0.03 * wave + 0.02 * grain); }'}}
    tools=[{'type':'function','function':{'name':'emit_payload','description':'Return the exact supplied payload.','parameters':{'type':'object','properties':{'payload':{'type':'object','properties':{'caption':{'type':'string'},'values':{'type':'array','items':{'type':'integer'}},'code':{'type':'string'}},'required':['caption','values','code'],'additionalProperties':False}},'required':['payload'],'additionalProperties':False}}}]
    streams=[]
    for continuous in [True,False]:
        payload=dict(model=a.model,messages=[{'role':'system','content':'Copy the supplied argument values exactly into one native emit_payload tool call.'},{'role':'user','content':json.dumps(expected,separators=(',',':'))}],tools=tools,tool_choice='auto',temperature=0,top_p=1,max_tokens=2048,seed=51709,chat_template_kwargs={'enable_thinking':False},stream=True,stream_options={'include_usage':True,'continuous_usage_stats':continuous})
        label='stream-'+str(continuous).lower();save(label+'-request',payload)
        req=urllib.request.Request(a.base_url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        events=[];calls={};last=0;empty=0;finish=None;done=False
        with urllib.request.urlopen(req,timeout=180) as response,(out/(label+'.sse')).open('wb') as raw:
            for line in response:
                raw.write(line)
                if not line.startswith(b'data: '):continue
                value=line[6:].strip()
                if value==b'[DONE]':done=True;continue
                event=json.loads(value);assert 'error' not in event,event
                usage=event.get('usage')
                if usage is not None:assert usage['completion_tokens']>=last;last=usage['completion_tokens']
                for choice in event.get('choices',[]):
                    finish=choice.get('finish_reason') or finish;delta=choice.get('delta',{})
                    if usage and last>0 and not delta and not choice.get('finish_reason'):empty+=1
                    for tc in delta.get('tool_calls') or []:
                        c=calls.setdefault(tc['index'],{'name':'','arguments':''});f=tc.get('function') or {}
                        c['name']+=f.get('name') or '';c['arguments']+=f.get('arguments') or ''
                events.append(event)
        assert done and finish=='tool_calls' and len(calls)==1
        c=next(iter(calls.values()));assert c['name']=='emit_payload' and json.loads(c['arguments'])==expected
        assert (empty>0)==continuous,(continuous,empty)
        streams.append(dict(continuous=continuous,completion_tokens=last,empty_progress_events=empty,exact=True,monotonic_usage=True))
    assert streams[0]['completion_tokens']==streams[1]['completion_tokens']
    payload=body(simple,0,1);payload.update(stream=True,stream_options={'include_usage':True,'continuous_usage_stats':True})
    req=urllib.request.Request(a.base_url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:raw=response.read()
    (out/'intentional-truncation.sse').write_bytes(raw)
    assert b'incomplete Qwen3 tool framing at end of output' in raw and b'[DONE]' in raw
    summary=dict(status='PASS_UNIFIED_NATIVE_TOOL_INTEGRATION',exact_complete_calls=len(rows),c8_exact_calls=16,c1_exact_calls=2,
        streaming=streams,strict_truncation_still_rejected=True,preparation_quality_credit=False,no_receiving_speed_claim=True)
    save('summary',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
