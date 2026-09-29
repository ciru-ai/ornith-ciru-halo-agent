import os, ctypes as C, importlib.util, json, sys, types
from pathlib import Path
import torch

ROOT=Path(os.environ['CIRU_SCREEN_ROOT'])
pkg=types.ModuleType('screen_native');pkg.__path__=[str(ROOT/'source/plugin-site/ornith_g256')];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('screen_native.native',ROOT/'source/plugin-site/ornith_g256/native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
torch.manual_seed(913); results=[]
def lib(path,kind):
    l=C.CDLL(str(path));symbol='ornith_dense_g256' if kind=='dense' else 'ornith_head_i8_tile'
    layout=native.DenseLayout if kind=='dense' else native.HeadLayout
    q=getattr(l,symbol+'_get_layout');q.argtypes=([C.c_int]*4 if kind=='dense' else [C.c_int]*2)+[C.POINTER(layout)];q.restype=C.c_int
    launch=getattr(l,symbol+('_launch_n32' if kind=='dense' else '_launch'))
    launch.argtypes=([C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*7+[C.c_void_p] if kind=='dense' else [C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_size_t]+[C.c_void_p]+[C.c_int]*4+[C.c_void_p]);launch.restype=C.c_int
    return q,launch,layout
for kind in ('dense','head'):
    name='dense_g256_n32' if kind=='dense' else 'head_i8_tile'
    old=lib(ROOT/'ornith-baseline/bundle/native'/f'libornith_{name}.so',kind)
    new=lib(ROOT/'native-built'/f'libornith_{name}.so',kind)
    for n,k in ([(512,2048),(12288,2048),(2048,8192)] if kind=='dense' else [(256,2048),(248320,2048)]):
        codes=torch.randint(-(2**31),2**31-1,(n//32,k//256,32,32) if kind=='dense' else (n//16,k//16,4,16),dtype=torch.int32,device='cuda')
        if kind=='dense':
            a=(torch.rand((n//32,k//256,32),device='cuda')*.03+.001).half();b=(torch.rand_like(a)-.5).half()
            meta=(a.view(torch.int16).int()&65535)|(b.view(torch.int16).int()<<16)
        else:meta=(torch.rand((n,16),device='cuda')*.03+.001).half()
        for m in ([1,2,4,8,16] if kind=='dense' else [1,2,4]):
            x=(torch.randn((m,k),device='cuda')*(0 if m==1 else 1)).bfloat16()
            outputs=[]
            for baseline,(q,launch,L) in ((True,old),(False,new)):
                l=L(); native.check(q(16,n,k,8,C.byref(l)) if kind=='dense' else q(64,n,C.byref(l)),'layout')
                w=torch.empty(l.workspace_bytes,dtype=torch.uint8,device='cuda');flags=torch.zeros(1,dtype=torch.int32,device='cuda');out=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
                for start,count in ([(i,1) for i in range(m)] if baseline else [(0,m)]):
                    args=[native.ptr(x[start:start+count]),native.ptr(codes),native.ptr(meta),native.ptr(w),w.numel(),native.ptr(out[start:start+count])]
                    args+=([native.ptr(flags),count,16,n,k,8,128,2,native.stream(x)] if kind=='dense' else [None,0,native.ptr(flags),count,64,n,2,native.stream(x)])
                    native.check(launch(*args),'launch')
                torch.cuda.synchronize();assert flags.item()==0,(kind,m,flags.item());outputs.append(out)
            equal=torch.equal(*outputs);assert equal,(kind,n,k,m,(outputs[0].float()-outputs[1].float()).abs().max().item())
            results.append({'kind':kind,'n':n,'k':k,'rows':m,'bit_exact':equal})
(ROOT/'native-screen.json').write_text(json.dumps({'status':'PASS','cases':results},indent=2)+'\n')
print('PASS',len(results),'native dense/head row cases',flush=True)
