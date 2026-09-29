import os, json
from pathlib import Path
root=Path(os.environ['CIRU_SCREEN_ROOT'])
# Test nonzero M1 explicitly, then compare the retained normal routed entry.
source=(root/'native_screen.py').read_text().replace('*(0 if m==1 else 1)','*1').replace("root/'native-screen.json'","root/'native-nonzero-screen.json'").replace("ROOT/'native-screen.json'","ROOT/'native-nonzero-screen.json'")
exec(compile(source,str(root/'native_screen.py'),'exec'),{})
s=(root/'plugin_screen.py').read_text().split("native=module('native')")[1]
s="native=module('native')"+s
s=s.split("layout=native.RoutedLayout();native.check(new.ornith_routed_shared_get_layout")[0]
namespace={}
exec(compile((root/'plugin_screen.py').read_text().split("gemv=module('small_gemv')")[0],str(root/'plugin_screen.py'),'exec'),namespace)
exec(compile(s,str(root/'plugin_screen.py'),'exec'),namespace)
C=namespace['C'];torch=namespace['torch'];native=namespace['native'];old=namespace['old'];new=namespace['new']
new.ornith_routed_direct_get_layout.argtypes=old.ornith_routed_direct_get_layout.argtypes
new.ornith_routed_direct_launch.argtypes=old.ornith_routed_direct_launch.argtypes
checks=[]
for count in (1,4,16):
    x=torch.randn((count,2048),device='cuda').bfloat16()
    weights=torch.rand((count,8),device='cuda');weights/=weights.sum(-1,keepdim=True)
    ids=torch.randint(0,256,(count,8),device='cuda',dtype=torch.int32)
    outputs=[]
    for lib in (old,new):
        layout=native.RoutedLayout();native.check(lib.ornith_routed_direct_get_layout(count,C.byref(layout)),'normal routed layout')
        work=torch.empty(layout.workspace_bytes,device='cuda',dtype=torch.uint8);out=torch.empty_like(x);flags=torch.zeros(1,dtype=torch.int32,device='cuda')
        args=[native.ptr(t) for t in (x,weights,ids,namespace['gate'],namespace['gm'],namespace['down'],namespace['dm'],work)]
        native.check(lib.ornith_routed_direct_launch(*args,work.numel(),native.ptr(out),native.ptr(flags),count,count,native.stream(x)),'normal routed retained')
        torch.cuda.synchronize();assert flags.item()==0;outputs.append(out)
    exact=torch.equal(*outputs);checks.append({'rows':count,'bit_exact':exact,'max_abs_error':(outputs[0].float()-outputs[1].float()).abs().max().item()})
    print(checks[-1],flush=True)
(root/'retained-routed-screen.json').write_text(json.dumps(checks,indent=2)+'\n')
assert all(c['bit_exact'] for c in checks)
