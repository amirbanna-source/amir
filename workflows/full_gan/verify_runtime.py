"""Verify pinned StyleGAN2-ADA on CUDA without training or reading image data.

Two recorded compatibility patches: extension module loading and NVIDIA's
newer grid-gradient API. No network architecture or CUDA kernel changes.
Requires a fresh output folder and a T4-compatible PyTorch/CUDA runtime.
"""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output_root = args.output.resolve()
assert not output_root.exists(), 'Use a fresh output directory'
output_root.mkdir(parents=True)


from pathlib import Path
import subprocess, sys, json, platform, importlib.util

UPSTREAM_COMMIT = 'd72cc7d041b42ec8e806021a205ed9349f87c6a4'
work = output_root/'source_checkout'
assert not work.exists(), 'Choose a fresh work folder; preserve previous evidence'
work.mkdir()
upstream = work / 'stylegan2-ada-pytorch'
import shutil, hashlib
vendor_root = Path(__file__).resolve().parent/'vendor'
source_manifest = json.loads((vendor_root/'SOURCE_MANIFEST.json').read_text())
actual = source_manifest['commit']
assert actual == UPSTREAM_COMMIT
for name, expected in source_manifest['files'].items():
    assert hashlib.sha256((vendor_root/'stylegan2-ada-pytorch'/name).read_bytes()).hexdigest() == expected, name
shutil.copytree(vendor_root/'stylegan2-ada-pytorch',upstream)
required = ['torch','click','requests','scipy','psutil','tqdm','PIL','ninja','imageio','imageio_ffmpeg']
missing = [name for name in required if importlib.util.find_spec(name) is None]
assert not missing, f'Install official dependencies in a compatible runtime first: {missing}'
import torch
environment = dict(python=platform.python_version(), torch=torch.__version__,
                   cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   upstream_commit=actual)
(work/'environment.json').write_text(json.dumps(environment,indent=2))
print(environment)


from pathlib import Path
import platform, shutil, subprocess, json, torch
RUN_FEASIBILITY = False
check = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count(), nvidia_smi=shutil.which('nvidia-smi'), nvcc=shutil.which('nvcc'), training_launched=False)
for name in ['nvidia-smi','nvcc']:
    executable = shutil.which(name)
    if executable:
        args = [executable] + (['--version'] if name == 'nvcc' else ['--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])
        r = subprocess.run(args,capture_output=True,text=True)
        check[name+'_output'] = r.stdout or r.stderr
if torch.cuda.is_available():
    x = torch.arange(16,device='cuda',dtype=torch.float32)
    check['cuda_tensor_test'] = float((x*x).sum().cpu()) == 1240.0
    check['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
(output_root/'gpu_runtime_check.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
print('GPU_RUNTIME_CHECK_COMPLETE')

assert check['cuda_available'] and check.get('cuda_tensor_test'), 'CUDA runtime unavailable'


import hashlib, difflib, importlib
loader = upstream/'torch_utils/custom_ops.py'
original_loader = loader.read_text()
assert hashlib.sha256(original_loader.encode()).hexdigest() == 'ee123ae0458fbe463000d5962a5b804d9e91b25d3fda8500eeaa6ddf6b5c932a'
assert original_loader.count('torch.utils.cpp_extension.load(') == 2
assert original_loader.count('module = importlib.import_module(module_name)') == 1
patched_loader = original_loader.replace('torch.utils.cpp_extension.load(', 'module = torch.utils.cpp_extension.load(').replace('        module = importlib.import_module(module_name)\n','')
loader.write_text(patched_loader)
patch_text = ''.join(difflib.unified_diff(original_loader.splitlines(True),patched_loader.splitlines(True),fromfile='a/torch_utils/custom_ops.py',tofile='b/torch_utils/custom_ops.py'))
(output_root/'stylegan_pytorch_loader.patch').write_text(patch_text)
patch_info = dict(upstream_commit=UPSTREAM_COMMIT, file='torch_utils/custom_ops.py', original_sha256=hashlib.sha256(original_loader.encode()).hexdigest(), patched_sha256=hashlib.sha256(patched_loader.encode()).hexdigest(), reason='Use cpp_extension.load returned Python module instead of a second import by name.', architecture_changed=False, cuda_kernel_source_changed=False)
(output_root/'stylegan_loader_patch.json').write_text(json.dumps(patch_info,indent=2))
print('LOADER_COMPATIBILITY_PATCH_APPLIED')


grid_file = upstream/'torch_utils/ops/grid_sample_gradfix.py'
grid_original = grid_file.read_text()
assert hashlib.sha256(grid_original.encode()).hexdigest() == 'a49a8195b734ecba95c9b63d035f26faa5d794a0b26eb3e9d5f3c521be29c536'
start = grid_original.index('def _should_use_custom_op():')
end = grid_original.index('#----------------------------------------------------------------------------',start)
grid_patched = grid_original[:start] + 'def _should_use_custom_op():\n    return enabled\n\n' + grid_original[end:]
old_call = "        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')\n        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)"
new_call = "        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')[0]\n        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])\n        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False, output_mask)"
assert grid_patched.count(old_call) == 1
grid_patched = grid_patched.replace(old_call,new_call)
grid_file.write_text(grid_patched)
grid_patch = ''.join(difflib.unified_diff(grid_original.splitlines(True),grid_patched.splitlines(True),fromfile='a/torch_utils/ops/grid_sample_gradfix.py',tofile='b/torch_utils/ops/grid_sample_gradfix.py'))
(output_root/'stylegan_grid_gradfix.patch').write_text(grid_patch)
grid_patch_info = dict(source='https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/grid_sample_gradfix.py', changes='Port NVIDIA newer API handling: use enabled gradfix, operation tuple[0], backward output_mask. Target runtime PyTorch 2.11 only.', original_sha256=hashlib.sha256(grid_original.encode()).hexdigest(), patched_sha256=hashlib.sha256(grid_patched.encode()).hexdigest(), architecture_changed=False)
(output_root/'stylegan_grid_patch.json').write_text(json.dumps(grid_patch_info,indent=2))
print('NVIDIA_GRID_API_COMPATIBILITY_UPDATE_APPLIED')


import os, sys, traceback, warnings
os.environ['MAX_JOBS'] = '2'
os.environ['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str,torch.cuda.get_device_capability(0)))
os.environ['TORCH_EXTENSIONS_DIR'] = str(output_root/'cuda_extensions')
sys.path.insert(0,str(upstream))
from torch_utils.ops import bias_act, upfirdn2d
from torch_utils import custom_ops
custom_ops.verbosity = 'full'
ops_report = dict(environment=environment, upstream_commit=UPSTREAM_COMMIT, training_launched=False, optimizer_updates=0, tests={})
for name, module in [('bias_act',bias_act),('upfirdn2d',upfirdn2d)]:
    print('COMPILING',name,flush=True)
    try:
        assert module._init(), 'Compiled plugin unavailable; fallback is not a compilation pass'
        x = torch.randn(2,8,16,16,device='cuda',requires_grad=True)
        if name == 'bias_act':
            b = torch.randn(8,device='cuda')
            y = module.bias_act(x,b,act='swish',impl='cuda')
            yr = module.bias_act(x,b,act='swish',impl='ref')
        else:
            f = module.setup_filter([1,3,3,1],device='cuda')
            y = module.upfirdn2d(x,f,padding=2,impl='cuda')
            yr = module.upfirdn2d(x,f,padding=2,impl='ref')
        torch.testing.assert_close(y,yr,rtol=1e-4,atol=1e-5)
        g = torch.autograd.grad(y.square().mean(),x,create_graph=True)[0]
        gr = torch.autograd.grad(yr.square().mean(),x,create_graph=True)[0]
        torch.testing.assert_close(g,gr,rtol=1e-4,atol=1e-5)
        gg = torch.autograd.grad(g.square().sum(),x)[0]
        ggr = torch.autograd.grad(gr.square().sum(),x)[0]
        torch.testing.assert_close(gg,ggr,rtol=2e-4,atol=1e-5)
        assert torch.isfinite(gg).all()
        torch.cuda.synchronize()
        ops_report['tests'][name] = dict(passed=True, compiled_plugin_loaded=module._plugin is not None, max_forward_error=float((y-yr).abs().max().detach().cpu()), forward_backward_second_derivative=True)
    except Exception:
        ops_report['tests'][name] = dict(passed=False,error=traceback.format_exc())
    (output_root/'stylegan_cuda_ops_check.json').write_text(json.dumps(ops_report,indent=2))
    print(name,ops_report['tests'][name],flush=True)
ops_report['all_passed'] = all(v['passed'] for v in ops_report['tests'].values())
(output_root/'stylegan_cuda_ops_check.json').write_text(json.dumps(ops_report,indent=2))
print('OFFICIAL_CUDA_OPS_CHECK_COMPLETE',ops_report['all_passed'])


warnings.filterwarnings('once',message='conv2d_gradfix not supported.*')
from training.networks import Generator, Discriminator
from training.augment import AugmentPipe
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
conv2d_gradfix.enabled = True
grid_sample_gradfix.enabled = True
network_check = dict(training_launched=False, optimizer_updates=0, real_images_read=False, resolution=128, c_dim=2, tests={})
try:
    torch.manual_seed(1142)
    G = Generator(z_dim=512,c_dim=2,w_dim=512,img_resolution=128,img_channels=3,mapping_kwargs=dict(num_layers=2),synthesis_kwargs=dict(channel_base=16384,channel_max=512,num_fp16_res=4,conv_clamp=256)).cuda()
    D = Discriminator(c_dim=2,img_resolution=128,img_channels=3,channel_base=16384,channel_max=512,num_fp16_res=4,conv_clamp=256).cuda()
    c = torch.eye(2,device='cuda')
    z = torch.randn(2,512,device='cuda')
    fake = G(z,c,noise_mode='const')
    score = D(fake,c)
    score.mean().backward()
    assert torch.isfinite(fake).all() and all(torch.isfinite(p.grad).all() for p in G.parameters() if p.grad is not None)
    network_check['tests']['generator_discriminator_backward'] = True
    G.zero_grad(set_to_none=True)
    D.zero_grad(set_to_none=True)
    a = AugmentPipe(xflip=1,rotate90=1,xint=1,scale=1,rotate=1,aniso=1,xfrac=1).cuda()
    a.p.copy_(torch.ones_like(a.p))
    x = torch.randn(2,3,128,128,device='cuda',requires_grad=True)
    logits = D(a(x),c)
    with conv2d_gradfix.no_weight_gradients():
        dx = torch.autograd.grad(logits.sum(),x,create_graph=True)[0]
    dx.square().sum((1,2,3)).mean().backward()
    assert torch.isfinite(dx).all() and all(torch.isfinite(p.grad).all() for p in D.parameters() if p.grad is not None)
    network_check['tests']['augmentation_and_r1_double_backward'] = True
    network_check['passed'] = True
except Exception:
    network_check['passed'] = False
    network_check['error'] = traceback.format_exc()
finally:
    (output_root/'stylegan_network_check.json').write_text(json.dumps(network_check,indent=2))
    print(json.dumps(network_check,indent=2))
    print('SYNTHETIC_NETWORK_CHECK_COMPLETE')


try:
    torch.manual_seed(2142)
    G.zero_grad(set_to_none=True)
    z = torch.randn(2,512,device='cuda')
    ws = G.mapping(z,c)
    imgs = G.synthesis(ws,noise_mode='const')
    noise = torch.randn_like(imgs)/(imgs.shape[2]*imgs.shape[3])**0.5
    with conv2d_gradfix.no_weight_gradients():
        dw = torch.autograd.grad((imgs*noise).sum(),ws,create_graph=True)[0]
    penalty = dw.square().sum(2).mean(1).add(1e-8).sqrt().square().mean()
    penalty.backward()
    assert torch.isfinite(penalty) and all(torch.isfinite(p.grad).all() for p in G.parameters() if p.grad is not None)
    network_check['tests']['path_length_double_backward'] = True
except Exception:
    network_check['passed'] = False
    network_check['path_length_error'] = traceback.format_exc()
(output_root/'stylegan_network_check.json').write_text(json.dumps(network_check,indent=2))

runtime_report = dict(status='gpu_and_targeted_runtime_checks_passed' if ops_report['all_passed'] and network_check['passed'] else 'runtime_check_failed', hardware=check, cuda_ops=ops_report, network_check=network_check, loader_patch=patch_info, grid_patch=grid_patch_info, conv2d_gradfix_native_fallback=True, training_launched=False, optimizer_updates=0, real_image_content_read=False, limitations='Targeted synthetic checks only; full training loop, checkpointing and model quality remain unvalidated.')
(output_root/'stylegan_gpu_runtime_report.json').write_text(json.dumps(runtime_report,indent=2))
print(json.dumps(runtime_report,indent=2))
assert ops_report['all_passed'] and network_check['passed'], 'Runtime validation failed; do not train'
