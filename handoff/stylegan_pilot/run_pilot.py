"""Fixed 20-kimg official StyleGAN2-ADA pilot. No fitting without --execute."""
import argparse,hashlib,io,json,os,subprocess,sys,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA_SHA='028216f4780719022b0200463c0e411a5cc18b64b4f4460e55c49a227a287f6d'
SPLIT_SHA='4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
COMMIT='d72cc7d041b42ec8e806021a205ed9349f87c6a4'

def validate_data():
    from PIL import Image
    data=ROOT/'data/isic_train128.zip'
    if hashlib.sha256(data.read_bytes()).hexdigest()!=DATA_SHA:raise RuntimeError('Training ZIP hash differs')
    audit=json.loads((ROOT/'data/isic_train128.audit.json').read_text())
    assert audit['dataset_sha256']==DATA_SHA and audit['split_sha256']==SPLIT_SHA
    assert audit['validation_test_image_content_read'] is False
    with zipfile.ZipFile(data) as z:
        assert z.testzip() is None
        labels=json.loads(z.read('dataset.json'))['labels']
        assert len(labels)==972 and Counter(t for _,t in labels)=={0:530,1:442}
        assert len({n for n,_ in labels})==972 and len(z.namelist())==973
        sources={r['image_id']:r['target'] for r in audit['training_sources']}
        for name,target in labels:
            assert sources[Path(name).stem]==target
            with Image.open(io.BytesIO(z.read(name))) as im:assert im.mode=='RGB' and im.size==(128,128)
    return data.resolve()

def logged(command,log,cwd=None,env=None):
    with open(log,'w',encoding='utf-8') as f:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            rc=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if rc:raise RuntimeError(f'Command failed ({rc}); see {log}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Run the fixed pilot after all checks pass')
    p.add_argument('--validate-data-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/pilot_seed142')
    args=p.parse_args()
    data=validate_data();print('DATA PASS: 972 training-only RGB128 images, 530 benign / 442 malignant',flush=True)
    if args.validate_data_only:return
    import torch
    assert torch.__version__.split('+')[0]=='2.11.0', 'Use the supplied PyTorch 2.11 environment'
    assert torch.version.cuda and torch.version.cuda.startswith('13.'), 'Use CUDA 13 PyTorch/toolkit as documented'
    assert torch.cuda.is_available(), 'CUDA unavailable; no training was started'
    signature=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    statefile=out/'pilot_state.json'
    if statefile.exists():
        state=json.loads(statefile.read_text())
        if state['status']=='complete':print('This pilot is already complete:',out);return
        if state['status']!='prepared':raise RuntimeError('Previous pilot started or failed. Preserve it; use a new --output directory for a deliberate new run. No automatic resume.')
        assert state['dataset_sha256']==DATA_SHA and state['runtime_signature']==signature
    runtime=out/'runtime'
    if not runtime.exists():logged([sys.executable,'-u',str(ROOT/'verify_runtime.py'),'--output',str(runtime)],out/'runtime_check.log')
    report=json.loads((runtime/'stylegan_gpu_runtime_report.json').read_text())
    assert report['status']=='gpu_and_targeted_runtime_checks_passed'
    assert report['network_check']['tests']['path_length_double_backward']
    source=runtime/'source_checkout/stylegan2-ada-pytorch'
    env=os.environ.copy();env.update(MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='.'.join(map(str,signature['capability'])),TORCH_EXTENSIONS_DIR=str(runtime/'cuda_extensions'))
    command=[sys.executable,'-u',str(source/'train.py'),f'--outdir={out/"runs"}',f'--data={data}','--gpus=1','--cfg=auto','--cond=1','--aug=ada','--augpipe=bg','--mirror=0','--batch=16','--kimg=20','--snap=1','--metrics=none','--seed=142','--workers=2']
    assert not (out/'runs').exists(),'Training output already exists; preserve it and choose a new --output'
    logged(command+['--dry-run'],out/'official_dry_run.log',cwd=source,env=env)
    assert 'Dry run; exiting.' in (out/'official_dry_run.log').read_text()
    state=dict(status='prepared',training_launched=False,dataset_sha256=DATA_SHA,split_sha256=SPLIT_SHA,upstream_commit=COMMIT,runtime_signature=signature,command=command,seed=142,kimg=20,batch=16,gpus=1,classes={'0':'benign','1':'malignant'},pretrained_weights=None,prepared_at=datetime.now(timezone.utc).isoformat())
    statefile.write_text(json.dumps(state,indent=2))
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    if not args.execute:
        print('READY: runtime, data, regularization and official dry run passed. Training NOT started.',flush=True);return
    state.update(status='running',training_launched=True,started_at=datetime.now(timezone.utc).isoformat());statefile.write_text(json.dumps(state,indent=2))
    try:
        logged(command,out/'training.log',cwd=source,env=env)
        checkpoints=sorted((out/'runs').rglob('network-snapshot-*.pkl'))
        assert checkpoints and checkpoints[-1].name=='network-snapshot-000020.pkl','Expected final 20-kimg snapshot missing'
        final=checkpoints[-1]
        from PIL import Image
        for target in [0,1]:
            images=out/f'samples_class{target}'
            logged([sys.executable,str(source/'generate.py'),f'--network={final}','--seeds=0-15',f'--class={target}','--trunc=1',f'--outdir={images}'],out/f'generate_class{target}.log',cwd=source,env=env)
            canvas=Image.new('RGB',(512,512))
            for j,image in enumerate(sorted(images.glob('seed*.png'))):
                with Image.open(image) as im:canvas.paste(im.convert('RGB'),((j%4)*128,(j//4)*128))
            canvas.save(out/f'grid_class{target}_final.png')
        state.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),final_checkpoint=str(final),final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:statefile.write_text(json.dumps(state,indent=2))
    print('PILOT COMPLETE. Preserve this entire results folder; image quality still requires review.',flush=True)

if __name__=='__main__':main()
