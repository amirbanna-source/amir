from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np,pandas as pd
from verify_binary_results import verify
root=Path('binary_work/lower_lr');root.mkdir(exist_ok=True)
archive=Path('binary_work/binary_lr0001_all_artifacts.zip')
manifest=pd.read_csv('binary_work/isic2019_binary_split.csv')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 z.extractall(root)
 for seed in [42,43,44]:
  run=f'binary_lr0001_no_aug_seed{seed}'
  temporary=root/f'verify_seed{seed}.zip'
  with zipfile.ZipFile(temporary,'w') as sub:
   for name in z.namelist():
    if name.startswith(run) or name in [f'initial_weights_seed{seed}.npz','isic2019_binary_split.csv']:
     sub.writestr(name,z.read(name))
  metrics=verify(temporary,manifest)
  old=json.loads(Path(f'binary_work/binary_no_aug_seed{seed}_metrics.json').read_text())
  assert old['initial_weights_sha256']==metrics['initial_weights_sha256']
  assert old['split_fingerprint']==metrics['split_fingerprint']
  temporary.unlink()
old=pd.read_csv('binary_work/binary_baseline_results.csv').set_index('seed')
new=pd.read_csv(root/'binary_lr0001_results.csv').set_index('seed')
metrics=['accuracy','macro_f1','balanced_accuracy','malignant_sensitivity','specificity','roc_auc','average_precision']
comparison=[]
for seed in [42,43,44]:
 for metric in metrics:
  a,b=float(old.loc[seed,metric]),float(new.loc[seed,metric])
  comparison.append(dict(seed=seed,metric=metric,original=a,lower_lr=b,delta=b-a))
pd.DataFrame(comparison).to_csv(root/'paired_comparison.csv',index=False)
print(new[['epochs_trained','selected_epoch','validation_loss']+metrics].to_string())
print('\nMEAN / SAMPLE SD')
for m in metrics: print(m,'original',old[m].mean(),'lower',new[m].mean(),'sd',new[m].std(),'delta',(new[m]-old[m]).mean())
print('\nIndependent verification passed for all three runs')
