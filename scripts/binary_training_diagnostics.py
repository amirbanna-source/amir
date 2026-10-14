# Training-only CNN sanity check. No validation/test images are loaded.
from pathlib import Path
from hashlib import sha256
import json, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
out = Path('/kaggle/working/binary_diagnostics')
out.mkdir(exist_ok=True)
splits = pd.read_csv('/kaggle/working/binary_baseline/isic2019_binary_split.csv')
fingerprint = sha256(splits[['image_id','label','group','split']].sort_values('image_id').to_csv(index=False).encode()).hexdigest()
assert fingerprint == '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
assert splits.image_id.is_unique and splits.groupby('group').split.nunique().max() == 1
train = splits[splits.split == 'train'].copy()
assert len(train) == 972
assert (train.target == train.label.map({'benign':0,'malignant':1})).all()
def load_image(path, target):
    x = tf.io.decode_image(tf.io.read_file(path), channels=3,
                           expand_animations=False)
    x.set_shape([None, None, 3])
    return tf.cast(tf.image.resize(x, [128,128]), tf.float32)/255., target

def dataset(part, seed, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((part.path.astype(str).to_numpy(),
                                            part.target.to_numpy('int32')))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(part), seed=seed, reshuffle_each_iteration=True)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.with_options(opts).batch(32).prefetch(tf.data.AUTOTUNE)

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.Input((128,128,3)),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(.35,seed=12345),
        tf.keras.layers.Dense(1,activation='sigmoid')])


# One image per group; deterministic, balanced 16 + 16 training-only subset.
candidates = train.sort_values('image_id').drop_duplicates('group')
subset = pd.concat([candidates[candidates.target == y].sample(16, random_state=2026+y) for y in [0,1]]).sample(frac=1,random_state=2026).reset_index(drop=True)
assert subset.group.is_unique and set(subset.image_id) <= set(train.image_id)
subset.to_csv(out/'training_subset.csv',index=False)
x = tf.stack([load_image(p,0)[0] for p in subset.path])
y = tf.convert_to_tensor(subset.target.to_numpy('float32').reshape(-1,1))
xb,yb = next(iter(dataset(train,42,True)))
audit = dict(split_fingerprint=fingerprint,tensorflow_version=tf.__version__,training_only=True,test_evaluated=False,subset_size=32,class_counts=subset.target.value_counts().to_dict(),image_shape=x.shape.as_list(),label_shape=y.shape.as_list(),pipeline_batch_shape=xb.shape.as_list(),pipeline_label_shape=yb.shape.as_list(),pixel_min=float(tf.reduce_min(x)),pixel_max=float(tf.reduce_max(x)),pixel_std=float(tf.math.reduce_std(x)),protocol=dict(seed=42,subset_seed=2026,learning_rates=[.001,.0001],updates=300,report_every=25,dropout=.35,class_weight=None,augmentation=False,threshold=.5))
assert bool(tf.reduce_all(tf.math.is_finite(x)))
assert 0 <= audit['pixel_min'] <= audit['pixel_max'] <= 1
assert np.all(np.std(x.numpy(),axis=(1,2,3)) > 0)
assert set(yb.numpy()) == {0,1}
(out/'audit.json').write_text(json.dumps(audit,indent=2))
print('BATCH_AUDIT',json.dumps(audit),flush=True)
summary=[]
init_hashes=[]
for lr in [.001,.0001]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model=build_cnn()
    init_hash=sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    init_hashes.append(init_hash)
    assert init_hash == 'f0b9a0ddce8f8452c7308e6e7ae6c7d37f9d3b7eb850da4831cbb7f2651b3953'
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='binary_crossentropy')
    # Separate inference gradient audit avoids advancing Dropout RNG before fitting.
    with tf.GradientTape() as tape:
        initial_p=model(x,training=False)
        loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,initial_p))
    gradients=tape.gradient(loss,model.trainable_variables)
    gradrows=[]
    for variable,gradient in zip(model.trainable_variables,gradients):
        assert gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient)))
        gradrows.append(dict(variable=variable.path,norm=float(tf.linalg.global_norm([gradient])),finite=True))
    assert all(r['norm'] > 0 for r in gradrows if 'kernel' in r['variable'])
    tag=str(lr).replace('.','p')
    pd.DataFrame(gradrows).to_csv(out/f'gradients_lr{tag}.csv',index=False)
    history=[]
    for step in range(301):
        if step % 25 == 0:
            p=model(x,training=False).numpy().ravel()
            pred=(p>=.5).astype(int)
            tn,fp,fn,tp=confusion_matrix(subset.target,pred,labels=[0,1]).ravel()
            row=dict(update=step,learning_rate=lr,loss=float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,p.reshape(-1,1)))),accuracy=float(accuracy_score(subset.target,pred)),malignant_sensitivity=float(tp/16),specificity=float(tn/16),probability_min=float(p.min()),probability_max=float(p.max()))
            history.append(row)
            print('OVERFIT_CHECK',json.dumps(row),flush=True)
        if step < 300:
            model.train_on_batch(x,y)
    pd.DataFrame(history).to_csv(out/f'history_lr{tag}.csv',index=False)
    final=subset[['image_id','label','target','group']].copy()
    final['malignant_probability']=p
    final['prediction']=pred
    final.to_csv(out/f'predictions_lr{tag}.csv',index=False)
    summary.append(dict(**history[-1],initial_weights_sha256=init_hash,training_only=True))
assert len(set(init_hashes)) == 1
pd.DataFrame(summary).to_csv(out/'diagnostics_results.csv',index=False)
with zipfile.ZipFile('/kaggle/working/binary_diagnostics.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in out.iterdir(): z.write(f,f.name)
print('DIAGNOSTICS_COMPLETE',json.dumps(summary),flush=True)
