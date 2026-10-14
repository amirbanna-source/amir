"""Independently check downloaded binary run evidence; never load test images."""
from pathlib import Path
import io, json, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix)

def verify(archive_path, manifest):
    with zipfile.ZipFile(archive_path) as z:
        assert z.testzip() is None, 'Corrupt archive'
        names = z.namelist()
        metric_name = next(n for n in names if n.endswith('_metrics.json'))
        run = metric_name.removesuffix('_metrics.json')
        metrics = json.loads(z.read(metric_name))
        assert metrics['test_evaluated'] is False and metrics['threshold'] == .5
        with zipfile.ZipFile(io.BytesIO(z.read(run+'.keras'))) as model:
            assert model.testzip() is None
        predictions = pd.read_csv(io.BytesIO(z.read(run+'_validation_predictions.csv')))
        reference = manifest[manifest.split == 'val'].set_index('image_id')
        assert len(predictions) == 329 and predictions.image_id.is_unique
        assert set(predictions.image_id) == set(reference.index)
        ref = reference.loc[predictions.image_id]
        assert (predictions.label.to_numpy() == ref.label.to_numpy()).all()
        assert (predictions.original_label.to_numpy() == ref.original_label.to_numpy()).all()
        y = ref.target.to_numpy()
        p = predictions.malignant_probability.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        pred = (p >= .5).astype(int)
        assert np.array_equal(pred, predictions.prediction.to_numpy())
        tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
        recalculated = dict(accuracy=accuracy_score(y,pred),
            macro_f1=f1_score(y,pred,labels=[0,1],average='macro',zero_division=0),
            balanced_accuracy=balanced_accuracy_score(y,pred),
            malignant_sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
            roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p))
        for key,value in recalculated.items():
            assert np.isclose(value,metrics[key],rtol=1e-7,atol=1e-8),(run,key)
        matrix = pd.read_csv(io.BytesIO(z.read(run+'_confusion_matrix.csv')),index_col=0)
        assert np.array_equal(matrix.to_numpy(), [[tn,fp],[fn,tp]])
        history = pd.read_csv(io.BytesIO(z.read(run+'_history.csv')))
        assert len(history) == metrics['epochs_trained']
        assert history.val_loss.argmin()+1 == metrics['selected_epoch']
        assert np.isclose(history.val_loss.min(),metrics['validation_loss'])
        weights = np.load(io.BytesIO(z.read(f"initial_weights_seed{metrics['seed']}.npz")))
        from hashlib import sha256
        digest = sha256(b''.join(weights[f'arr_{i}'].tobytes()
            for i in range(len(weights.files)))).hexdigest()
        assert digest == metrics['initial_weights_sha256']
        return metrics

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('archives',nargs='+',type=Path)
    args=parser.parse_args()
    manifest=pd.read_csv(args.manifest)
    results=[verify(p,manifest) for p in args.archives]
    for row in results: print(json.dumps(row))
    assert len({r['seed'] for r in results}) == len(results)
    print('Verified run count:',len(results))
