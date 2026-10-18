"""Check saved head predictions, data separation and feature collapse."""
from features import *
from collections import Counter, defaultdict
from train import fit, metrics

def main():
    report=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
    preds=json.loads((ROOT/'oof_predictions.json').read_text(encoding='utf8'))
    rows=report['training_data'];records=[]
    assert len(rows)==len({r['sha256'] for r in rows})
    for r in rows:
        d=json.loads((ROOT/'feature_cache'/f'{r["sha256"]}_{SPEC_HASH[:12]}.json').read_text(encoding='utf8'))
        assert d['spec_hash']==SPEC_HASH and d['sha256']==r['sha256']
        assert len(d['sampled_timestamps_sec'])==SPEC['samples']
        assert d['sampled_timestamps_sec']==sorted(d['sampled_timestamps_sec'])
        records.append(d)
    X=np.stack([vector(d['states']) for d in records]);y=np.asarray([r['label'] for r in rows]);folds=np.asarray([r['fold'] for r in rows])
    p=np.zeros(len(y))
    for f in sorted(set(folds)):
        tr=folds!=f;te=~tr
        assert set(rows[i]['sha256'] for i in np.flatnonzero(tr)).isdisjoint(rows[i]['sha256'] for i in np.flatnonzero(te))
        p[te]=fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    assert [r['video_id'] for r in rows]==[r['video_id'] for r in preds]
    np.testing.assert_allclose(p,[r['score'] for r in preds],atol=1e-12)
    recomputed=metrics(y,p)
    assert recomputed==report['metrics']
    groups=defaultdict(list)
    for i,d in enumerate(records):groups[tuple(d['states'])].append(i)
    conflicts=[dict(videos=[rows[i]['video_id'] for i in ids],labels=[int(y[i]) for i in ids])
               for ids in groups.values() if len(set(y[ids]))>1]
    quality=dict(validation='PASS: saved OOF predictions independently recomputed; exact duplicates absent; no hash overlap within folds',
                 unique_feature_vectors=len(groups),total_videos=len(rows),
                 conflicting_identical_feature_groups=conflicts,
                 per_cue={key:dict(Counter(d['states'][j] for d in records)) for j,(key,_) in enumerate(CUES)},
                 note='Checks validate the training pipeline, not the truth of Qwen observations or independent-test accuracy.')
    (ROOT/'verification.json').write_text(json.dumps(quality,indent=2),encoding='utf8')
    print(json.dumps(quality,indent=2))
if __name__=='__main__':main()
