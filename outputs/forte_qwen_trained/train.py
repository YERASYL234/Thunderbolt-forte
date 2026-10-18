"""Train a supervised binary head on frozen Qwen observations; no Qwen fine-tuning."""
from features import *
import argparse, html, math, contextlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score

def fit(X,y):
    return LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=20261017).fit(X,y)

def metrics(y,p):
    z=np.asarray(p)>=.5
    return dict(n=len(y),accuracy=float(accuracy_score(y,z)),balanced_accuracy=float(balanced_accuracy_score(y,z)),
                precision=float(precision_score(y,z,zero_division=0)),recall=float(recall_score(y,z,zero_division=0)),
                f1=float(f1_score(y,z,zero_division=0)),roc_auc=float(roc_auc_score(y,p)),
                confusion_matrix=confusion_matrix(y,z,labels=[0,1]).tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu-layers',type=int,default=99);args=ap.parse_args()
    ds=ROOT.parent/'forte_dataset';vs=json.loads((ds/'annotations/videos.json').read_text(encoding='utf8'))
    assert len({v['sha256'] for v in vs})==len(vs), 'Duplicate videos must be grouped before training'
    cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True)
    (ROOT/'feature_spec.json').write_text(json.dumps(dict(**SPEC,spec_hash=SPEC_HASH),indent=2),encoding='utf8')
    records=[]
    with contextlib.ExitStack() as stack:
        model=None
        for i,v in enumerate(vs):
            path=ds/v['path'];assert digest(path)==v['sha256']
            dest=cache/(v['sha256']+'_'+SPEC_HASH[:12]+'.json')
            if dest.exists():
                data=json.loads(dest.read_text(encoding='utf8'));assert data['spec_hash']==SPEC_HASH and data['sha256']==v['sha256']
                vector(data['states'])
            else:
                if model is None:model=stack.enter_context(LocalQwen(ROOT/'training.server.log',args.gpu_layers))
                data=model.extract(path);dest.write_text(json.dumps(data,indent=2),encoding='utf8')
            records.append(data)
            print(f'FEATURES {i+1}/{len(vs)} {v["video_id"]} {data["elapsed_sec"]:.1f}s',flush=True)
    X=np.stack([vector(r['states']) for r in records]);y=np.asarray([int(v['source_verdict']=='violation') for v in vs]);folds=np.asarray([v['provisional_fold'] for v in vs])
    if np.unique(X,axis=0).shape[0]<2:raise ValueError('Qwen produced identical features for every video. Cannot learn a useful classifier.')
    p=np.zeros(len(y));majority=np.zeros(len(y));fold_metrics=[]
    for fold in sorted(set(folds)):
        tr=folds!=fold;te=~tr
        clf=fit(X[tr],y[tr]);p[te]=clf.predict_proba(X[te])[:,1];majority[te]=float(y[tr].mean()>=.5)
        fold_metrics.append(dict(fold=int(fold),train_count=int(tr.sum()),validation_count=int(te.sum()),**metrics(y[te],p[te])))
    final=fit(X,y);scores=final.predict_proba(X)[:,1]
    checkpoint=ROOT/'classifier.npz';np.savez_compressed(checkpoint,coef=final.coef_,intercept=final.intercept_,spec_hash=SPEC_HASH)
    with np.load(checkpoint,allow_pickle=False) as w:
        restored=1/(1+np.exp(-np.clip(X@w['coef'].T+w['intercept'],-50,50)))
    np.testing.assert_allclose(restored.ravel(),scores,atol=1e-12)
    cfg=dict(model_version='forte_qwen_observation_head_v1',trained_component='L2 logistic regression on 48 one-hot Qwen observation features',
             qwen_weights_updated=False,qwen_lora_trained=False,feature_spec_hash=SPEC_HASH,feature_dim=X.shape[1],
             C=1.,class_weight='balanced',decision_threshold=.5,hyperparameter_search=False,
             labels={'0':'no_violation','1':'violation'},final_training_videos=len(vs),event_classes_supported=False)
    preds=[dict(video_id=v['video_id'],fold=int(folds[i]),true_label=int(y[i]),predicted_label=int(p[i]>=.5),score=float(p[i]),correct=bool(y[i]==(p[i]>=.5))) for i,v in enumerate(vs)]
    result=metrics(y,p)
    n=len(y);acc=result['accuracy'];z=1.96;den=1+z*z/n
    center=(acc+z*z/(2*n))/den;half=z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/den
    report=dict(config=cfg,evaluation='5-fold out-of-fold development evaluation; not an independent final test',metrics=result,
                accuracy_wilson_95_interval_descriptive_only=[center-half,center+half],fold_metrics=fold_metrics,
                majority_baseline=metrics(y,majority),training_fit_NOT_generalization=metrics(y,scores),
                feature_extraction_total_seconds=sum(r['elapsed_sec'] for r in records),
                limitations=['Only 40 unique videos. Scenes, actors and scenarios may overlap across folds.',
                             'Same development folds were already used for earlier model comparisons; no final holdout.',
                             '8 sampled frames can miss short actions; 448-pixel input can obscure keys and cash.',
                             'Qwen observations can be wrong; their correctness has not been independently annotated.',
                             'Binary source labels are not independently adjudicated. No legal determination.',
                             'Only the supervised classifier head is trained; Qwen is frozen.',
                             'Scores are uncalibrated; no 95% accuracy guarantee; no class or timecode localization.'],
                training_data=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=int(folds[i]),label=int(y[i])) for i,v in enumerate(vs)])
    for name,obj in [('config.json',cfg),('metrics.json',report),('oof_predictions.json',preds)]:
        (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    rows=''.join(f'<tr><td>{r["video_id"]}</td><td>{r["fold"]}</td><td>{r["true_label"]}</td><td>{r["predicted_label"]}</td><td>{r["score"]:.3f}</td><td>{"✓" if r["correct"] else "Қате"}</td></tr>' for r in preds)
    baselines=[]
    for fn,label in [('metrics.json','ResNet18'),('temporal_metrics.json','R3D18')]:
        old=json.loads((ROOT.parent/'forte_model'/fn).read_text(encoding='utf8'))
        baselines.append((label,old['metrics']))
    comparisons=''.join(f'<tr><td>{label}</td><td>{m["accuracy"]:.1%}</td><td>{m["precision"]:.1%}</td><td>{m["recall"]:.1%}</td><td>{m["f1"]:.1%}</td></tr>' for label,m in baselines+[('Qwen + оқытылған классификатор',result)])
    assessment='Бұл нұсқа көпшілік класын ғана болжаудан да төмен нәтиже берді. Қолдануға жеткілікті сапаға жеткен жоқ.' if result['accuracy']<=report['majority_baseline']['accuracy'] else 'Бұл эксперименттік нәтиже; тәуелсіз финалдық тест қажет.'
    (ROOT/'training_report.html').write_text(f'''<!doctype html><html lang="kk"><meta charset="utf-8"><title>Forte Qwen — оқыту нәтижесі</title>
<style>body{{max-width:1000px;margin:40px auto;padding:20px;font:17px system-ui;color:#172536}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.note{{background:#fff1d1;padding:20px;border-radius:12px}}</style>
<h1>Qwen негізіндегі классификатор оқытылды</h1><p>40 бірегей видео · 21 бұзушылық бар / 19 жоқ · 8 кадр/видео · 16 визуалды белгі.</p>
<p class="note"><b>{assessment}</b></p>
<p>Qwen3-VL-2B салмақтары өзгерген жоқ. Qwen шығарған белгілерден L2 logistic regression классификаторы оқытылды. Бұл Qwen fine-tuning немесе LoRA емес.</p>
<p class="note">5 топ бойынша ішкі тексеру. Әр болжам сол видеоны оқытуда көрмеген классификатордан алынды. Актерлер мен орта ұқсастығы сақталуы мүмкін; тәуелсіз финалдық тест жоқ. 95% дәлдік расталған жоқ.</p>
<h2>Бірдей development fold бойынша салыстыру</h2><table><tr><th>Модель</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{comparisons}</table>
<p>Qwen ROC-AUC: {result['roc_auc']:.3f}. Confusion matrix [TN, FP; FN, TP]: {result['confusion_matrix']}.</p>
<p>Көпшілік класты болжау accuracy: {report['majority_baseline']['accuracy']:.1%}. Оқыту деректеріндегі нәтиже бағалау метрикасы ретінде қолданылмады.</p>
<h2>Шектеулер</h2><p>8 кадр арасындағы әрекет жоғалуы мүмкін. Qwen ұсақ заттарды немесе әрекетті қате сипаттауы мүмкін. Модель бұзушылық түрін, нақты уақытын не нормативті анықтамайды. Score — калибрленген ықтималдық емес.</p>
<h2>Оқытуға кірмеген видеолар бойынша болжамдар</h2><table><tr><th>Видео</th><th>Fold</th><th>Белгі</th><th>Болжам</th><th>Score</th><th>Тексеру</th></tr>{rows}</table>
<p><a href="metrics.json">Толық метрикалар</a> · <a href="README.txt">Іске қосу</a> · <a href="feature_spec.json">Белгілер және prompt</a></p></html>''',encoding='utf8')
    print('TRAINED',json.dumps(result),flush=True)

if __name__=='__main__':main()
