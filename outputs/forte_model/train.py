from model import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,balanced_accuracy_score
import sklearn,torchvision,collections,platform,html

SEED=20261017
def pipeline():return make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=SEED))
def metrics(y,p):
    pred=(p>=.5).astype(int)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),roc_auc=float(roc_auc_score(y,p)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),n=int(len(y)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default=str(ROOT.parent/'forte_dataset'));ap.add_argument('--download-weights',action='store_true');args=ap.parse_args()
    ds=Path(args.dataset);vs=json.loads((ds/'annotations/videos.json').read_text(encoding='utf8'));assert len({v['sha256'] for v in vs})==len(vs)
    weights=ROOT/'resnet18_imagenet.pth'
    if not weights.exists():
        if not args.download_weights:raise ValueError('Missing local encoder weights; use --download-weights once with network permission.')
        from torchvision.models import ResNet18_Weights
        torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url,str(weights),hash_prefix='f37072fd',progress=False)
    cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True);m=encoder(weights);xs=[];timings=[];sampling=[]
    for i,v in enumerate(vs):
        path=ds/v['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==v['sha256']
        f=cache/(v['sha256']+'_resnet18_48_fullresize_v1.npz')
        if f.exists():
            data=np.load(f,allow_pickle=False);x=data['features'];meta=json.loads(str(data['metadata']))
        else:
            x,meta=extract(path,m,48);np.savez_compressed(f,features=x,metadata=json.dumps(meta))
        xs.append(x);timings.append(meta['elapsed_sec']);sampling.append(dict(video_id=v['video_id'],**meta))
        print(f'{i+1}/{len(vs)} {v["video_id"]} extracted {meta["elapsed_sec"]:.1f}s',flush=True)
    X=np.stack(xs);y=np.array([v['source_verdict']=='violation' for v in vs],dtype=int);folds=np.array([v['provisional_fold'] for v in vs]);p=np.zeros(len(y));pd=np.zeros(len(y));pm=np.zeros(len(y));fold_metrics=[]
    duration=np.array([v['duration_sec'] for v in vs]).reshape(-1,1)
    for fold in range(5):
        tr=folds!=fold;te=~tr;assert set(np.where(tr)[0]).isdisjoint(np.where(te)[0])
        clf=pipeline().fit(X[tr],y[tr]);p[te]=clf.predict_proba(X[te])[:,1]
        dc=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED)).fit(duration[tr],y[tr]);pd[te]=dc.predict_proba(duration[te])[:,1]
        pm[te]=int(y[tr].mean()>=.5);fold_metrics.append(dict(fold=fold,**metrics(y[te],p[te])))
    result=metrics(y,p);print('OUT_OF_FOLD',json.dumps(result),flush=True)
    final=pipeline().fit(X,y);sc=final[0];lr=final[1]
    np.savez_compressed(ROOT/'classifier.npz',mean=sc.mean_,scale=sc.scale_,coef=lr.coef_,intercept=lr.intercept_)
    np.testing.assert_allclose([score(x,np.load(ROOT/'classifier.npz',allow_pickle=False)) for x in X],final.predict_proba(X)[:,1],rtol=1e-6,atol=1e-7)
    cfg=dict(model_version='forte_binary_resnet18_v1',encoder='torchvision ResNet18 IMAGENET1K_V1 frozen',trained_component='L2 logistic regression video-level readout',samples_per_video=48,preprocessing='Full frame resized to 224x224; ImageNet normalization',pooling=['mean','std','max','mean_absolute_adjacent_change'],feature_dim=int(X.shape[1]),decision_threshold=.5,threshold_tuned_on_evaluation=False,C=.01,class_weight='balanced',seed=SEED,final_training_videos=len(vs),labels={'0':'no_violation','1':'violation'},event_model=False,encoder_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    report=dict(config=cfg,evaluation='5-fold out-of-fold development evaluation. Existing deduplicated provisional folds; NOT independent scene/session/actor holdout.',metrics=result,fold_metrics=fold_metrics,majority_baseline=metrics(y,pm),duration_only_diagnostic=metrics(y,pd),training_fit_metrics_NOT_generalization=metrics(y,final.predict_proba(X)[:,1]),feature_extraction_seconds=dict(mean=float(np.mean(timings)),max=float(max(timings)),total=float(sum(timings))),versions=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,sklearn=sklearn.__version__,av=av.__version__,numpy=np.__version__),limitations=['Only 40 unique source videos; no independent final test.','Scene, actor and scenario similarity remains across folds.','48 uniform frames can miss brief events.','Frozen image encoder and pooled changes do not establish semantic or causal understanding.','Binary source labels retained; event annotations not used.','No event classes, timecodes, evidence selection or legal grounding.','Classifier scores are not calibrated probabilities.','No 95% accuracy guarantee; no hyperparameter search performed.'],training_data=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=v['provisional_fold'],label=v['source_verdict']) for v in vs])
    predictions=[dict(video_id=v['video_id'],fold=int(folds[i]),true_label=int(y[i]),predicted_label=int(p[i]>=.5),score=float(p[i]),correct=bool(y[i]==(p[i]>=.5)),duration_sec=v['duration_sec']) for i,v in enumerate(vs)]
    for name,obj in [('config.json',cfg),('metrics.json',report),('oof_predictions.json',predictions),('sampling_audit.json',sampling)]: (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    table=''.join(f'<tr><td>{r["video_id"]}</td><td>{r["fold"]}</td><td>{r["true_label"]}</td><td>{r["predicted_label"]}</td><td>{r["score"]:.3f}</td><td>{"✓" if r["correct"] else "Қате"}</td></tr>' for r in predictions)
    (ROOT/'training_report.html').write_text(f'<!doctype html><html lang="kk"><meta charset="utf-8"><style>body{{max-width:950px;margin:40px auto;font:17px system-ui;color:#172536;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff1d1;padding:20px}}</style><h1>ForteBank: бірінші оқытылған модель</h1><p>40 бірегей видео · 21 оң / 19 теріс · 48 кадр/видео · CPU</p><p>ResNet18 мұздатылған визуалды белгілерінен бинарлық классификатор оқытылды. ResNet салмақтары қайта үйретілген жоқ.</p><p class="note">Бұл — 5 топ бойынша ішкі тексеру. Әр видеоның бағасы сол видеоны көрмеген классификатордан алынды, бірақ орта мен сценарий ұқсастығы сақталуы мүмкін. Тәуелсіз финалдық тест емес.</p><h2>Нәтиже</h2><p>Accuracy: <b>{result["accuracy"]:.1%}</b> · Precision: {result["precision"]:.1%} · Recall: {result["recall"]:.1%} · F1: {result["f1"]:.1%}</p><p>Confusion matrix [TN, FP; FN, TP]: {result["confusion_matrix"]}</p><p>Тек ұзақтыққа сүйенген диагностикалық модель accuracy: {report["duration_only_diagnostic"]["accuracy"]:.1%}. Бұл көрсеткіш датасеттегі кездейсоқ тәуелділіктерді тексеруге арналған.</p><p>Модель оқиға түрін, таймкодты немесе нормативті бермейді. 95% нәтиже расталған жоқ.</p><h2>Оқытуға кірмеген видеолар бойынша болжамдар</h2><p>Әр жол өз fold-ының тексеру нәтижесі; 1 = бұзушылық бар, 0 = жоқ.</p><table><tr><th>Видео</th><th>Fold</th><th>Бастапқы белгі</th><th>Болжам</th><th>Score</th><th>Дұрыс па?</th></tr>{table}</table><p>Модель: <a href="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html">Torchvision ResNet18</a>. Классификатор: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn LogisticRegression</a>.</p>',encoding='utf8')
    print('SAVED',str(ROOT),flush=True)
if __name__=='__main__':main()
