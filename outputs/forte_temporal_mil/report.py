from common import *
import html,collections

def main():
    result=read(ROOT/'metrics.json');m=result['development_metrics'];rows=read(ROOT/'oof_predictions.json');verification=read(ROOT/'verification.json')
    events=read(DS/'annotations/events.json');taxonomy={t['class_id']:t for t in read(DS/'annotations/taxonomy.json')};vs={v['video_id']:v for v in videos()}
    counts=collections.Counter(e['class_id'] for e in events)
    categories=[]
    for key,n in counts.most_common():
        matched=[e for e in events if e['class_id']==key]
        categories.append(dict(class_id=key,name=taxonomy[key]['name_kk'],events=n,videos=len({e['video_id'] for e in matched}),
            verified_intervals=sum(e.get('start_sec') is not None and e.get('end_sec') is not None and e.get('rule_status')=='verified' for e in matched),
            classifier_status='not_trained_insufficient_verified_event_labels'))
    save(ROOT/'event_class_support.json',categories)
    # This diagnostic is computed after training, never used for selection or inference.
    hits=0;total=0;point_rows=[]
    byid={r['video_id']:r for r in rows}
    for e in events:
        t=e.get('source_timecode_sec')
        if not isinstance(t,(int,float)):continue
        r=byid[e['video_id']];hit=any(a<=t<=b and p>=.5 for (a,b),p in zip(r['window_bounds_sec'],r['window_scores']))
        hits+=hit;total+=1;point_rows.append(dict(event_id=e['event_id'],hit=bool(hit)))
    point=dict(hits=hits,total=total,weak_point_coverage=hits/total if total else None,
        interpretation='Post-training diagnostic on provisional held-out point labels, NOT event recall/AP/F1. No gold boundaries or false-positive interval labels.',events=point_rows)
    save(ROOT/'weak_point_diagnostic.json',point)
    pct=lambda v:f'{100*v:.1f}%'
    cards=''.join(f'<div class="stat"><b>{pct(m[k])}</b><span>{title}</span></div>' for k,title in [('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy'),('f1','F1'),('recall','Recall')])
    cards+=f'<div class="stat"><b>{m["roc_auc"]:.3f}</b><span>ROC AUC</span></div>'
    cm=m['confusion_matrix'];foldrows=''.join(f'<tr><td>{r["outer_fold"]+1}</td><td>{r["selected_epochs"]}</td><td>{pct(r["metrics"]["accuracy"])}</td><td>{pct(r["metrics"]["f1"])}</td></tr>' for r in read(ROOT/'fold_audit.json'))
    classrows=''.join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["events"]}</td><td>{c["videos"]}</td><td>{c["verified_intervals"]}</td></tr>' for c in categories)
    detail=''
    for r in sorted(rows,key=lambda r:(r['correct'],r['video_id'])):
        v=vs[r['video_id']];width=700;height=110;duration=v['duration_sec']
        pts=' '.join(f'{((a+b)/2/duration)*width:.2f},{height-p*height:.2f}' for (a,b),p in zip(r['window_bounds_sec'],r['window_scores']))
        marks=''.join(f'<line x1="{t/duration*width:.2f}" x2="{t/duration*width:.2f}" y1="0" y2="110" stroke="#c76c14" stroke-dasharray="3,4"/>' for t in sorted({e['source_timecode_sec'] for e in events if e['video_id']==r['video_id'] and isinstance(e.get('source_timecode_sec'),(int,float))}))
        true='Бұзушылық' if r['true_label'] else 'Бұзушылық жоқ';pred='Бұзушылық' if r['predicted_label'] else 'Бұзушылық жоқ'
        intervals=', '.join(f'{e["start_sec"]:.1f}–{e["end_sec"]:.1f} с' for e in r['candidate_events']) or 'Шектен асқан үзінді жоқ'
        detail+=f'<details><summary>{r["video_id"]} · {"Дұрыс" if r["correct"] else "Қате"} · score {r["score"]:.3f}</summary><p>Бастапқы белгі: {true}. Болжам: {pred}. Fold: {r["fold"]+1}.</p><svg viewBox="0 0 700 110" role="img" aria-label="Уақыт бойынша күмәнділік бағасы"><rect width="700" height="110" fill="#eff4f8"/><line x1="0" x2="700" y1="55" y2="55" stroke="#8795a3" stroke-dasharray="4,4"/>{marks}<polyline points="{pts}" fill="none" stroke="#126b80" stroke-width="2"/></svg><p>0 → {duration:.1f} с. Көк: score; сұр: 0.5 шегі; қызғылт: бастапқы таймкод (тек салыстыру).</p><p>Кандидат аралықтар: {intervals}. Типі: анықталмаған күмәнді әрекет.</p><video controls preload="none" src="../forte_dataset/{v["path"]}"></video></details>'
    old=read(ROOT.parent/'forte_improved/selection.json')['development_metrics']
    prior_path=ROOT/'experiments/min30_checkpoint_selection/metrics.json'
    prior_note=''
    if prior_path.exists():
        prior=read(prior_path)['development_metrics']
        prior_note=f'<p>Алғашқы іске қосуда checkpoint таңдауы 30-epoch-тан басталған: accuracy {pct(prior["accuracy"])}, F1 {pct(prior["f1"])}. Ішкі validation loss ертерек ең жақсы мәнге жеткендіктен, early stopping түзетілді: best checkpoint 1-epoch-тан бастап сақталады. Алғашқы нәтиже <a href="experiments/min30_checkpoint_selection/metrics.json">архивте</a>. Бұл түзету де сол development жиынында тексерілген.</p>'
    page=f'''<!doctype html><html lang="kk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forte — DINOv2 + Temporal MIL</title><style>
    body{{font:16px system-ui;color:#152a3b;background:#f5f7fb;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}h1{{font-size:34px}}h2{{margin-top:36px}}.stats{{display:flex;flex-wrap:wrap;gap:12px}}.stat{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;min-width:140px}}.stat b{{display:block;font-size:30px}}.stat span{{color:#596d7e}}.notice{{background:#fff0ce;padding:18px;border-radius:10px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border-bottom:1px solid #dce3eb;text-align:left;padding:12px}}details{{background:white;margin:12px 0;padding:16px;border-radius:10px}}summary{{cursor:pointer;font-weight:600}}svg{{width:100%;max-width:850px}}video{{width:100%;max-height:440px;background:#111}}a{{color:#006b87}}code{{background:#e7edf3;padding:2px 4px}}img{{max-width:100%}}</style><main>
    <p>40 видео · 21 positive / 19 negative · тек жергілікті өңдеу</p><h1>DINOv2 ViT-S/14 + Temporal MIL</h1>
    <p>Frozen backbone, 2 FPS, full frame + desk crop, 4 секундтық 1D-CNN, MIL pooling. Оқыту аяқталды.</p><div class="stats">{cards}</div>
    <p class="notice">Бұл — бастапқы видео бойынша grouped 5-fold DEVELOPMENT бағалау. Адамдар/сессиялар бойынша тәуелсіздік расталмаған, бөлек final test жоқ. Осы 40 видео алдыңғы эксперименттерде де қолданылған. 95% дәлдік расталған жоқ. Терезе score-лары калибрленген ықтималдық емес.</p>
    <h2>Нәтиже</h2><p>Бұрынғы R3D18: accuracy {pct(old['accuracy'])}, F1 {pct(old['f1'])}. Жаңа Temporal MIL: accuracy {pct(m['accuracy'])}, F1 {pct(m['f1'])}. Бұл салыстыру сол шағын development жиынында жасалды.</p>{prior_note}
    <table><tr><th></th><th>Болжам: қалыпты</th><th>Болжам: бұзушылық</th></tr><tr><td>Бастапқы белгі: қалыпты</td><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr><tr><td>Бастапқы белгі: бұзушылық</td><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>
    <h2>Қалай оқытылды</h2><p>{verification['total_sampled_frames']} уақыттық кадр, {verification['embedding_views']} көрініс embedding-і. Әр видео <code>embeddings/X-xxx.npy</code> файлына [T, 2, 384] пішімінде сақталған. Уақыт белгілері қасындағы JSON файлында.</p>
    <p>8 кадр = 4 секунд; қадам 2 кадр = 1 секунд. 1D-CNN: 768→64→32, kernel 3, GELU, dropout 0.3. MIL үзінді logits-терінің ең жоғары 10%-ын орташалайды. AdamW, learning rate 0.001, weight decay 0.01, batch 4 видео.</p>
    <p>Loss = video BCE + 0.5 × point BCE + 0.1 × smoothness. Point: таймкодтан ±1 секундтағы үзінді орталықтары; шетте ең жақын үзінді. Positive видеода белгісіз бөліктер negative деп белгіленбейді. Бұл терезелер қолмен расталған шекаралар емес.</p>
    <p>Әр сыртқы fold үшін бөлек ішкі validation бар. Ішкі оқыту 30–60 epoch, patience 10, video BCE; ең жақсы checkpoint 1-epoch-тан бастап сақталады. Сондықтан таңдалған epoch 30-дан аз болуы мүмкін. Таңдалған epoch санымен сыртқы train қайта оқытылды. Шек алдын ала 0.5. Соңғы модель барлық 40 видеода {result['final_epochs']} epoch оқытылды; есептегі метрикалар осы соңғы модельдің train accuracy-і емес.</p>
    <table><tr><th>Fold</th><th>Таңдалған epoch</th><th>Accuracy</th><th>F1</th></tr>{foldrows}</table>
    <h2>Үстел crop-ы</h2><p>Барлық кадрға бірдей координаттар: x=0–82%, y=32–72%. Толық кадр да сақталады. Бұл crop overview кадрларында тексерілген; басқа камераға бейімдеу қажет.</p><img src="crop_preview.jpg" alt="Толық кадрлар және үстел crop-тары">
    <h2>Оқиға түрлері</h2><p>27 кандидат оқиға, 14 категория. Расталған уақыт аралықтары мен ереже белгілері жеткіліксіз болғандықтан, жеке event classifier оқытылмады. Детектордың ішкі түрі: <strong>«Типі анықталмаған күмәнді әрекет»</strong>. Оны нақты бұзушылық түрі ретінде көрсетуге болмайды.</p>
    <table><tr><th>Кандидат түр</th><th>Оқиға</th><th>Видео</th><th>Расталған аралық + ереже</th></tr>{classrows}</table>
    <p>Бастапқы таймкодтардың {hits}/{total}-ы шектен асқан кандидат аралыққа түсті. Бұл тек әлсіз point белгілерімен кейінгі салыстыру; event recall немесе event F1 емес.</p>
    <h2>Әр видеоның held-out нәтижесі</h2><p>Алдымен қате болжамдар көрсетілген. Әр графикті сол видеоны оқытуда көрмеген fold моделі шығарған.</p>{detail}
    <h2>Файлдар</h2><p><a href="README.txt">Іске қосу нұсқаулығы</a> · <a href="metrics.json">Метрикалар</a> · <a href="verification.json">Тексерулер</a> · <a href="temporal_mil.pt">Оқытылған temporal модель</a> · <a href="training_plan.json">Оқыту параметрлері</a></p>
    <p>Backbone: <a href="https://github.com/facebookresearch/dinov2">Meta DINOv2 ресми репозиторийі</a>. Модель файлы мен код нұсқасының fingerprint-тері encoder_provenance.json ішінде.</p></main></html>'''
    (ROOT/'report.html').write_text(page,encoding='utf8')
    (ROOT/'README.txt').write_text(f'''FORTE TEMPORAL MIL — trained local experiment

Architecture: frozen DINOv2 ViT-S/14, full frame + fixed desk crop, 2 FPS, 224x224.
Embeddings: embeddings/X-xxx.npy, float32 [T,2,384], JSON timestamps/provenance next to each file.
Temporal network: 4-second ordered windows (8 frames), 1-second stride, Conv1d(768,64,3), GELU, Dropout(.3), Conv1d(64,32,3), GELU, Dropout(.3), time mean, linear head.
MIL: mean of top ceil(10% * number of windows) logits, then sigmoid.
Loss: video weighted BCE + 0.5 positive point BCE + 0.1 adjacent-window sigmoid MSE.
Point labels: weak +-1 second around window centers, nearest-window fallback; unknown positive regions are not negatives.
AdamW lr=.001, weight_decay=.01, batch=4 videos, gradient clip=1.

EVALUATION
Source-video/hash grouped 5-fold, existing provisional folds. NOT actor/session-disjoint.
Inner validation retains best video-BCE checkpoint from epoch 1, running at least 30 and at most 60 epochs, patience 10. Outer training refit uses the selected best epoch count, possibly below 30. Fixed threshold .5.
All data statistics and point supervision use the respective training split only.
Final temporal_mil.pt: trained on all 40 videos for {result['final_epochs']} epochs.
Reported metrics are held-out development predictions, not final model training performance.
Accuracy {pct(m['accuracy'])}; balanced accuracy {pct(m['balanced_accuracy'])}; F1 {pct(m['f1'])}; ROC AUC {m['roc_auc']:.4f}.
No independent final test, no 95% claim, weak source labels, same development set used previously.

EVENT TYPES
No supervised type classifier: 14 provisional types, 27 events, no verified rule+interval labels.
Candidate output type: suspicious_action_untyped / Типі анықталмаған күмәнді әрекет.
Returned intervals are overlapping high-score windows merged into candidates, not exact event boundaries.
Window scores are not calibrated probabilities. Candidate intervals may be emitted even if aggregate video score is below threshold.

RUN IN POWERSHELL (this workspace)
& "{ROOT/'run.ps1'}" -Video "C:\\path\\new_video.mp4" -Output "C:\\path\\prediction.json"
Optional -Python path uses another interpreter with torch, numpy, av, Pillow installed.
Current portable runtime: C:\\Users\\alpys\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe
Dependencies are in ../../work/train_deps, ../../work/pydeps, ../../work/scideps (see requirements.txt).
CPU execution, 4 threads. No video upload or paid API.

REPRODUCE
python extract.py     # all videos, resumable embedding cache
python train.py       # five outer folds with inner early stopping; final model
python verify.py      # invariants and held-out reload inference
python report.py      # report
Source dataset expected at ../forte_dataset. Vendored DINO code and weights are bundled here.
Keep crop/sampling/features unchanged for an existing checkpoint.

FILES
temporal_mil.pt — final temporal weights, train-only scaling, config and training hashes
fold_models/fold_0.pt ... fold_4.pt — out-of-fold model checkpoints
oof_predictions.json — held-out scores and candidate windows
fold_audit.json / training_history.json — splits and per-epoch loss logs
encoder_provenance.json / feature_spec.json — frozen encoder and preprocessing
event_class_support.json — event category evidence count
verification.json — reproducibility checks, not label certification
report.html — results and per-video score plots
''',encoding='utf8')
    names=['temporal_mil.pt','common.py','extract.py','model.py','train.py','predict.py','verify.py','metrics.json','fold_audit.json','oof_predictions.json','feature_spec.json']
    save(ROOT/'SHA256SUMS.json',{name:sha(ROOT/name) for name in names})
    print('REPORT_READY',str(ROOT/'report.html'),flush=True)
if __name__=='__main__':main()
