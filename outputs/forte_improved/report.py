from data import *
from search import evaluate
import html,math
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
    vs=videos();y=np.array([int(v['source_verdict']=='violation') for v in vs]);ids=[v['video_id'] for v in vs]
    rows={};metrics={};labels={};validation={}
    def add(key,label,metric_path,pred_path,nested=False):
        r=read(metric_path);pr=read(pred_path);assert [v['video_id'] for v in pr]==ids
        metrics[key]=r['nested_metrics' if nested else 'metrics'];rows[key]=pr;labels[key]=label
        validation[key]=r.get('evaluation','5-fold development')
    old=ROOT.parent/'forte_model'
    add('resnet','ResNet18: бұрынғы модель',old/'metrics.json',old/'oof_predictions.json')
    add('r3d','R3D18: бұрынғы модель',old/'temporal_metrics.json',old/'temporal_oof_predictions.json')
    add('qwen','Qwen2B + классификатор',ROOT.parent/'forte_qwen_trained/metrics.json',ROOT.parent/'forte_qwen_trained/oof_predictions.json')
    add('existing_search','Қолдағы белгілер: nested search',ROOT/'existing_metrics.json',ROOT/'existing_oof_predictions.json',True)
    add('dino_pipeline','DINO және визуалды белгілер: nested search',ROOT/'all_metrics.json',ROOT/'all_oof_predictions.json',True)
    add('finetune','ResNet18 partial fine-tuning v1',ROOT/'finetune_metrics.json',ROOT/'finetune_oof_predictions.json')
    add('finetune_stable','ResNet18 partial fine-tuning v2',ROOT/'finetune_stable_metrics.json',ROOT/'finetune_stable_oof_predictions.json')
    add('finetune_matched','ResNet18: кадрлар тобы сәйкестендірілген v3',ROOT/'finetune_matched_metrics.json',ROOT/'finetune_matched_oof_predictions.json')
    add('window','DINO: таймкодпен оқытылған терезелер',ROOT/'window_metrics.json',ROOT/'window_oof_predictions.json',True)
    for name,a,b,label in [('ensemble','dino_pipeline','finetune','DINO + fine-tuning, 50/50'),('window_r3d','window','r3d','Таймкод моделі + R3D18, 50/50')]:
        p=(np.array([r['score'] for r in rows[a]])+np.array([r['score'] for r in rows[b]]))/2
        metrics[name]=evaluate(y,p);labels[name]=label;validation[name]='Predeclared equal-weight blend of held-out predictions; development only'
        rows[name]=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=v['provisional_fold'],true_label=int(y[i]),score=float(p[i]),threshold=.5,predicted_label=int(p[i]>=.5),correct=bool((p[i]>=.5)==y[i])) for i,v in enumerate(vs)]
    deployable=['r3d','dino_pipeline','finetune','finetune_stable','finetune_matched','ensemble','window','window_r3d']
    selected=max(deployable,key=lambda k:(metrics[k]['balanced_accuracy'],metrics[k].get('macro_f1',metrics[k]['f1'])))
    chosen=metrics[selected];old_acc=metrics['r3d']['accuracy'];improvement=chosen['accuracy']-old_acc
    selection=dict(selected_model=selected,display_name=labels[selected],development_metrics=chosen,
        selection_rule='Highest development balanced accuracy, then macro F1, among deployable candidates',
        warning='Selection uses this small development set; reported winner performance is selection-biased. No independent final test. 95% accuracy is not established.',
        accuracy_change_vs_previous_r3d=improvement,operational_use_recommended=False)
    for name,obj in [('selection.json',selection),('comparison.json',dict(metrics=metrics,labels=labels,validation=validation,selection=selection)),('selected_oof_predictions.json',rows[selected])]:
        (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    # Prioritize reviewing errors and visually similar opposite-label clips. Never change source labels automatically.
    x=np.stack([np.load(ROOT/'feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False)['sequence'][:,1].mean(0) for v in vs])
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);similarity=x@x.T
    queue=[]
    for i,v in enumerate(vs):
        opposite=np.flatnonzero(y!=y[i]);j=int(opposite[np.argmax(similarity[i,opposite])])
        wrong=sum(int(rows[k][i]['predicted_label'])!=y[i] for k in rows)
        queue.append(dict(video_id=v['video_id'],original_label=int(y[i]),selected_prediction=rows[selected][i]['predicted_label'],
            models_disagreeing_with_label=int(wrong),nearest_opposite_label_video=vs[j]['video_id'],cosine_similarity=float(similarity[i,j]),
            candidate_classes=v['class_ids'],review_action='Check full operation, visible event interval and applicable rule; retain original label until adjudicated'))
    queue.sort(key=lambda r:(r['models_disagreeing_with_label'],r['cosine_similarity']),reverse=True)
    (ROOT/'priority_review_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf8')
    leaderboard=read(ROOT/'all_development_leaderboard.json');stress=read(ROOT/'background_stress.json');shortcut=read(ROOT/'duration_shortcut_diagnostic.json')
    top=leaderboard[0]
    table=''.join(f'<tr class="{"best" if k==selected else ""}"><td>{html.escape(labels[k])}</td><td>{m["accuracy"]:.1%}</td><td>{m["balanced_accuracy"]:.1%}</td><td>{m["precision"]:.1%}</td><td>{m["recall"]:.1%}</td><td>{m["f1"]:.1%}</td></tr>' for k,m in metrics.items())
    n=len(y);a=chosen['accuracy'];z=1.96;den=1+z*z/n;c=(a+z*z/(2*n))/den;h=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/den
    review=''.join(f'<tr><td><a href="../forte_dataset/review.html">{r["video_id"]}</a></td><td>{r["original_label"]}</td><td>{r["selected_prediction"]}</td><td>{r["models_disagreeing_with_label"]}</td><td>{r["nearest_opposite_label_video"]}</td></tr>' for r in queue[:12])
    support=read(DS/'annotations/summary.json')['candidate_class_counts']
    supports=''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k,v in sorted(support.items(),key=lambda r:-r[1]))
    outcome=f'Бұрынғы R3D18-мен салыстырғандағы development accuracy өзгерісі: {improvement*100:+.1f} пайыздық пункт.'
    text=f'''<!doctype html><html lang="kk"><meta charset="utf-8"><title>Forte — модельдерді жақсарту</title><style>
body{{font:16px system-ui;line-height:1.55;max-width:1120px;margin:32px auto;padding:20px;color:#172536}}h1{{font-size:32px}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}}th{{background:#eef3f8}}.note{{background:#fff0ca;padding:18px;border-radius:10px}}.best{{background:#e4f4ed}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#eef3f8;border-radius:10px;padding:16px;min-width:160px}}.value{{font-size:30px;font-weight:700}}a{{color:#145db0}}</style>
<h1>Forte: модельдерді қайта оқыту нәтижесі</h1><p>40 бірегей видео · 21 бұзушылық бар / 19 жоқ · жергілікті өңдеу.</p>
<p><b>Таңдалған нұсқа: {html.escape(labels[selected])}.</b> {outcome}</p>
<div class="cards"><div class="card">Accuracy<div class="value">{chosen['accuracy']:.1%}</div></div><div class="card">Balanced accuracy<div class="value">{chosen['balanced_accuracy']:.1%}</div></div><div class="card">Recall<div class="value">{chosen['recall']:.1%}</div></div><div class="card">F1<div class="value">{chosen['f1']:.1%}</div></div></div>
<p class="note"><b>95% дәлдік расталған жоқ.</b> Бұл — бұрын да қолданылған 40 видео бойынша development салыстыру. Nested CV кезінде параметрлер ішкі fold-тарда таңдалды, бірақ осы есептегі бірнеше тәсілдің жеңімпазы development нәтижелерімен таңдалды. Жаңа тәуелсіз финалдық тест жоқ; орта/актер/сценарий ұқсастығы сақталуы мүмкін.</p>
<h2>Не жасалды</h2><ul><li>Ресми DINOv2 ViT-B/14 моделі жүктелді: 24 уақыт нүктесі, толық кадр және жұмыс аймағы, 2×2 жергілікті белгілер, уақыттық өзгерістер.</li><li>90 конфигурация nested CV арқылы салыстырылды; threshold тек ішкі тексеруде таңдалды.</li><li>ResNet18 layer4 және видео head екі рецептпен нақты fine-tuning алды. Алғашқы қабаттар frozen.</li><li>Пайдаланушы таймкодтары тек training видеоларында қолданылған терезелік классификатор үйретілді. Validation таймкодтары болжамға берілген жоқ.</li><li>Ансамбльдер, фон топтары арасындағы тасымалдау және ұзақтыққа тәуелділік тексерілді.</li></ul>
<h2>Модельдерді салыстыру</h2><p>Кестедегі барлық нәтижелер development бағалау. Қарапайым baseline-дар fixed CV, іздеу тәсілдері nested CV арқылы бағаланған.</p><table><tr><th>Модель</th><th>Accuracy</th><th>Balanced acc.</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{table}</table>
<p>Таңдалған модель confusion matrix [TN, FP; FN, TP]: {chosen['confusion_matrix']}.</p>
<p>Accuracy үшін сипаттамалық Wilson аралығы: {c-h:.1%}–{c+h:.1%}. Бұл аралық model-selection bias пен ұқсас сценарийлерді есепке алмайды.</p>
<h2>Жоғары метриканы қалай түсіну керек</h2><p>Бір конфигурацияның іздеу кезіндегі ең жоғары ішкі balanced accuracy-і: {top['metrics']['balanced_accuracy']:.1%}. Ол параметр таңдауға пайдаланылған, сондықтан тәуелсіз test метрикасы ретінде берілмейді.</p>
<p>Тек видео ұзақтығын көретін диагностикалық модель accuracy: {shortcut['metrics']['accuracy']:.1%}. Ол іс-әрекетті көрмейді және таңдалатын модельдерге қосылған жоқ.</p>
<p>Фон бойынша екі топ арасындағы exploratory тексеру accuracy: {stress['metrics']['accuracy']:.1%}. Топтар автоматты анықталған; бір бағытта оқытуға тек 8 видео бар. Бұл расталған камера/session holdout емес.</p>
<h2>Қай видеоларды алдымен қайта белгілеу керек</h2><p>Бұл тізім label қате деген қорытынды емес. Толық операцияны, оқиға басталу/аяқталуын және қолданылатын ережені тексеру қажет. Бастапқы белгілер өзгертілген жоқ.</p><table><tr><th>Видео</th><th>Белгі</th><th>Таңдалған болжам</th><th>Белгімен келіспеген модельдер</th><th>Ұқсас, басқа белгісі бар видео</th></tr>{review}</table>
<h2>Класстарға мысал саны</h2><p>Бұл provisional оқиға жазбаларының саны; кейбір видеоларда бірнеше оқиға бар. 14 класқа сенімді классификатор оқытылған жоқ.</p><table><tr><th>Класс</th><th>Жазба саны</th></tr>{supports}</table>
<h2>Іске қосу</h2><p><a href="README.txt">Нұсқаулық</a> · <a href="selection.json">Таңдалған модель</a> · <a href="comparison.json">Толық метрикалар</a> · <a href="priority_review_queue.json">Белгілеу кезегі</a></p>
<pre>.\\run.ps1 -Video 'C:\\path\\new_video.mov' -Output 'C:\\path\\prediction.json'</pre>
<p>Модель түрін --model арқылы таңдауға болады. Болжамдар нормативтік қорытынды емес; score калибрленбеген. Терезе уақыты — тек тексеруге арналған кандидат, расталған оқиға шекарасы емес.</p></html>'''
    text=text.replace('екі рецептпен','үш нұсқамен')
    best_inner=read(ROOT/'all_metrics.json')['final_selection']['inner_metrics']
    extra=f'<p class="note">DINO-ның таңдалған конфигурациясының параметр іріктеу кезіндегі accuracy: {best_inner["accuracy"]:.1%}, F1: {best_inner["f1"]:.1%}, recall: {best_inner["recall"]:.1%}. False positive: {best_inner["confusion_matrix"][0][1]}/19 қалыпты видео. Бұл сандар параметр таңдауға қолданылған, тәуелсіз бағалау емес. Ол нұсқа <code>-Model dino_pipeline</code> арқылы қолжетімді.</p>'
    text=text.replace('<h2>Жоғары метриканы қалай түсіну керек</h2>','<h2>Жоғары метриканы қалай түсіну керек</h2>'+extra)
    (ROOT/'report.html').write_text(text,encoding='utf8')
    print(json.dumps(selection,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
