from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pc(v):return f'{100*v:.1f}%'
def metric(k,v):return f'{v:.3f}' if k=='roc_auc' else pc(v)
def esc(v):return html.escape(str(v))
def main():
    test=read(ROOT/'test_metrics.json');nested=read(ROOT/'nested_metrics.json');old=read(ROOT.parent/'forte_transfer30/test_metrics.json')
    board=read(ROOT/'development_leaderboard.json');pred=read(ROOT/'test_predictions.json');split=read(ROOT/'split.json')
    sel=test['selection'];new=test['metrics'];before=old['metrics'];cv=nested['metrics']
    audit=read(ROOT/'feature_audit.json') if (ROOT/'feature_audit.json').exists() else None
    diagnostic=''
    if audit:
        groups=audit['feature_collision_diagnostic']['qwen4_summary']['conflicting_groups']
        largest=max(groups,key=lambda g:len(g['video_ids'])) if groups else None
        if largest:
            diagnostic=f'<div class="notice">Qwen белгілерінің шектеуі: {len(largest["video_ids"])} оқу видеосына бірдей feature вектор шыққан. Олардың {largest["positive_count"]}-інде бастапқы белгі бойынша бұзушылық бар, {largest["negative_count"]}-і қалыпты. Тек осы Qwen белгілеріне сүйенген классификатор оларды бір-бірінен ажырата алмайды. Бұл датасеттің міндетті түрде қате екенін білдірмейді. <a href="feature_audit.json">Толық диагностика</a>.</div>'
    rows=''.join(f'<tr><td>{label}</td><td>{metric(k,before[k])}</td><td><b>{metric(k,new[k])}</b></td></tr>' for k,label in [('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy'),('macro_f1','Macro F1'),('f1','F1 · бұзушылық'),('recall','Recall · бұзушылық'),('specificity','Specificity · қалыпты видеоны тану'),('roc_auc','ROC AUC')])
    diagnostics=''.join(f'<tr><td>{esc(r["candidate"]["name"])}</td><td>{pc(r["metrics"]["balanced_accuracy"])}</td><td>{pc(r["metrics"]["recall"])}</td><td>{pc(r["metrics"]["specificity"])}</td><td>{r["threshold"]}</td></tr>' for r in board)
    predictions=''.join(f'<tr><td>{r["video_id"]}</td><td>{"Бұзушылық" if r["true_label"] else "Қалыпты"}</td><td>{"Күмәнді" if r["predicted_label"] else "Белгі табылмады"}</td><td>{r["score"]:.3f}</td><td>{"✓" if r["correct"] else "Қате"}</td></tr>' for r in pred)
    cm=new['confusion_matrix'];low,high=test['accuracy_95_wilson_interval']
    text=f'''<!doctype html><html lang="kk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forte · Qwen салыстыруы</title><style>section{{overflow-x:auto}}
    body{{font:16px/1.65 Segoe UI,Arial,sans-serif;background:#f5f6fa;color:#203044;margin:0}}main{{max-width:1040px;margin:38px auto;padding:24px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px;margin-top:0}}.tag{{color:#a43158;font-size:13px;font-weight:700;letter-spacing:1px}}section{{background:white;border:1px solid #e1e5ed;border-radius:16px;padding:26px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #e3e8ef}}th{{background:#f4f6fa}}.notice{{background:#fff3dc;border-left:4px solid #c08a26;padding:16px;border-radius:6px}}.cards{{display:flex;gap:20px;flex-wrap:wrap}}.cards div{{flex:1;min-width:170px;background:#f3f5f9;padding:18px;border-radius:10px}}.cards b{{display:block;font-size:31px;color:#a43158}}small{{color:#687589}}a{{color:#a43158}}details{{overflow-x:auto}}code{{background:#f0f2f6;padding:2px 5px}}</style><main>
    <div class="tag">FORTE VIDEO LAB · 30 TRAIN / 10 TEST</div><h1>Qwen қосылған жаңа эксперимент</h1>
    <p>Таңдалған нұсқа: <strong>{esc(sel['candidate']['name'])}</strong>. Шешім шегі: <b>{sel['threshold']}</b>. Модель мен шек тек 30 оқу видеосындағы тексеру арқылы таңдалды.</p>
    <div class="cards"><div>10 видеодағы Accuracy<b>{pc(new['accuracy'])}</b></div><div>Қалыпты видеоны тану<b>{pc(new['specificity'])}</b></div><div>Бұзушылықты табу<b>{pc(new['recall'])}</b></div></div>
    <section><h2>Бұрынғы нұсқамен салыстыру</h2><p>Екі нәтиже де бірдей 10 видеоға қатысты. Жаңа нұсқаның параметрлері тестке дейін бекітілді.</p><table><thead><tr><th>Метрика</th><th>Бұрынғы модель</th><th>Жаңа таңдалған модель</th></tr></thead><tbody>{rows}</tbody></table>
    <p>Жаңа модель: {cm[1][1]} дұрыс дабыл, {cm[0][1]} жалған дабыл, {cm[1][0]} өткізіп алған бұзушылық, {cm[0][0]} дұрыс танылған қалыпты видео.</p>
    <div class="notice">Бұл 10 видео осы оқытуға кірмеді, бірақ бұрынғы эксперименттерде қолданылған. Сондықтан бұл жаңа тәуелсіз тест емес. 10 видеода бір қате Accuracy-ді 10 пайыздық тармаққа өзгертеді. Сипаттамалық 95% Wilson аралығы: {pc(low)}–{pc(high)}.</div></section>
    <section><h2>30 видеоның ішіндегі тексеру</h2><p>5 сыртқы fold × 4 ішкі fold: әр сыртқы видеоны бағалағанда ол классификаторды оқытуға, модельді таңдауға және шекті таңдауға қатыспайды.</p>
    <p>Nested development Accuracy: <b>{pc(cv['accuracy'])}</b>; Balanced accuracy: <b>{pc(cv['balanced_accuracy'])}</b>; Macro F1: <b>{pc(cv['macro_f1'])}</b>. Бұл оқыту дерегіндегі зерттеу бағасы; камералар мен адамдардың тәуелсіздігі қамтамасыз етілмеген.</p>
    <details><summary>{len(board)} нұсқаның таңдау кестесі</summary><p class="notice">Төмендегі сандар модельді таңдауға қолданылған CV. Оларды соңғы тест дәлдігі деп көрсетуге болмайды.</p><table><tr><th>Нұсқа</th><th>Balanced accuracy</th><th>Recall</th><th>Specificity</th><th>Шек</th></tr>{diagnostics}</table></details></section>
    <section><h2>Не өзгерді</h2><p>Qwen3-VL-4B Q4 әр видеодан алты сәтті қарайды: толық кадр және үстелдің жақын көрінісі. Әр сәт бойынша тоғыз көрінетін белгі шығарады. Уақыт реті, өзгеріс және бірге байқалған белгілерден шағын классификатор оқытылады. Qwen салмақтары өзгертілмейді; бұл Qwen-нің толық fine-tuning-і емес.</p>
    <p>Qwen2B, DINOv2, R3D18, ResNet18 және комбинация да сол 30 видео ішінде салыстырылды. Файл аты, видео нөмірі, бастапқы түсініктеме және оқиғаның белгіленген уақыты классификатор кірісіне берілмейді. Видеолар сыртқы сервиске жіберілмейді.</p>{diagnostic}
    <p>Сайт оқиға тобын, уақытын, кадрын және JSON-ды сақтайды. Оқиға түрі мен нақты уақытқа жауап беретін бөлек DINO моделі бұрынғы қалпында. Жоғарыдағы binary метрикалар оқиға түрі немесе уақыт дәлдігін дәлелдемейді. Qwen байқаулары да тексерілмеген модель нәтижелері.</p></section>
    <section><h2>Тестке қалған 10 видео</h2><p>{', '.join(v['video_id'] for v in split['test'])}</p><table><tr><th>Видео</th><th>Бастапқы белгі</th><th>Болжам</th><th>Ұпай</th><th>Нәтиже</th></tr>{predictions}</table></section>
    <section><h2>Тексерілетін файлдар</h2><p><a href="test_metrics.json">Тест метрикалары</a> · <a href="nested_metrics.json">Nested тексеру</a> · <a href="selection_locked.json">Тестке дейінгі бекіту</a> · <a href="training_plan.json">Эксперимент жоспары</a> · <a href="split.json">30/10 бөлінуі</a></p><p><a href="http://127.0.0.1:8765/">Видео тексеру сайтын ашу →</a></p><small>Модель көзі: <a href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF">Qwen ресми GGUF репозиторийі</a>. Жүктелген файлдардың SHA-256 мәндері ресми LFS мәндерімен тексерілді.</small></section></main></html>'''
    (ROOT/'report.html').write_text(text,encoding='utf8')
    print('REPORT_READY',ROOT/'report.html')
if __name__=='__main__':main()
