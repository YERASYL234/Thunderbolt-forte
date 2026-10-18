"""Loopback-only transfer-learning test UI; uploads are processed locally."""
from pathlib import Path
import sys,json,time,uuid,threading,hashlib,shutil,urllib.parse,argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent.parent
sys.path.insert(0,str(ROOT.parent/'forte_model'))
import temporal
import numpy as np
import av
from evidence import build_evidence
ACTIVE=json.loads((ROOT/'active_model.json').read_text(encoding='utf8')) if (ROOT/'active_model.json').exists() else {'directory':'forte_transfer30'}
if ACTIVE['directory'] not in ['forte_transfer30','forte_qwen30','forte_qwen_plus30']:raise ValueError('Unknown active model')
MODEL_DIR=ROOT.parent/ACTIVE['directory']
sys.path.insert(0,str(MODEL_DIR))
if ACTIVE['directory'] in ['forte_qwen30','forte_qwen_plus30']:
    sys.path.insert(0,str(ROOT.parent/'forte_qwen30'))
    from predict import Predictor as QwenPredictor,display_name
    def Predictor():return QwenPredictor(model_dir=MODEL_DIR)
else:from predict30 import Predictor,display_name
RUNTIME=BASE/'work/r3d_web_runtime';RUNTIME.mkdir(parents=True,exist_ok=True)
VIDEOS=json.loads((ROOT.parent/'forte_dataset/annotations/videos.json').read_text(encoding='utf8'))
KNOWN={v['sha256']:v['video_id'] for v in VIDEOS}
SPLIT=json.loads((MODEL_DIR/'split.json').read_text(encoding='utf8'))
MODEL_METRICS=json.loads((MODEL_DIR/'test_metrics.json').read_text(encoding='utf8'))['metrics']
SELECTION=json.loads((MODEL_DIR/'selection_locked.json').read_text(encoding='utf8'))['selection']
MODEL_NAME=display_name(SELECTION)
CM=MODEL_METRICS['confusion_matrix']
PERFORMANCE_NOTE=f"Тест: {CM[1][1]} дұрыс дабыл, {CM[0][1]} жалған дабыл, {CM[1][0]} өткізіп алған бұзушылық, {CM[0][0]} дұрыс танылған қалыпты видео."
TRAIN_HASHES={v['sha256'] for v in SPLIT['train']}
TEST_HASHES={v['sha256'] for v in SPLIT['test']}
assert len(TRAIN_HASHES)==30 and len(TEST_HASHES)==10 and TRAIN_HASHES.isdisjoint(TEST_HASHES)
DEMO=ROOT.parent/'forte_dataset/videos/X-026.mov'
POOL=ThreadPoolExecutor(max_workers=1);LOCK=threading.Lock();JOBS={};BUSY=False;ENCODER=None
TOKEN=uuid.uuid4().hex;MAX_BYTES=512*1024*1024

def update(key,**data):
    with LOCK:JOBS[key].update(data)

def process(key,path,name):
    global BUSY,ENCODER
    try:
        start=time.monotonic();update(key,status='processing',stage='Видеоны тексеру')
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
        with av.open(str(path)) as c:
            if not c.streams.video:raise ValueError('Видеотрек табылмады.')
            s=c.streams.video[0]
            duration=float(s.duration*s.time_base) if s.duration else float(c.duration or 0)/1e6
            if not 0<duration<=600:raise ValueError('Ұзақтығы 10 минуттан аспайтын видео таңдаңыз.')
        if ENCODER is None:
            update(key,stage='30 видеода оқытылған модельді жүктеу');ENCODER=Predictor()
        update(key,stage='Видео үзінділерін талдау')
        prediction=ENCODER.predict(path,progress=lambda stage:update(key,stage=stage))
        p=prediction['score'];threshold=prediction['threshold']
        update(key,stage='Оқиға белгілері мен кадрларды дайындау')
        event_analysis=prediction['event_analysis']
        evidence=build_evidence(path,KNOWN.get(digest.hexdigest()),prediction['review_timestamps_sec'],duration,event_analysis)
        qwen=prediction.get('qwen_observations') or {}
        evidence['qwen_cards']=[dict(timestamp_sec=view['timestamp_sec'],source='qwen_observation',
            title='Qwen қараған сәт',events=[],observations=qwen['observations'][f'f{i}'],frames=view['frames'])
            for i,view in enumerate(qwen.get('evidence_views',[]))]
        events=[dict(event,frames=card['frames']) for card in evidence['cards'] if card['source']=='event_window_model' for event in card['events']]
        result=dict(schema_version=2,model=MODEL_NAME,model_version=SPLIT['model_version'],filename=name,score=p,threshold=threshold,
            prediction='violation' if p>=threshold else 'no_violation',duration_sec=duration,processing_sec=time.monotonic()-start,
            seen_in_training=digest.hexdigest() in TRAIN_HASHES,
            training_video_id=KNOWN.get(digest.hexdigest()) if digest.hexdigest() in TRAIN_HASHES else None,
            dataset_video_id=KNOWN.get(digest.hexdigest()),reserved_test_video=digest.hexdigest() in TEST_HASHES,
            training_video_count=30,reserved_test_count=10,
            score_is_calibrated_probability=False,event_localization_supported=True,event_class_supported=True,
            localization_status='candidate_sampled_windows_not_verified_intervals',
            event_classification_status=event_analysis['status'],event_model_version=event_analysis['model_version'],
            event_classification_validation=event_analysis['validation_status'],
            class_granularity=event_analysis['granularity'],classification_message=event_analysis['message'],
            supported_event_classes=event_analysis['supported_classes'],events=events,
            verified_rule_violation=False,
            sampling=prediction['sampling'],member_scores=prediction['member_scores'],
            member_weights=prediction.get('member_weights'),qwen_observations=prediction.get('qwen_observations'),
            feature_cache_used=prediction.get('feature_cache_used',{}),
            review_timestamps_sec=prediction['review_timestamps_sec'],evidence=evidence)
        update(key,status='done',stage='Дайын',result=result)
    except Exception as e:
        print('INFERENCE_ERROR',type(e).__name__,str(e),flush=True)
        message=str(e) if isinstance(e,ValueError) else 'Видеоны оқу мүмкін болмады. MP4 немесе MOV файлын тексеріңіз.'
        update(key,status='error',error=message)
    finally:
        path.unlink(missing_ok=True)
        with LOCK:BUSY=False

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):print(fmt%args,flush=True)
    def allowed(self):return self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
    def send(self,code,data,kind='application/json; charset=utf-8'):
        if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode('utf8')
        if isinstance(data,str):data=data.encode('utf8')
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if not self.allowed():return self.send(403,{'error':'Invalid host'})
        path=urllib.parse.urlsplit(self.path).path
        if path=='/':return self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/api/config':return self.send(200,dict(token=TOKEN,model=MODEL_NAME,max_bytes=MAX_BYTES,
            model_version=SPLIT['model_version'],accuracy=MODEL_METRICS['accuracy'],f1=MODEL_METRICS['f1'],
            specificity=MODEL_METRICS['specificity'],recall=MODEL_METRICS['recall'],macro_f1=MODEL_METRICS['macro_f1'],
            threshold=SELECTION['threshold'],training_video_count=30,reserved_test_ids=[v['video_id'] for v in SPLIT['test']],
            event_class_supported=True,event_classes=['cash_placement','key_handling','customer_procedure'],
            metrics_scope='Single 10-video held-out evaluation after selection on train30; historically exposed data',
            performance_note=PERFORMANCE_NOTE))
        if path.startswith('/api/jobs/'):
            with LOCK:item=dict(JOBS.get(path.rsplit('/',1)[-1],{}))
            return self.send(200 if item else 404,item or {'error':'Тапсырма табылмады.'})
        if path=='/demo-video':
            size=DEMO.stat().st_size;start=0;end=size-1;partial=False
            value=self.headers.get('Range','')
            if value:
                try:
                    unit,span=value.split('=',1);a,b=span.split('-',1)
                    if unit!='bytes' or ',' in span:raise ValueError()
                    if a:start=int(a);end=min(int(b),size-1) if b else size-1
                    else:start=max(0,size-int(b))
                    if not 0<=start<=end<size:raise ValueError()
                    partial=True
                except ValueError:return self.send(416,{'error':'Invalid range'})
            self.send_response(206 if partial else 200);self.send_header('Content-Type','video/quicktime')
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with DEMO.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        block=f.read(min(65536,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        return self.send(404,{'error':'Not found'})
    def do_POST(self):
        global BUSY
        if not self.allowed() or self.headers.get('X-Local-Token')!=TOKEN:return self.send(403,{'error':'Бетті жаңартып, қайта көріңіз.'})
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{self.headers.get("Host")}':return self.send(403,{'error':'Invalid origin'})
        route=urllib.parse.urlsplit(self.path).path
        if route not in ['/api/upload','/api/demo']:return self.send(404,{'error':'Not found'})
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:return self.send(400,{'error':'Invalid length'})
        if route=='/api/upload' and not 0<length<=MAX_BYTES:return self.send(413,{'error':'Бос емес, 512 МБ-тан аспайтын видео таңдаңыз.'})
        with LOCK:
            if BUSY:return self.send(409,{'error':'Бір видео өңделіп жатыр. Аяқталғанын күтіңіз.'})
            BUSY=True
            if len(JOBS)>20:
                oldest=next(iter(JOBS));JOBS.pop(oldest)
        key=uuid.uuid4().hex;dest=RUNTIME/(key+'.video');name='X-026.mov'
        try:
            if route=='/api/demo':shutil.copyfile(DEMO,dest)
            else:
                name=urllib.parse.unquote(self.headers.get('X-Filename','video'))[:200]
                self.connection.settimeout(120)
                with dest.open('wb') as f:
                    remaining=length
                    while remaining:
                        b=self.rfile.read(min(1024*1024,remaining))
                        if not b:raise ValueError('Видео жүктеу аяқталмады.')
                        f.write(b);remaining-=len(b)
            with LOCK:JOBS[key]=dict(id=key,status='queued',stage='Өңдеу кезегінде')
            POOL.submit(process,key,dest,name)
            return self.send(202,{'id':key})
        except Exception:
            dest.unlink(missing_ok=True)
            with LOCK:BUSY=False
            return self.send(400,{'error':'Файл жүктелмеді. Қайта көріңіз.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Local: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();POOL.shutdown(wait=True)
