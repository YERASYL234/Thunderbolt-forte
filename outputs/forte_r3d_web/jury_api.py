"""Authenticated jury gateway; inference stays on the existing local service."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default
import json, secrets, hmac, urllib.request, urllib.error, urllib.parse, threading, time, uuid, argparse
ROOT=Path(__file__).resolve().parent
KEY_FILE=ROOT/'jury_api_key.txt'
if not KEY_FILE.exists():KEY_FILE.write_text(secrets.token_urlsafe(36),encoding='utf8')
API_KEY=KEY_FILE.read_text(encoding='utf8').strip()
BACKEND='http://127.0.0.1:8765'
MAX_BYTES=100*1024*1024
JOBS={};LOCK=threading.Lock()
CATALOG=json.loads((ROOT/'fortebank_cash_rules.json').read_text(encoding='utf8'))
RULES={r['id']:r for r in CATALOG['rules']}
MAPPING=json.loads((ROOT/'event_rule_mapping.json').read_text(encoding='utf8'))['mapping']

def legal_basis(class_id):
    matches=[]
    for rule_id in MAPPING.get(class_id,[]):
        r=RULES[rule_id]
        matches.append(dict(rule_id=rule_id,document_title=r['act_title'],clause=r['clause'],url=r['source_url'],requirement=r['requirement'],assessment='conditions_not_verified' if r['enabled_for_video_screening'] else 'review_required',applicability_conditions=r['applicability_conditions'],exceptions=r['exceptions'],applicability_verified=False))
    return dict(status='candidate_reference' if matches else 'no_supported_mapping',verified_violation=False,source='user_provided_rule_catalog',rules=matches)

def jury_result(job_id,result):
    verdict=result['prediction']
    if verdict not in ('violation','no_violation'):raise ValueError('Invalid verdict')
    violations=[]
    if verdict=='violation':
        for e in result.get('events',[]):
            frames=[]
            for f in e.get('frames',[])[:3]:
                prefix='data:image/jpeg;base64,'
                if not f['image_data_url'].startswith(prefix):raise ValueError('Invalid frame')
                frames.append(dict(timestamp_sec=f['actual_sec'],image_base64=f['image_data_url'][len(prefix):]))
            if not frames:raise ValueError('Missing event evidence')
            violations.append(dict(type=e['class_name'],timecode_sec=e['timestamp_sec'],description=' '.join(filter(None,[e['description'],'Модель болжамы; адам тексеруі қажет.',e.get('verification_needed')])),evidence_frames=frames,legal_basis=legal_basis(e['class_id'])))
        if not violations:raise ValueError('Missing localized event')
    return dict(job_id=job_id,status='done',verdict=verdict,violations=violations)

def backend(path,data=None,headers=None,timeout=15):
    req=urllib.request.Request(BACKEND+path,data=data,headers=headers or {})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,data):
        body=json.dumps(data,ensure_ascii=False).encode('utf8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def authenticated(self):
        if not hmac.compare_digest(self.headers.get('X-API-Key',''),API_KEY):self.send(401,{'error':'Invalid or missing X-API-Key'});return False
        return True
    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path=='/':
            body=(ROOT/'jury_portal.html').read_bytes()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body);return
        if not self.authenticated():return
        path=urllib.parse.urlsplit(self.path).path
        if path=='/api/v1/health':
            try:backend('/api/config');return self.send(200,{'status':'ok'})
            except Exception:return self.send(503,{'status':'unavailable'})
        if not path.startswith('/api/v1/jobs/'):return self.send(404,{'error':'Not found'})
        job_id=path.rsplit('/',1)[-1]
        with LOCK:job=JOBS.get(job_id)
        if not job:return self.send(404,{'error':'Unknown job_id'})
        if job.get('terminal'):return self.send(200,job['terminal'])
        if time.monotonic()-job['started']>300:
            value={'job_id':job_id,'status':'failed','error':'Processing exceeded 300 seconds'}
        else:
            try:
                state=backend('/api/jobs/'+job['backend_id'])
                if state['status']=='done':value=jury_result(job_id,state['result'])
                elif state['status']=='error':value={'job_id':job_id,'status':'failed','error':state.get('error','Processing failed')}
                else:return self.send(200,{'job_id':job_id,'status':'processing'})
            except Exception:return self.send(503,{'job_id':job_id,'error':'Local processing service unavailable'})
        with LOCK:job['terminal']=value
        return self.send(200,value)
    def do_POST(self):
        if not self.authenticated():return
        if urllib.parse.urlsplit(self.path).path!='/api/v1/analyze':return self.send(404,{'error':'Not found'})
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:return self.send(400,{'error':'Invalid Content-Length'})
        if not 0<length<=MAX_BYTES:return self.send(413,{'error':'Maximum request size is 100 MiB'})
        content_type=self.headers.get('Content-Type','')
        if not content_type.lower().startswith('multipart/form-data;'):return self.send(415,{'error':'Use multipart/form-data with field file'})
        try:
            self.connection.settimeout(60)
            raw=self.rfile.read(length)
            if len(raw)!=length:raise ValueError('Incomplete upload')
            message=BytesParser(policy=default).parsebytes(('Content-Type: '+content_type+'\r\nMIME-Version: 1.0\r\n\r\n').encode()+raw)
            files=[part for part in message.iter_parts() if part.get_param('name',header='content-disposition')=='file']
            if len(files)!=1:raise ValueError('Exactly one file field is required')
            name=files[0].get_filename() or ''
            if not name.lower().endswith('.mov'):raise ValueError('A .mov file is required')
            payload=files[0].get_payload(decode=True)
            if not payload:raise ValueError('Empty file')
        except Exception as exc:return self.send(400,{'error':str(exc) if isinstance(exc,ValueError) else 'Invalid multipart upload'})
        try:
            cfg=backend('/api/config')
            task=backend('/api/upload',payload,{'X-Local-Token':cfg['token'],'X-Filename':urllib.parse.quote(name),'Content-Type':'application/octet-stream'},timeout=120)
        except urllib.error.HTTPError as exc:return self.send(409 if exc.code==409 else 502,{'error':'Service is busy; retry later' if exc.code==409 else 'Upload was rejected'})
        except Exception:return self.send(503,{'error':'Local processing service unavailable'})
        job_id=uuid.uuid4().hex
        with LOCK:
            # Results stay in memory only; keep a bounded number of jobs.
            if len(JOBS)>=100:JOBS.pop(next(iter(JOBS)))
            JOBS[job_id]={'backend_id':task['id'],'started':time.monotonic()}
        return self.send(202,{'job_id':job_id,'status':'processing'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Jury API: http://127.0.0.1:{args.port}/api/v1',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
