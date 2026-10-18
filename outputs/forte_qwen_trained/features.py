"""Frozen local Qwen observation features, independent of labels and filenames."""
from pathlib import Path
import sys, os, json, time, socket, subprocess, urllib.request, secrets, hashlib, io, base64
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / 'work'
for p in [WORK/'pydeps', WORK/'scideps', WORK/'train_deps']:
    if p.exists(): sys.path.insert(0, str(p))
import av
import numpy as np
from PIL import Image

QWEN = ROOT.parent / 'forte_qwen'
CUES = [
    ('banknotes_on_table', 'Banknotes clearly visible on the tabletop.'),
    ('bag_on_table', 'A bag is on the tabletop; do not infer its contents.'),
    ('multiple_people_behind_desk', 'More than one person is on the worker side of the desk.'),
    ('person_enters_worker_area', 'Across frames a person visibly moves into the worker side of the desk.'),
    ('person_leaves_worker_area', 'Across frames a person visibly leaves the worker side of the desk.'),
    ('desk_unoccupied', 'The worker side of the desk is unoccupied in at least one frame.'),
    ('glass_door_open', 'The glass doorway is visibly open in at least one frame.'),
    ('glass_door_state_changes', 'The glass door visibly changes from open to closed or closed to open.'),
    ('cabinet_door_open', 'A safe or storage cabinet door is visibly open.'),
    ('cabinet_door_state_changes', 'A safe or cabinet door visibly changes between open and closed.'),
    ('key_in_lock', 'A key is clearly visible in a lock; use unclear for small indistinct objects.'),
    ('banknote_counter_used', 'A person visibly places banknotes into or removes them from a banknote counting machine.'),
    ('cash_handover', 'Banknotes are visibly passed between two people.'),
    ('cash_placed_on_table', 'A person visibly places banknotes on the tabletop.'),
    ('object_placed_in_bag', 'A person visibly puts an object inside a bag.'),
    ('phone_in_hand', 'A person visibly holds a mobile phone in their hand.'),
]
STATES = ['yes', 'no', 'unclear']
SPEC = dict(version='qwen_cues_v2_named_fields', samples=8, max_side=448, image_max_tokens=256,
            cue_definitions=CUES, states=STATES,
            sampling='8 uniformly spaced frames from 2% to 98% of duration; no labels or event timestamps',
            model='Qwen3-VL-2B-Instruct Q4_K_M; Q8_0 projector; frozen', runtime='llama.cpp b11669')
SPEC_HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()
SCHEMA = dict(type='object', properties={key:dict(type='string',enum=STATES) for key,_ in CUES}, required=[key for key,_ in CUES], additionalProperties=False)
SYSTEM = ('You are a visual observation extractor. The images are uniformly sampled in chronological order from one video. '
          'For each numbered cue, output yes only if directly visible, no if clearly absent in all supplied images, '
          'and unclear if small, occluded, ambiguous, or requiring unseen intermediate actions. '
          'Do not infer authorization, ownership, intent, regulatory violations, or actions outside the sampled frames. '
          'Do not identify people. Return a JSON object with the named fields below. Each value must be yes, no, or unclear. '
          'Evaluate every field separately from visual evidence.\n' +
          '\n'.join(f'{key}: {desc}' for key, desc in CUES))
SPEC['prompt']=SYSTEM
SPEC_HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def vector(states):
    if not isinstance(states,list) or len(states)!=len(CUES) or any(s not in STATES for s in states):
        raise ValueError('Invalid feature states')
    return np.asarray([float(s==category) for s in states for category in STATES], dtype=np.float64)

def sample(path):
    images=[]; actual=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        duration=float(s.duration*s.time_base) if s.duration else c.duration/1e6
        origin=float((s.start_time or 0)*s.time_base)
        targets=np.linspace(.02*duration, .98*duration, SPEC['samples'])
        pos=0
        for f in c.decode(s):
            if f.pts is None: continue
            t=float(f.pts*s.time_base)-origin
            if pos<len(targets) and t>=targets[pos]:
                im=f.to_image().convert('RGB'); im.thumbnail((SPEC['max_side'],SPEC['max_side']))
                b=io.BytesIO(); im.save(b,format='JPEG',quality=88)
                enc='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
                while pos<len(targets) and t>=targets[pos]:
                    images.append(enc);actual.append(t);pos+=1
            if pos==len(targets): break
    if len(images)!=SPEC['samples']: raise ValueError('Could not sample all requested frames')
    return images, actual

class LocalQwen:
    def __init__(self, log_path, gpu_layers=99):
        self.log_path=Path(log_path);self.gpu_layers=gpu_layers;self.proc=None;self.log=None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.key=secrets.token_hex(24)
    def __enter__(self):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        cmd=[str(QWEN/'runtime/llama-server.exe'),'-m',str(QWEN/'Qwen3VL-2B-Instruct-Q4_K_M.gguf'),
             '--mmproj',str(QWEN/'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'),'--host','127.0.0.1','--port',str(port),
             '--api-key',self.key,'-ngl',str(self.gpu_layers),'-c','4096','-np','1','-b','128','-ub','128','-t','4',
             '--image-max-tokens',str(SPEC['image_max_tokens'])]
        self.log=self.log_path.open('wb')
        try:
            self.proc=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if self.proc.poll() is not None: raise RuntimeError(f'Local model failed; see {self.log_path}')
                try:
                    req=urllib.request.Request(self.url+'/health',headers={'Authorization':'Bearer '+self.key})
                    with self.opener.open(req,timeout=2) as r:
                        if r.status==200: return self
                except OSError: pass
                time.sleep(1)
            raise TimeoutError('Local Qwen startup timeout')
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        if self.log:self.log.close()
    def extract(self,path):
        started=time.monotonic(); images,times=sample(path)
        content=[dict(type='text',text='Assess each named visual cue using these 8 chronological frames. Return all 16 named fields as JSON.')]
        for i,img in enumerate(images):
            content.extend([dict(type='text',text=f'Frame {i+1}:'),dict(type='image_url',image_url=dict(url=img))])
        payload=dict(messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
                     temperature=0,seed=20261017,max_tokens=512,stream=False,cache_prompt=False,
                     response_format=dict(type='json_object',schema=SCHEMA))
        req=urllib.request.Request(self.url+'/v1/chat/completions',data=json.dumps(payload).encode(),
                                   headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        with self.opener.open(req,timeout=240) as r: response=json.load(r)
        choice=response['choices'][0]; raw=choice['message']['content']
        if choice.get('finish_reason')!='stop':raise ValueError('Truncated observation response: '+raw)
        data=json.loads(raw);states=[data[key] for key,_ in CUES];vector(states)
        return dict(spec_hash=SPEC_HASH,sha256=digest(path),states=states,
                    cue_values=data,sampled_timestamps_sec=times,
                    elapsed_sec=time.monotonic()-started,raw_response=raw,usage=response.get('usage'),
                    model_observations_not_verified_facts=True)
