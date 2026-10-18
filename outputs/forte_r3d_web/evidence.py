"""Model-selected event frames, with annotations kept as a separate reference."""
from pathlib import Path
import json,io,base64
import av
ROOT=Path(__file__).resolve().parent
DS=ROOT.parent/'forte_dataset'
EVENTS=json.loads((DS/'annotations/events.json').read_text(encoding='utf8'))
TAXONOMY={x['class_id']:x for x in json.loads((DS/'annotations/taxonomy.json').read_text(encoding='utf8'))}

def frame_at_times(path,targets,duration):
    targets=sorted(set(round(min(max(0,float(t)),max(0,duration-.05)),3) for t in targets))
    frames={};pos=0;last=None;last_t=None
    def pack(frame,t,target):
        im=frame.to_image().convert('RGB');im.thumbnail((560,560));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=85)
        return dict(requested_sec=target,actual_sec=round(t,3),image_data_url='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))
    with av.open(str(path)) as c:
        s=c.streams.video[0];origin=float((s.start_time or 0)*s.time_base)
        for frame in c.decode(s):
            if frame.pts is None:continue
            t=float(frame.pts*s.time_base)-origin;last=frame;last_t=t
            while pos<len(targets) and t+1e-6>=targets[pos]:
                frames[targets[pos]]=pack(frame,t,targets[pos]);pos+=1
            if pos==len(targets):break
        if last is not None:
            for target in targets[pos:]:frames[target]=pack(last,last_t,target)
    if len(frames)!=len(targets):raise ValueError('Көрсету үшін кадрлар алынбады.')
    return frames

def evidence_plan(video_id,clip_times,duration):
    matched=[e for e in EVENTS if e['video_id']==video_id and isinstance(e.get('source_timecode_sec'),(int,float))]
    groups={}
    for e in matched:
        t=float(e['source_timecode_sec']);g=groups.setdefault(t,dict(timestamp_sec=t,source='user_dataset_annotation',
            title='Датасетте белгіленген оқиға',rules_verified=False,events=[]))
        taxonomy=TAXONOMY.get(e['class_id'],{})
        g['events'].append(dict(class_id=e['class_id'],description=e['normalized_description'],
            verification_needed=taxonomy.get('review_requirement','Толық оқиға мен банк ережесін тексеру қажет'),
            rule_id=e.get('rule_id'),rule_status=e.get('rule_status','not_verified'),source_note=e.get('source_note','')))
    if groups:
        cards=[groups[t] for t in sorted(groups)]
        for g in cards:g['targets']=[min(max(0,g['timestamp_sec']+offset),max(0,duration-.05)) for offset in [-1,0,1]]
        mode='annotation_reference'
        note='Оқиға атаулары мен таймкодтар — сенің датасеттегі белгілерің. Модель оларды анықтаған жоқ. Кадрлар жүктелген видеодан алынды; ереже бұзылғаны расталмаған.'
    else:
        cards=[dict(timestamp_sec=float(times[len(times)//2]),source='uniform_model_sampling',
            title=f'Шолу кадры {i+1}',rules_verified=False,events=[],targets=[float(times[len(times)//2])]) for i,times in enumerate(clip_times)]
        mode='review_only'
        note='Бұл видеоға оқиға түрі мен таймкод белгісі табылмады. Төменде модель қараған үзінділердің шолу кадрлары көрсетілген. Олар күмәнділік бойынша таңдалмаған; нақты бұзылған ереже анықталған жоқ.'
    return mode,note,cards

def build_evidence(path,video_id,clip_times,duration,event_analysis=None):
    mode,note,cards=evidence_plan(video_id,clip_times,duration)
    reference_cards=[]
    if event_analysis is not None:
        reference_cards=cards if mode=='annotation_reference' else []
        if event_analysis['events']:
            cards=[]
            for e in event_analysis['events']:
                t=e['timestamp_sec']
                targets=[max(0,min(duration-.05,t+offset)) for offset in [-1,0,1]]
                cards.append(dict(timestamp_sec=t,source='event_window_model',
                    title=e['class_name']+' · болжам',rules_verified=False,events=[e],
                    window_start_sec=e['window_start_sec'],window_end_sec=e['window_end_sec'],targets=targets))
            mode='model_candidates'
            note=event_analysis['message']+' Кадрлар осы модель таңдаған уақыттан алынды.'
        else:
            _,_,cards=evidence_plan(None,clip_times,duration)
            mode='review_only';note=event_analysis['message']+' Төменде шолу кадрлары берілген.'
    frames=frame_at_times(path,[t for c in cards+reference_cards for t in c['targets']],duration)
    for c in cards+reference_cards:
        c['frames']=[frames[round(t,3)] for t in c.pop('targets')]
    return dict(mode=mode,note=note,cards=cards,reference_cards=reference_cards,
                automatic_rule_detection=False,verified_rule_violation=False,
                model_event_classification=event_analysis is not None)
