"""Label-blind background clusters for a stricter exploratory transfer check."""
from data import *
from sklearn.cluster import KMeans
from collections import Counter
vs=videos();xs=[]
for v in vs:
    p=next(r['path'] for r in v['review_frames'] if r['tag']=='overview_0')
    im=Image.open(DS/p).convert('RGB');w,h=im.size
    # Upper quarter mostly contains walls/door frames; no labels or event-selected images.
    a=np.asarray(im.crop((0,0,w,round(h*.25))).resize((32,16)),dtype=float)/255
    xs.append(a.reshape(-1))
labels=KMeans(n_clusters=2,random_state=SEED,n_init=20).fit_predict(np.stack(xs))
rows=[dict(video_id=v['video_id'],background_group=int(g)) for v,g in zip(vs,labels)]
counts={str(g):dict(Counter(v['source_verdict'] for v,k in zip(vs,labels) if k==g)) for g in sorted(set(labels))}
r=dict(method='KMeans on resized upper-quarter pixels of first uniform overview frame; label-blind',
       warning='Automatic background clusters are not verified camera/session/actor identities.',counts=counts,videos=rows)
(ROOT/'background_groups.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
