from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
DS=ROOT.parent/'forte_dataset'
for p in [BASE/'work/pydeps',BASE/'work/scideps',BASE/'work/train_deps']:
    if p.exists():sys.path.insert(0,str(p))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import torch
torch.set_num_threads(4)
SEED=20261017
SPEC={'encoder':'DINOv2 ViT-S/14','frozen':True,'fps':2.0,'resize':[224,224],
      'views':{'full':[0,0,1,1],'desk':[0,.32,.82,.72]},'embedding_dim_per_view':384,
      'crop_note':'Fixed normalized desk-and-hands crop checked on label-blind overview montage; not object tracking.',
      'sampling':'targets at 0,0.5,... before last frame; first decoded frame at/after each target',
      'version':'vits14_2fps_full_desk_v1'}
CONFIG={'window_frames':8,'stride_frames':2,'hidden_channels':[64,32],'dropout':.3,
        'mil_top_fraction':.1,'lambda_point':.5,'mu_smooth':.1,'point_radius_sec':1.0,
        'learning_rate':.001,'weight_decay':.01,'batch_size':4,'max_epochs':60,
        'min_epochs':30,'patience':10,'threshold':.5,'seed':SEED}
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def videos():return read(DS/'annotations/videos.json')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def encoder():
    p=read(ROOT/'encoder_provenance.json');sys.path.insert(0,str(ROOT/p['vendor_path']))
    from dinov2.hub.backbones import dinov2_vits14
    m=dinov2_vits14(pretrained=False)
    m.load_state_dict(torch.load(ROOT/'dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    m.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in m.parameters())
    return m
