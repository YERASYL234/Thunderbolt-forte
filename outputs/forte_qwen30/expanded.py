"""Train-only small-data extension, chosen before consulting any new test results.

Preserves the initial Qwen experiment. The deployed selection is this expanded
train30 search, regardless of its test outcome; no test-based winner picking.
"""
from pathlib import Path
import argparse,json,datetime,shutil
import train as experiment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE=Path(__file__).resolve().parent
experiment.ROOT=BASE.parent/'forte_qwen_plus30';experiment.ROOT.mkdir(exist_ok=True)
experiment.VERSION='forte_qwen4_plus_train30_v1'
original_estimator=experiment.estimator
def estimator(c):
    if c['kind']=='knn':return make_pipeline(Normalizer(),KNeighborsClassifier(n_neighbors=c['k'],metric='cosine',weights='distance',n_jobs=2))
    if c['kind']=='pca_lda':return make_pipeline(StandardScaler(),PCA(n_components=4,svd_solver='full'),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))
    return original_estimator(c)
experiment.estimator=estimator
experiment.CANDIDATES.extend([
    *[dict(name=f'{f}_cosine_{k}',members=[dict(feature=f,kind='knn',k=k)],weights=[1.])
      for f in ['r3d','dino_motion','resnet','dino_r3d'] for k in [3,7]],
    *[dict(name=f'{f}_PCA4_LDA',members=[dict(feature=f,kind='pca_lda')],weights=[1.])
      for f in ['r3d','dino_motion','resnet']],
])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','test','report'],required=True);a=p.parse_args()
    if a.stage=='train':
        experiment.save('extension_protocol.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reason='Initial train30 nested development accuracy was 40%; evaluate small-data classifiers on training videos only.',
            initial_test_report_consulted_for_this_extension=False,candidate_count=len(experiment.CANDIDATES),
            deployment_policy='Use the expanded train30-selected classifier, regardless of new test outcome. No picking between test results.',
            limitations='Repeated adaptive development on train30. All 40 videos historically exposed; test10 is not a new independent external test.',
            initial_experiment_preserved='../forte_qwen30',feature_cache='../forte_qwen30/feature_cache'))
        experiment.train()
        for name in ['feature_audit.json','feature_spec.json','provenance.json','input_verification.json']:
            shutil.copyfile(BASE/name,experiment.ROOT/name)
    elif a.stage=='test':experiment.test()
    else:
        import importlib.util
        module=importlib.util.spec_from_file_location('forte_qwen_report',BASE/'report.py')
        report=importlib.util.module_from_spec(module);module.loader.exec_module(report)
        report.ROOT=experiment.ROOT;report.main()
