# NB1: SYMPTOM MODEL — CORRECTED (No SMOTE — dataset pre-balanced)
# Saves: best_symptom_model.keras, final_symptom_model.keras, scaler, encoder
# Run on Colab GPU. Dataset: symbipredict_2022.csv (4961 samples, 132 features, 41 classes, 121/class)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import keras, os, pickle, shutil
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Conv1D, Flatten, LSTM,
                          Bidirectional, Attention, concatenate, BatchNormalization)
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             recall_score, top_k_accuracy_score, roc_auc_score,
                             roc_curve, auc as sk_auc)
import warnings; warnings.filterwarnings('ignore')
for d in ['/content/saved_models','/content/figures','/content/results']: os.makedirs(d,exist_ok=True)

def ece_full(y_true, y_prob, n_bins=15):
    c=np.max(y_prob,1); p=np.argmax(y_prob,1); a=(p==y_true).astype(float)
    ece=0.0; ba,bc,bn=[],[],[]
    bnd=np.linspace(0,1,n_bins+1)
    for i in range(n_bins):
        bl,bu=bnd[i],bnd[i+1]
        m=(c>=bl)&(c<=bu) if i==n_bins-1 else (c>=bl)&(c<bu)
        cnt=np.sum(m)
        if cnt>0:
            acc=np.mean(a[m]); conf=np.mean(c[m])
            ece+=np.abs(conf-acc)*(cnt/len(c))
            ba.append(acc); bc.append(conf); bn.append(cnt)
    return ece, np.array(ba), np.array(bc), np.array(bn)

def brier_multi(yo,yp): return np.mean(np.sum((yp-yo)**2,1))

class TestCB(keras.callbacks.Callback):
    def __init__(s,Xt,yt): super().__init__(); s.Xt=Xt; s.yt=yt; s.tl=[]; s.ta=[]
    def on_epoch_end(s,ep,logs=None):
        l,a=s.model.evaluate(s.Xt,s.yt,verbose=0); s.tl.append(l); s.ta.append(a)

# ═══════ LOAD DATA ═══════
dataset = pd.read_csv('/content/symbipredict_2022.csv')
X_all = dataset.drop('prognosis',axis=1).values; y_raw = dataset['prognosis'].values
le = LabelEncoder(); y_all = le.fit_transform(y_raw); NC = len(le.classes_)
print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features, {NC} classes")
print(f"Balanced: {pd.Series(y_all).value_counts().unique()} samples/class — NO SMOTE NEEDED")

X_tv, X_te, y_tv, y_te = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=42)
print(f"Train+Val: {X_tv.shape[0]} | Test: {X_te.shape[0]}")

# Exact disease names (note trailing spaces on Diabetes and Hypertension)
ACUTE_LABELS = ['Heart Attack','Pneumonia','Tuberculosis','Typhoid','Hepatitis E',
                'Dengue','Malaria','Jaundice','Paralysis (brain hemorrhage)',
                'Alcoholic Hepatitis','Drug Reaction','Gastroenteritis']
ACUTE_CRITICAL = ['Heart Attack','Paralysis (brain hemorrhage)','AIDS']
ACUTE_MODERATE = ['Malaria','Dengue','Typhoid','Hepatitis E','Pneumonia',
                  'Tuberculosis','Alcoholic Hepatitis','Drug Reaction']
CHRONIC = ['Diabetes ','Hypertension ','Arthritis','Hypothyroidism',
           'Hyperthyroidism','Hypoglycemia','Osteoarthritis',
           'Cervical Spondylosis','Bronchial Asthma','GERD']
acute_idx = [np.where(le.classes_==l)[0][0] for l in ACUTE_LABELS if l in le.classes_]

# ═══════ MODEL ═══════
L2R = 1e-3
def DenseNetBlock(inp,gr=24,layers=5):
    for _ in range(layers):
        x=Conv1D(gr,3,padding='same',activation='relu',kernel_regularizer=l2(L2R))(inp)
        x=BatchNormalization()(x); x=Dropout(0.5)(x); inp=concatenate([inp,x])
    return inp
def LSTMAttnBlock(inp):
    o=Bidirectional(LSTM(200,return_sequences=True,kernel_regularizer=l2(L2R)))(inp)
    return Attention()([o,o])
def build_model(shape,nc):
    i=Input(shape=shape); dn=i
    for _ in range(3): dn=DenseNetBlock(dn)
    a1=Attention()([dn,dn]); lo=a1
    for _ in range(2): lo=LSTMAttnBlock(lo)
    f=concatenate([Flatten()(a1),Flatten()(lo)])
    x=Dense(512,activation='relu',kernel_regularizer=l2(L2R))(f); x=Dropout(0.6)(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(L2R))(x); x=Dropout(0.6)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(L2R))(x); x=Dropout(0.6)(x)
    o=Dense(nc,activation='softmax')(x)
    m=Model(i,o); m.compile(Adam(1e-3),'categorical_crossentropy',['accuracy']); return m

# ═══════ 5-FOLD CV ═══════
EP,BS = 85,32
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
fold_res,fold_pred,all_yt,all_yp,all_yprob=[],[],[],[],[]
agg_cm=np.zeros((NC,NC),dtype=int); best_va,best_fi=-1,-1
y_te_ohe=to_categorical(y_te,NC)

for fold,(ti,vi) in enumerate(skf.split(X_tv,y_tv)):
    print(f"\n{'='*55}\n  FOLD {fold+1}\n{'='*55}")
    Xtr,Xvl=X_tv[ti],X_tv[vi]; ytr,yvl=y_tv[ti],y_tv[vi]
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xvl=sc.transform(Xvl); Xtest=sc.transform(X_te)
    # NO SMOTE — data already balanced at 121/class
    ytr_o=to_categorical(ytr,NC); yvl_o=to_categorical(yvl,NC)
    Xtr3=np.expand_dims(Xtr,-1); Xvl3=np.expand_dims(Xvl,-1); Xte3=np.expand_dims(Xtest,-1)
    mdl=build_model((Xtr3.shape[1],1),NC)
    ckpt=f'/content/saved_models/symptom_fold_{fold+1}.keras'
    tcb=TestCB(Xte3,y_te_ohe)
    h=mdl.fit(Xtr3,ytr_o,validation_data=(Xvl3,yvl_o),epochs=EP,batch_size=BS,
              callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,min_lr=1e-7),
                         ModelCheckpoint(ckpt,monitor='val_accuracy',save_best_only=True,mode='max'),tcb],verbose=1)
    tl,ta=mdl.evaluate(Xte3,y_te_ohe,verbose=0)
    yp=mdl.predict(Xte3,verbose=0); ypc=np.argmax(yp,1)
    va=h.history['val_accuracy'][-1]
    if va>best_va: best_va=va; best_fi=fold
    cm=confusion_matrix(y_te,ypc,labels=range(NC)); agg_cm+=cm
    all_yt.extend(y_te); all_yp.extend(ypc); all_yprob.append(yp)
    fold_pred.append({'y_true':y_te.copy(),'y_pred_prob':yp.copy()})
    f1=f1_score(y_te,ypc,average='macro')
    t3=top_k_accuracy_score(y_te,yp,k=3); t5=top_k_accuracy_score(y_te,yp,k=5)
    ec,_,_,_=ece_full(y_te,yp); br=brier_multi(y_te_ohe,yp)
    try: auc=roc_auc_score(y_te_ohe,yp,average='macro',multi_class='ovr')
    except: auc=0.0
    se=recall_score(y_te,ypc,labels=acute_idx,average='macro',zero_division=0)
    npvs=[]
    for idx in acute_idx:
        tp=cm[idx,idx];fn=np.sum(cm[idx,:])-tp;tn=np.sum(cm)-np.sum(cm[idx,:])-np.sum(cm[:,idx])+tp
        npvs.append(tn/(tn+fn) if (tn+fn)>0 else 0)
    npv=np.mean(npvs) if npvs else 0.0
    print(f"  Test:{ta:.4f} F1:{f1:.4f} AUC:{auc:.4f} Top3:{t3:.4f} Top5:{t5:.4f}")
    print(f"  ECE:{ec:.6f} Brier:{br:.6f} AcuteSens:{se:.4f} NPV:{npv:.4f}")
    fold_res.append({'train_loss':h.history['loss'][-1],'train_acc':h.history['accuracy'][-1],
        'val_loss':h.history['val_loss'][-1],'val_acc':va,'test_loss':tl,'test_acc':ta,
        'macro_f1':f1,'roc_auc':auc,'top3':t3,'top5':t5,'ece':ec,'brier':br,
        'sens_acute':se,'npv_acute':npv})
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    a1.plot(h.history['loss'],label='Train');a1.plot(h.history['val_loss'],label='Val')
    a1.plot(tcb.tl,label='Test');a1.set_title(f'Loss-Fold {fold+1}');a1.legend();a1.grid(alpha=0.3)
    a2.plot(h.history['accuracy'],label='Train');a2.plot(h.history['val_accuracy'],label='Val')
    a2.plot(tcb.ta,label='Test');a2.set_title(f'Acc-Fold {fold+1}');a2.legend();a2.grid(alpha=0.3)
    plt.tight_layout();plt.savefig(f'/content/figures/symptom_fold_{fold+1}.png',dpi=200);plt.show()

shutil.copy(f'/content/saved_models/symptom_fold_{best_fi+1}.keras','/content/saved_models/best_symptom_model.keras')
print(f"\n✅ Best model: Fold {best_fi+1} (Val Acc: {best_va:.4f})")

# ═══════ FINAL RETRAIN ═══════
print(f"\n{'='*55}\n  FINAL MODEL\n{'='*55}")
fsc=StandardScaler(); Xtvs=fsc.fit_transform(X_tv); Xtes=fsc.transform(X_te)
ytvo=to_categorical(y_tv,NC); Xtv3=np.expand_dims(Xtvs,-1); Xte3f=np.expand_dims(Xtes,-1)
fm=build_model((Xtv3.shape[1],1),NC)
fm.fit(Xtv3,ytvo,validation_split=0.1,epochs=EP,batch_size=BS,
       callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,min_lr=1e-7)],verbose=1)
fl,fa=fm.evaluate(Xte3f,y_te_ohe,verbose=0)
print(f"  Final: Acc={fa:.4f} Loss={fl:.4f}")
fm.save('/content/saved_models/final_symptom_model.keras')
with open('/content/saved_models/symptom_scaler.pkl','wb') as f: pickle.dump(fsc,f)
with open('/content/saved_models/symptom_label_encoder.pkl','wb') as f: pickle.dump(le,f)

# ═══════ Q3 AGGREGATE ═══════
print(f"\n{'='*70}\n  Q3: SYMPTOM MODEL AGGREGATE\n{'='*70}")
MK=list(fold_res[0].keys()); rt={}
for k in MK:
    vs=[r[k] for r in fold_res]; m,s=np.mean(vs),np.std(vs,ddof=1); rt[k]={'mean':m,'std':s}
    fmt='.6f' if k in ['ece','brier'] else '.4f'
    print(f"  {k.replace('_',' ').title():<25}: {m:{fmt}} ± {s:{fmt}}")
pd.DataFrame(fold_res,index=[f'Fold {i+1}' for i in range(5)]).to_csv('/content/results/symptom_fold_metrics.csv',float_format='%.6f')
print(f"\n  CLASSIFICATION REPORT:")
print(classification_report(all_yt,all_yp,target_names=le.classes_))
plt.figure(figsize=(14,12))
sns.heatmap(agg_cm,annot=True,fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)
plt.title('Symptom Model — Aggregate CM');plt.xticks(rotation=90,fontsize=7);plt.yticks(fontsize=7)
plt.tight_layout();plt.savefig('/content/figures/symptom_cm.png',dpi=300,bbox_inches='tight');plt.show()

# ROC
all_prob=np.concatenate(all_yprob);all_true=np.array(all_yt);ato=to_categorical(all_true,NC)
fpr,tpr,rd={},{},{}
for i in range(NC): fpr[i],tpr[i],_=roc_curve(ato[:,i],all_prob[:,i]);rd[i]=sk_auc(fpr[i],tpr[i])
afpr=np.unique(np.concatenate([fpr[i] for i in range(NC)]))
mtpr=np.zeros_like(afpr)
for i in range(NC): mtpr+=np.interp(afpr,fpr[i],tpr[i])
mtpr/=NC; mauc=sk_auc(afpr,mtpr)
plt.figure(figsize=(8,7))
plt.plot(afpr,mtpr,color='#2E75B6',lw=3,label=f'Macro ROC (AUC={mauc:.4f})')
plt.plot([0,1],[0,1],'k--');plt.xlabel('FPR');plt.ylabel('TPR')
plt.title('Symptom Model — ROC',fontweight='bold');plt.legend(fontsize=12);plt.grid(alpha=0.3)
plt.tight_layout();plt.savefig('/content/figures/symptom_roc.png',dpi=300,bbox_inches='tight');plt.show()

# ═══════ Q4 CALIBRATION ═══════
print(f"\n{'='*70}\n  Q4: CALIBRATION\n{'='*70}")
aq4t=np.concatenate([fp['y_true'] for fp in fold_pred])
aq4p=np.concatenate([fp['y_pred_prob'] for fp in fold_pred])
ece_a,ba,bc,bn=ece_full(aq4t,aq4p,15)
fig,(a1,a2)=plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios':[3,1]},sharex=True)
a1.plot([0,1],[0,1],'k--',alpha=0.7)
a1.plot(bc,ba,'o-',color='#2E75B6',lw=2,ms=8,label=f'ECE={ece_a:.4f}')
a1.fill_between(bc,ba,bc,alpha=0.15,color='#2E75B6')
a1.set_ylabel('Observed Acc');a1.set_title('Reliability Diagram',fontweight='bold');a1.legend();a1.grid(alpha=0.3)
conf_all=np.max(aq4p,1)
a2.hist(conf_all,bins=15,range=(0,1),color='#2E75B6',alpha=0.7,edgecolor='white')
a2.set_xlabel('Confidence');a2.set_ylabel('Count');a2.grid(alpha=0.3)
plt.tight_layout();plt.savefig('/content/figures/symptom_reliability.png',dpi=300,bbox_inches='tight');plt.show()

# Threshold sweep
pred_q4=np.argmax(aq4p,1);cor_q4=(pred_q4==aq4t).astype(float);am=np.isin(aq4t,acute_idx)
sw=[]
for t in np.arange(0.50,1.00,0.025):
    mk=conf_all>=t;cov=mk.mean();ac=cor_q4[mk].mean() if mk.sum()>0 else 0
    acm=am&mk;afn=(1-cor_q4[acm].mean()) if acm.sum()>0 else 0
    sw.append({'threshold':t,'coverage':cov,'acc':ac,'acute_fn':afn,'rejection':1-cov})
sdf=pd.DataFrame(sw);sdf.to_csv('/content/results/symptom_threshold.csv',index=False)
print("  Threshold Sweep:")
for t in [0.80,0.90,0.95,0.975]:
    r=sdf.loc[(sdf['threshold']-t).abs().idxmin()]
    print(f"  θ={r['threshold']:.3f} → Cov:{r['coverage']:.4f} Acc:{r['acc']:.4f} AcFN:{r['acute_fn']:.4f}")

fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
a1.plot(sdf['threshold'],sdf['coverage'],'b-o',ms=4,label='Coverage')
a1.plot(sdf['threshold'],sdf['acc'],'g-s',ms=4,label='Accuracy')
a1.axvline(0.95,color='red',ls='--',alpha=0.7,label='θ=0.95')
a1.legend();a1.grid(alpha=0.3);a1.set_title('Coverage vs Accuracy',fontweight='bold')
a2.plot(sdf['threshold'],sdf['acute_fn'],'r-^',ms=5,label='Acute FN')
a2.plot(sdf['threshold'],sdf['rejection'],'-D',color='orange',ms=4,label='Rejection')
a2.axvline(0.95,color='red',ls='--',alpha=0.7,label='θ=0.95')
a2.legend();a2.grid(alpha=0.3);a2.set_title('Safety',fontweight='bold')
plt.tight_layout();plt.savefig('/content/figures/symptom_threshold.png',dpi=300,bbox_inches='tight');plt.show()

# Cost analysis
cfn,cfp={},{}
for i,c in enumerate(le.classes_):
    cfp[i]=1.0
    if c in ACUTE_CRITICAL: cfn[i]=10.0
    elif c in ACUTE_MODERATE: cfn[i]=5.0
    elif c in CHRONIC: cfn[i]=2.0
    else: cfn[i]=1.0
fc=[]
for fp in fold_pred:
    yt,yp=fp['y_true'],np.argmax(fp['y_pred_prob'],1)
    cost=sum(np.sum((yt==c)&(yp!=c))*cfn[c]+np.sum((yt!=c)&(yp==c))*cfp[c] for c in range(NC))
    fc.append(cost/len(yt))
print(f"\n  Cost/sample: {np.mean(fc):.4f} ± {np.std(fc,ddof=1):.4f}")
for sn,sl in {'Acute-Critical':ACUTE_CRITICAL,'Acute-Moderate':ACUTE_MODERATE,'Chronic':CHRONIC}.items():
    si=[np.where(le.classes_==l)[0][0] for l in sl if l in le.classes_]
    if si:
        sm=np.isin(aq4t,si)
        if sm.sum()>0: print(f"    {sn}: Sens={cor_q4[sm].mean():.4f} ({sm.sum()} samples)")

print(f"\n✅ DONE! Models: /content/saved_models/ | Figures: /content/figures/ | Results: /content/results/")
