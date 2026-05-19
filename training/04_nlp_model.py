"""
================================================================================
NB4: NLP MODEL — TRAINING + MODEL SAVING + Q3 + Q4 COMPLETE
================================================================================
Fixes: validation≠test, 5-fold stratified CV, fresh model per fold,
       softmax probs for ROC, saves best model + final retrained model.

Outputs: saved models, fold metrics, reliability diagram, threshold sweep,
         cost analysis, confusion matrix, ROC, classification report.

Dataset: bert_train.csv (55 classes after filtering singletons)
Run on: Colab GPU
================================================================================
"""

# ═══════ CELL 1: IMPORTS ═══════
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import keras, os, pickle, shutil
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D,
                          LSTM, Bidirectional, Attention, Embedding, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             recall_score, roc_auc_score, top_k_accuracy_score,
                             roc_curve, auc as sk_auc)
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import wordnet
import random, warnings
warnings.filterwarnings('ignore')
for d in ['/content/saved_models','/content/figures','/content/results']:
    os.makedirs(d, exist_ok=True)

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

def brier_multi(y_ohe, y_prob):
    return np.mean(np.sum((y_prob-y_ohe)**2, axis=1))

class TestCB(keras.callbacks.Callback):
    def __init__(self, Xt, yt):
        super().__init__(); self.Xt=Xt; self.yt=yt; self.tl=[]; self.ta=[]
    def on_epoch_end(self, ep, logs=None):
        l,a=self.model.evaluate(self.Xt,self.yt,verbose=0)
        self.tl.append(l); self.ta.append(a)

# ═══════ CELL 2: DATA LOADING & AUGMENTATION ═══════
df = pd.read_csv('/content/bert_train.csv')
vc = df['Label'].value_counts()
df = df[df['Label'].isin(vc[vc>1].index)]
print(f"After filtering: {len(df)} samples, {df['Label'].nunique()} classes")

AUG_THRESH = 20
def syn_replace(text, n=1):
    words=text.split(); nw=words.copy()
    wl=list(set([w for w in words if wordnet.synsets(w)]))
    random.shuffle(wl); rep=0
    for rw in wl:
        ss=wordnet.synsets(rw)
        if ss:
            s=ss[0].lemmas()[0].name()
            nw=[s if w==rw else w for w in nw]; rep+=1
        if rep>=n: break
    return ' '.join(nw)

aug=[]
for label, count in df['Label'].value_counts().items():
    if count<AUG_THRESH:
        sub=df[df['Label']==label]
        while count<AUG_THRESH:
            for _,row in sub.iterrows():
                aug.append({'Label':label,'text':syn_replace(row['text'])})
                count+=1
                if count>=AUG_THRESH: break
df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
print(f"After augmentation: {len(df)} samples")

X_text = df['text'].values; y_labels = df['Label'].values
tok = Tokenizer(num_words=15000, oov_token="<OOV>")
tok.fit_on_texts(X_text)
X_pad = pad_sequences(tok.texts_to_sequences(X_text), maxlen=200,
                      padding='post', truncating='post')
le = LabelEncoder(); y_enc = le.fit_transform(y_labels)
NC = len(le.classes_)
print(f"Classes: {NC}")

# STRICT SPLIT
X_tv, X_te, y_tv, y_te = train_test_split(
    X_pad, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
y_te_ohe = to_categorical(y_te, NC)
cw = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
cw_dict = {i:cw[i] for i in range(len(cw))}

acute_labels = ['Heart Attack','Pneumonia','Tuberculosis','Typhoid','Hepatitis E',
                'Dengue','Malaria','Jaundice','Paralysis (Brain Hemorrhage)',
                'Alcoholic Hepatitis','Drug Reaction','Gastroenteritis',
                'Hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Aids']
acute_idx = [np.where(le.classes_==l)[0][0] for l in acute_labels if l in le.classes_]
print(f"Acute conditions: {len(acute_idx)}")

# ═══════ CELL 3: MODEL DEFINITION ═══════
def build_nlp():
    i=Input(shape=(200,))
    x=Embedding(15000,256,input_length=200)(i)
    x=Conv1D(256,7,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.4)(x)
    x=Conv1D(128,5,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.3)(x)
    x=Bidirectional(LSTM(128,return_sequences=True,dropout=0.3))(x)
    x=Attention()([x,x]); x=Flatten()(x); x=Dropout(0.3)(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.4)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.3)(x)
    o=Dense(NC,activation='softmax')(x)
    m=Model(i,o); m.compile(Adam(3e-5,clipvalue=1.0),'categorical_crossentropy',['accuracy'])
    return m

# ═══════ CELL 4: 5-FOLD CV ═══════
EP, BS = 120, 32  # Optimized for free Colab T4 (~3.5 hrs). Early stopping handles convergence.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_res, fold_pred = [], []
all_yt, all_yp, all_yprob = [], [], []
agg_cm = np.zeros((NC,NC), dtype=int)
best_va, best_fi = -1, -1

for fold,(ti,vi) in enumerate(skf.split(X_tv, y_tv)):
    print(f"\n{'='*55}\n  FOLD {fold+1}\n{'='*55}")
    Xtr,Xvl,ytr,yvl = X_tv[ti],X_tv[vi],y_tv[ti],y_tv[vi]
    ytr_o=to_categorical(ytr,NC); yvl_o=to_categorical(yvl,NC)

    mdl = build_nlp()
    ckpt = f'/content/saved_models/nlp_fold_{fold+1}.keras'
    tcb = TestCB(X_te, y_te_ohe)
    h = mdl.fit(Xtr, ytr_o, validation_data=(Xvl, yvl_o),
                epochs=EP, batch_size=BS, class_weight=cw_dict,
                callbacks=[EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),
                           ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6),
                           ModelCheckpoint(ckpt,monitor='val_accuracy',save_best_only=True,mode='max'),
                           tcb], verbose=1)

    tl,ta = mdl.evaluate(X_te, y_te_ohe, verbose=0)
    yp = mdl.predict(X_te, verbose=0)
    ypc = np.argmax(yp, 1)

    va = h.history['val_accuracy'][-1]
    if va > best_va: best_va=va; best_fi=fold

    cm = confusion_matrix(y_te, ypc, labels=range(NC))
    agg_cm += cm
    all_yt.extend(y_te); all_yp.extend(ypc); all_yprob.append(yp)
    fold_pred.append({'y_true':y_te.copy(), 'y_pred_prob':yp.copy()})

    f1=f1_score(y_te,ypc,average='macro')
    t3=top_k_accuracy_score(y_te,yp,k=3)
    t5=top_k_accuracy_score(y_te,yp,k=5)
    ec,_,_,_=ece_full(y_te,yp)
    br=brier_multi(y_te_ohe,yp)
    try: auc=roc_auc_score(y_te_ohe,yp,average='macro',multi_class='ovr')
    except: auc=0.0
    se=recall_score(y_te,ypc,labels=acute_idx,average='macro',zero_division=0)
    npvs=[]
    for idx in acute_idx:
        tp=cm[idx,idx]; fn=np.sum(cm[idx,:])-tp
        tn=np.sum(cm)-np.sum(cm[idx,:])-np.sum(cm[:,idx])+tp
        npvs.append(tn/(tn+fn) if (tn+fn)>0 else 0)
    npv=np.mean(npvs) if npvs else 0.0

    print(f"  Test: {ta:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"  Top3: {t3:.4f} | Top5: {t5:.4f} | ECE: {ec:.6f} | Brier: {br:.6f}")
    print(f"  Acute Sens: {se:.4f} | NPV: {npv:.4f}")

    fold_res.append({'train_loss':h.history['loss'][-1],'train_acc':h.history['accuracy'][-1],
        'val_loss':h.history['val_loss'][-1],'val_acc':va,'test_loss':tl,'test_acc':ta,
        'macro_f1':f1,'roc_auc':auc,'top3':t3,'top5':t5,'ece':ec,'brier':br,
        'sens_acute':se,'npv_acute':npv})

    # Per-fold curves
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    a1.plot(h.history['loss'],label='Train'); a1.plot(h.history['val_loss'],label='Val')
    a1.plot(tcb.tl,label='Test'); a1.set_title(f'Loss—Fold {fold+1}'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(h.history['accuracy'],label='Train'); a2.plot(h.history['val_accuracy'],label='Val')
    a2.plot(tcb.ta,label='Test'); a2.set_title(f'Acc—Fold {fold+1}'); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'/content/figures/nlp_fold_{fold+1}.png',dpi=200); plt.show()

# Save best fold model
shutil.copy(f'/content/saved_models/nlp_fold_{best_fi+1}.keras',
            '/content/saved_models/best_nlp_model.keras')
print(f"\n✅ Best model: Fold {best_fi+1} (Val Acc: {best_va:.4f})")

# ═══════ CELL 5: RETRAIN FINAL MODEL ON FULL DATA ═══════
print(f"\n{'='*55}\n  RETRAINING FINAL NLP MODEL\n{'='*55}")
y_tv_ohe = to_categorical(y_tv, NC)
final = build_nlp()
final.fit(X_tv, y_tv_ohe, validation_split=0.1, epochs=EP, batch_size=BS,
          class_weight=cw_dict,
          callbacks=[EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6)],
          verbose=1)
fl,fa = final.evaluate(X_te, y_te_ohe, verbose=0)
print(f"  Final Model: Test Acc={fa:.4f} | Loss={fl:.4f}")
final.save('/content/saved_models/final_nlp_model.keras')
with open('/content/saved_models/nlp_tokenizer.pkl','wb') as f: pickle.dump(tok,f)
with open('/content/saved_models/nlp_label_encoder.pkl','wb') as f: pickle.dump(le,f)
print("  Saved model, tokenizer, and label encoder")

# ═══════ CELL 6: Q3 AGGREGATE RESULTS ═══════
print(f"\n{'='*70}\n  Q3: NLP MODEL AGGREGATE METRICS\n{'='*70}")
MK = list(fold_res[0].keys())
rt = {}
for k in MK:
    vs=[r[k] for r in fold_res]; m,s=np.mean(vs),np.std(vs,ddof=1)
    rt[k]={'mean':m,'std':s}
    fmt='.6f' if k in ['ece','brier'] else '.4f'
    print(f"  {k.replace('_',' ').title():<25}: {m:{fmt}} ± {s:{fmt}}")

pd.DataFrame(fold_res, index=[f'Fold {i+1}' for i in range(5)]).to_csv(
    '/content/results/nlp_fold_metrics.csv', float_format='%.6f')

print(f"\n  AGGREGATE CLASSIFICATION REPORT:")
print(classification_report(all_yt, all_yp, target_names=le.classes_))

# Confusion matrix
plt.figure(figsize=(18,15))
sns.heatmap(agg_cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=le.classes_,yticklabels=le.classes_)
plt.title('NLP Model — Aggregate Confusion Matrix',fontsize=13)
plt.xticks(rotation=90,fontsize=6); plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig('/content/figures/nlp_aggregate_cm.png',dpi=300,bbox_inches='tight')
plt.show()

# ROC (using softmax probs — FIXED)
all_prob=np.concatenate(all_yprob); all_true=np.array(all_yt)
all_true_ohe=to_categorical(all_true,NC)
fpr,tpr,rd={},{},{}
for i in range(NC):
    fpr[i],tpr[i],_=roc_curve(all_true_ohe[:,i],all_prob[:,i])
    rd[i]=sk_auc(fpr[i],tpr[i])
afpr=np.unique(np.concatenate([fpr[i] for i in range(NC)]))
mtpr=np.zeros_like(afpr)
for i in range(NC): mtpr+=np.interp(afpr,fpr[i],tpr[i])
mtpr/=NC; mauc=sk_auc(afpr,mtpr)

plt.figure(figsize=(8,7))
plt.plot(afpr,mtpr,color='#E74C3C',linewidth=3,label=f'Macro ROC (AUC={mauc:.4f})')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('NLP Model — ROC (5-Fold)',fontweight='bold')
plt.legend(fontsize=12); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('/content/figures/nlp_roc.png',dpi=300,bbox_inches='tight')
plt.show()

# ═══════ CELL 7: Q4 CALIBRATION & COST-SENSITIVITY ═══════
print(f"\n{'='*70}\n  Q4: CALIBRATION — NLP MODEL\n{'='*70}")

# Reliability diagram
aq4t=np.concatenate([fp['y_true'] for fp in fold_pred])
aq4p=np.concatenate([fp['y_pred_prob'] for fp in fold_pred])
ece_a,ba,bc,bn=ece_full(aq4t,aq4p,15)
fig,(a1,a2)=plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios':[3,1]},sharex=True)
a1.plot([0,1],[0,1],'k--',alpha=0.7); a1.plot(bc,ba,'o-',color='#E74C3C',lw=2,ms=8,
    label=f'NLP Model (ECE={ece_a:.4f})')
a1.fill_between(bc,ba,bc,alpha=0.15,color='#E74C3C')
a1.set_ylabel('Observed Accuracy'); a1.set_title('Reliability Diagram — NLP Model',fontweight='bold')
a1.legend(); a1.grid(alpha=0.3)
conf_all=np.max(aq4p,1)
a2.hist(conf_all,bins=15,range=(0,1),color='#E74C3C',alpha=0.7,edgecolor='white')
a2.set_xlabel('Confidence'); a2.set_ylabel('Count'); a2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/figures/nlp_reliability.png',dpi=300,bbox_inches='tight')
plt.show()

# Threshold sweep
pred_q4=np.argmax(aq4p,1); cor_q4=(pred_q4==aq4t).astype(float)
am=np.isin(aq4t,acute_idx)
sw=[]
for t in np.arange(0.50,1.00,0.025):
    mk=conf_all>=t; cov=mk.mean()
    ac=cor_q4[mk].mean() if mk.sum()>0 else 0
    acm=am&mk; afn=(1-cor_q4[acm].mean()) if acm.sum()>0 else 0
    sw.append({'threshold':t,'coverage':cov,'acc':ac,'acute_fn':afn,'rejection':1-cov})
sdf=pd.DataFrame(sw)
sdf.to_csv('/content/results/nlp_threshold_sweep.csv',index=False)

print("  Threshold Sweep:")
for t in [0.80,0.90,0.95,0.975]:
    r=sdf.loc[(sdf['threshold']-t).abs().idxmin()]
    print(f"  θ={r['threshold']:.3f} → Cov:{r['coverage']:.4f} Acc:{r['acc']:.4f} AcuteFN:{r['acute_fn']:.4f}")

fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
a1.plot(sdf['threshold'],sdf['coverage'],'b-o',ms=4,label='Coverage')
a1.plot(sdf['threshold'],sdf['acc'],'g-s',ms=4,label='Accuracy')
a1.axvline(0.95,color='red',ls='--',alpha=0.7,label='θ=0.95')
a1.legend(); a1.grid(alpha=0.3); a1.set_title('Coverage vs Accuracy',fontweight='bold')
a2.plot(sdf['threshold'],sdf['acute_fn'],'r-^',ms=5,label='Acute FN')
a2.plot(sdf['threshold'],sdf['rejection'],'-D',color='orange',ms=4,label='Rejection')
a2.axvline(0.95,color='red',ls='--',alpha=0.7,label='θ=0.95')
a2.legend(); a2.grid(alpha=0.3); a2.set_title('Safety Analysis',fontweight='bold')
plt.tight_layout()
plt.savefig('/content/figures/nlp_threshold_sweep.png',dpi=300,bbox_inches='tight')
plt.show()

# Cost analysis
AC=['Heart Attack','Paralysis (Brain Hemorrhage)','Aids']
AM=['Malaria','Dengue','Typhoid','Hepatitis E','Pneumonia','Tuberculosis',
    'Alcoholic Hepatitis','Drug Reaction']
CH=['Diabetes','Hypertension','Arthritis','Hypothyroidism','Hyperthyroidism',
    'Hypoglycemia','Osteoarthristis','Cervical Spondylosis','Bronchial Asthma','Gerd']
cfn,cfp={},{}
for i,c in enumerate(le.classes_):
    cfp[i]=1.0
    if c in AC: cfn[i]=10.0
    elif c in AM: cfn[i]=5.0
    elif c in CH: cfn[i]=2.0
    else: cfn[i]=1.0
fc=[]
for fp in fold_pred:
    yt,yp=fp['y_true'],np.argmax(fp['y_pred_prob'],1)
    cost=sum(np.sum((yt==c)&(yp!=c))*cfn[c]+np.sum((yt!=c)&(yp==c))*cfp[c] for c in range(NC))
    fc.append(cost/len(yt))
print(f"\n  Cost/sample: {np.mean(fc):.4f} ± {np.std(fc,ddof=1):.4f}")

print(f"\n{'='*70}\n  COMPLETE NLP SUMMARY\n{'='*70}")
for k in MK:
    m,s=rt[k]['mean'],rt[k]['std']
    fmt='.6f' if k in ['ece','brier'] else '.4f'
    print(f"  {k.replace('_',' ').title():<25}: {m:{fmt}} ± {s:{fmt}}")
print(f"  {'Cost/Sample':<25}: {np.mean(fc):.4f} ± {np.std(fc,ddof=1):.4f}")
print(f"\n✅ NLP Model complete! Models saved in /content/saved_models/")
