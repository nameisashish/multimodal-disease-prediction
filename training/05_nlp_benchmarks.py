"""
================================================================================
NB5: NLP BENCHMARKS — 5-FOLD CV (Reviewer 1, Point 3)
================================================================================
Models: CNN, XGBoost, GRU+Attention, TextCNN+Attention
Run AFTER NB4 so variables exist, OR paste NB4 Cells 1-2 for standalone.
Values only — no model saving.
================================================================================
"""

# Assumes these from NB4: X_tv, X_te, y_tv, y_te, y_te_ohe, NC, le,
#                         acute_idx, cw_dict

import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D,
                          GRU, Attention, Embedding, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, top_k_accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')
os.makedirs('/content/results',exist_ok=True)
os.makedirs('/content/figures',exist_ok=True)

def ece_fn(yt,yp,n=10):
    c=np.max(yp,1); p=np.argmax(yp,1); a=(p==yt).astype(float); e=0.0
    for i in range(n):
        bl,bu=i/n,(i+1)/n
        m=(c>=bl)&(c<=bu) if i==n-1 else (c>=bl)&(c<bu)
        pp=np.mean(m)
        if pp>0: e+=np.abs(np.mean(c[m])-np.mean(a[m]))*pp
    return e
def brier(yo,yp): return np.mean(np.sum((yp-yo)**2,1))

# ── Model Builders ──
def mk_cnn():
    i=Input(shape=(200,))
    x=Embedding(15000,256,input_length=200)(i)
    x=Conv1D(256,7,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.4)(x)
    x=Conv1D(128,5,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.3)(x)
    x=Flatten()(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.4)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.3)(x)
    o=Dense(NC,activation='softmax')(x)
    m=Model(i,o); m.compile(Adam(3e-5,clipvalue=1.0),'categorical_crossentropy',['accuracy']); return m

def mk_gru_attn():
    i=Input(shape=(200,))
    x=Embedding(15000,256,input_length=200)(i)
    x=Conv1D(256,7,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.4)(x)
    x=GRU(128,return_sequences=True,kernel_regularizer=l2(0.01))(x)
    x=Attention()([x,x]); x=Flatten()(x); x=Dropout(0.3)(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.4)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.3)(x)
    o=Dense(NC,activation='softmax')(x)
    m=Model(i,o); m.compile(Adam(3e-5,clipvalue=1.0),'categorical_crossentropy',['accuracy']); return m

def mk_textcnn_attn():
    i=Input(shape=(200,))
    x=Embedding(15000,256,input_length=200)(i)
    x=Conv1D(256,7,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.4)(x)
    x=Conv1D(128,5,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=MaxPooling1D(2)(x); x=Dropout(0.3)(x)
    x=Attention()([x,x]); x=Flatten()(x); x=Dropout(0.3)(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.4)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(0.01))(x)
    x=BatchNormalization()(x); x=Dropout(0.3)(x)
    o=Dense(NC,activation='softmax')(x)
    m=Model(i,o); m.compile(Adam(3e-5,clipvalue=1.0),'categorical_crossentropy',['accuracy']); return m

# ── Runner ──
def bench(name, builder, is_keras=True, ep=80, bs=32):  # Optimized for free Colab T4
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    res=[]
    for fold,(ti,vi) in enumerate(skf.split(X_tv,y_tv)):
        Xtr,Xvl,ytr,yvl=X_tv[ti],X_tv[vi],y_tv[ti],y_tv[vi]
        if is_keras:
            mdl=builder()
            mdl.fit(Xtr,to_categorical(ytr,NC),
                    validation_data=(Xvl,to_categorical(yvl,NC)),
                    epochs=ep,batch_size=bs,class_weight=cw_dict,
                    callbacks=[EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),
                               ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6)],
                    verbose=0)
            tl,ta=mdl.evaluate(X_te,y_te_ohe,verbose=0)
            yp=mdl.predict(X_te,verbose=0)
        else:
            Xtr_f=Xtr.reshape(Xtr.shape[0],-1)
            Xte_f=X_te.reshape(X_te.shape[0],-1)
            mdl=builder(); mdl.fit(Xtr_f,ytr)
            ta=mdl.score(Xte_f,y_te); yp=mdl.predict_proba(Xte_f)
            tl=-np.mean(np.sum(y_te_ohe*np.log(yp+1e-10),1))

        ypc=np.argmax(yp,1)
        f1=f1_score(y_te,ypc,average='macro')
        t3=top_k_accuracy_score(y_te,yp,k=3)
        t5=top_k_accuracy_score(y_te,yp,k=5)
        ec=ece_fn(y_te,yp); br=brier(y_te_ohe,yp)
        se=recall_score(y_te,ypc,labels=acute_idx,average='macro',zero_division=0)
        try: auc=roc_auc_score(y_te_ohe,yp,average='macro',multi_class='ovr')
        except: auc=0
        print(f"  F{fold+1}: Acc={ta:.4f} F1={f1:.4f} AUC={auc:.4f}")
        res.append({'test_acc':ta,'test_loss':tl,'macro_f1':f1,'roc_auc':auc,
                    'top3':t3,'top5':t5,'ece':ec,'brier':br,'sens_acute':se})
    print(f"\n  Summary:")
    for k in res[0]:
        vs=[r[k] for r in res]
        print(f"    {k:<15}: {np.mean(vs):.4f} ± {np.std(vs,ddof=1):.4f}")
    return res

# ── Run ──
R={}
R['CNN'] = bench('CNN', mk_cnn)
R['XGBoost'] = bench('XGBoost',
    lambda: XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=6,
                          random_state=42,use_label_encoder=False,eval_metric='mlogloss'),
    is_keras=False)
R['GRU+Attention'] = bench('GRU + Attention', mk_gru_attn)
R['TextCNN+Attention'] = bench('TextCNN + Attention', mk_textcnn_attn)

# ── Table ──
print(f"\n{'='*80}\n  NLP BENCHMARKS TABLE\n{'='*80}")
rows=[]
for nm,res in R.items():
    row={'Model':nm}
    for k in res[0]:
        vs=[r[k] for r in res]
        row[k]=f"{np.mean(vs):.4f} ± {np.std(vs,ddof=1):.4f}"
    rows.append(row)
df=pd.DataFrame(rows).set_index('Model')
print(df.to_string())
df.to_csv('/content/results/nlp_benchmark_comparison.csv')

# Chart
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,mk,tt in zip(axes,['test_acc','macro_f1','ece'],['Test Acc','Macro-F1','ECE (↓)']):
    nms=list(R.keys())
    ms=[np.mean([r[mk] for r in R[n]]) for n in nms]
    ss=[np.std([r[mk] for r in R[n]],ddof=1) for n in nms]
    cs=['#3498DB','#E67E22','#2ECC71','#9B59B6']
    ax.bar(range(len(nms)),ms,yerr=ss,color=cs,capsize=5,edgecolor='white')
    ax.set_xticks(range(len(nms))); ax.set_xticklabels(nms,rotation=20,ha='right',fontsize=9)
    ax.set_title(tt,fontweight='bold'); ax.grid(alpha=0.3,axis='y')
plt.suptitle('NLP Benchmarks (5-Fold CV)',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('/content/figures/nlp_benchmarks.png',dpi=300,bbox_inches='tight')
plt.show()
print("\n✅ NLP benchmarks complete!")
