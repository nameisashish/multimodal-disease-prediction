# NB2: SYMPTOM ORDERING ANALYSIS (R1-P2) — Clinical vs Shuffled
# No SMOTE (dataset pre-balanced). No model saving (analysis only).
# Run on Colab GPU (~2-3 hrs)

import numpy as np,pandas as pd,matplotlib.pyplot as plt,random,os
from keras.models import Model
from keras.layers import (Input,Dense,Dropout,Conv1D,Flatten,LSTM,
                          Bidirectional,Attention,concatenate,BatchNormalization)
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import f1_score
import warnings;warnings.filterwarnings('ignore')
os.makedirs('/content/figures',exist_ok=True);os.makedirs('/content/results',exist_ok=True)

def ece_fn(yt,yp,n=10):
    c=np.max(yp,1);p=np.argmax(yp,1);a=(p==yt).astype(float);e=0.0
    for i in range(n):
        bl,bu=i/n,(i+1)/n;m=(c>=bl)&(c<=bu) if i==n-1 else (c>=bl)&(c<bu)
        pp=np.mean(m)
        if pp>0:e+=np.abs(np.mean(c[m])-np.mean(a[m]))*pp
    return e

CLINICAL_ORDER=[
    'itching','skin_rash','nodal_skin_eruptions','dischromic _patches',
    'pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze','red_spots_over_body',
    'continuous_sneezing','shivering','chills','cough','breathlessness',
    'phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
    'runny_nose','congestion','loss_of_smell','blood_in_sputum',
    'mucoid_sputum','rusty_sputum','patches_in_throat',
    'stomach_pain','acidity','ulcers_on_tongue','vomiting','indigestion',
    'nausea','abdominal_pain','diarrhoea','constipation',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','belly_pain','passage_of_gases',
    'joint_pain','muscle_wasting','muscle_weakness','muscle_pain',
    'stiff_neck','swelling_joints','movement_stiffness','knee_pain',
    'hip_joint_pain','painful_walking',
    'headache','dizziness','lethargy','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','weakness_in_limbs','slurred_speech',
    'altered_sensorium','lack_of_concentration','visual_disturbances',
    'spinning_movements','neck_pain','back_pain','pain_behind_the_eyes',
    'blurred_and_distorted_vision','watering_from_eyes',
    'chest_pain','fast_heart_rate','palpitations',
    'prominent_veins_on_calf','swollen_blood_vessels',
    'weight_gain','cold_hands_and_feets','mood_swings','weight_loss',
    'irregular_sugar_level','excessive_hunger','increased_appetite',
    'polyuria','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'abnormal_menstruation',
    'yellowish_skin','dark_urine','loss_of_appetite','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','swelling_of_stomach',
    'distention_of_abdomen',
    'burning_micturition','spotting_ urination','bladder_discomfort',
    'foul_smell_of urine','continuous_feel_of_urine',
    'fatigue','restlessness','high_fever','sweating','malaise',
    'dehydration','mild_fever','swelled_lymph_nodes','sunken_eyes',
    'anxiety','irritability','depression','family_history',
    'history_of_alcohol_consumption','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding',
    'fluid_overload','fluid_overload.1','extra_marital_contacts',
    'drying_and_tingling_lips','cramps','bruising','obesity',
    'swollen_legs','puffy_face_and_eyes','internal_itching',
    'toxic_look_(typhos)']

dataset=pd.read_csv('/content/symbipredict_2022.csv')
le=LabelEncoder();y_enc=le.fit_transform(dataset['prognosis'].values);NC=len(le.classes_)
L2R=1e-3

def DenseNetBlock(inp,gr=24,layers=5):
    for _ in range(layers):
        x=Conv1D(gr,3,padding='same',activation='relu',kernel_regularizer=l2(L2R))(inp)
        x=BatchNormalization()(x);x=Dropout(0.5)(x);inp=concatenate([inp,x])
    return inp
def LSTMAttnBlock(inp):
    o=Bidirectional(LSTM(200,return_sequences=True,kernel_regularizer=l2(L2R)))(inp)
    return Attention()([o,o])
def build_model(shape):
    i=Input(shape=shape);dn=i
    for _ in range(3):dn=DenseNetBlock(dn)
    a1=Attention()([dn,dn]);lo=a1
    for _ in range(2):lo=LSTMAttnBlock(lo)
    f=concatenate([Flatten()(a1),Flatten()(lo)])
    x=Dense(512,activation='relu',kernel_regularizer=l2(L2R))(f);x=Dropout(0.6)(x)
    x=Dense(256,activation='relu',kernel_regularizer=l2(L2R))(x);x=Dropout(0.6)(x)
    x=Dense(128,activation='relu',kernel_regularizer=l2(L2R))(x);x=Dropout(0.6)(x)
    o=Dense(NC,activation='softmax')(x)
    m=Model(i,o);m.compile(Adam(1e-3),'categorical_crossentropy',['accuracy']);return m

def run_experiment(cols,name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    X=dataset[cols].values
    Xtv,Xte,ytv,yte=train_test_split(X,y_enc,test_size=0.3,stratify=y_enc,random_state=42)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42);res=[]
    for fold,(ti,vi) in enumerate(skf.split(Xtv,ytv)):
        Xtr,Xvl=Xtv[ti],Xtv[vi];ytr,yvl=ytv[ti],ytv[vi]
        sc=StandardScaler();Xtr=sc.fit_transform(Xtr);Xvl=sc.transform(Xvl);Xtest=sc.transform(Xte)
        # NO SMOTE — dataset balanced
        ytr_o=to_categorical(ytr,NC);yvl_o=to_categorical(yvl,NC);yte_o=to_categorical(yte,NC)
        Xtr=np.expand_dims(Xtr,-1);Xvl=np.expand_dims(Xvl,-1);Xtest=np.expand_dims(Xtest,-1)
        mdl=build_model((Xtr.shape[1],1))
        mdl.fit(Xtr,ytr_o,validation_data=(Xvl,yvl_o),epochs=85,batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                           ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,min_lr=1e-7)],verbose=0)
        tl,ta=mdl.evaluate(Xtest,yte_o,verbose=0)
        yp=mdl.predict(Xtest,verbose=0);ece=ece_fn(yte,yp)
        f1=f1_score(yte,np.argmax(yp,1),average='macro')
        print(f"  F{fold+1}: Acc={ta:.4f} F1={f1:.4f} ECE={ece:.5f} Loss={tl:.4f}")
        res.append({'test_acc':ta,'test_loss':tl,'macro_f1':f1,'ece':ece})
    return res

clinical_res=run_experiment(CLINICAL_ORDER,"CLINICAL BODY-SYSTEM ORDERING")
shuffled=CLINICAL_ORDER.copy();random.seed(42);random.shuffle(shuffled)
shuffled_res=run_experiment(shuffled,"SHUFFLED RANDOM ORDERING")

def sm(r):
    d={}
    for k in r[0]:vs=[x[k] for x in r];d[k]=f"{np.mean(vs):.4f} ± {np.std(vs,ddof=1):.4f}"
    return d

print(f"\n{'='*70}\n  COMPARISON TABLE\n{'='*70}")
cdf=pd.DataFrame([{'Ordering':'Clinical',**sm(clinical_res)},
                   {'Ordering':'Shuffled',**sm(shuffled_res)}]).set_index('Ordering')
print(cdf.to_string());cdf.to_csv('/content/results/ordering_comparison.csv')

fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,k,t in zip(axes,['test_acc','test_loss','macro_f1'],['Test Accuracy','Test Loss','Macro-F1']):
    cv=[r[k] for r in clinical_res];sv=[r[k] for r in shuffled_res]
    x=np.arange(5);w=0.35
    ax.bar(x-w/2,cv,w,label='Clinical',color='#2E75B6')
    ax.bar(x+w/2,sv,w,label='Shuffled',color='#E67E22')
    ax.set_xlabel('Fold');ax.set_ylabel(t);ax.set_title(t)
    ax.set_xticks(x);ax.set_xticklabels([f'F{i+1}' for i in range(5)]);ax.legend()
plt.suptitle('Impact of Symptom Ordering',fontsize=14,fontweight='bold')
plt.tight_layout();plt.savefig('/content/figures/ordering_comparison.png',dpi=300,bbox_inches='tight');plt.show()
print("\n✅ Ordering analysis complete!")
