import os
import numpy as np
import pickle
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Health Checker Dual Model API")

# Allow all origins (since it will be hit from Vercel Serverless Functions / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Paths (Relative to app.py in Hugging Face Docker) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYM_DIR = os.path.join(BASE_DIR, 'models', 'symptom_model')
SYM_MODEL_PATH   = os.path.join(SYM_DIR, 'final_symptom_model.keras')
SYM_SCALER_PATH  = os.path.join(SYM_DIR, 'symptom_scaler.pkl')
SYM_ENCODER_PATH = os.path.join(SYM_DIR, 'symptom_label_encoder.pkl')
SYM_TEMP_PATH    = os.path.join(SYM_DIR, 'symptom_temperature.pkl')

NLP_DIR = os.path.join(BASE_DIR, 'models', 'nlp_model')
NLP_MODEL_PATH     = os.path.join(NLP_DIR, 'best_nlp_model.keras')
NLP_TOKENIZER_PATH = os.path.join(NLP_DIR, 'nlp_tokenizer.pkl')
NLP_ENCODER_PATH   = os.path.join(NLP_DIR, 'nlp_label_encoder.pkl')

# ─── Load Models at Startup ──────────────────────────────────────────────────
# Symptom Model
sym_model  = load_model(SYM_MODEL_PATH)
sym_scaler = pickle.load(open(SYM_SCALER_PATH, 'rb'))
sym_le     = pickle.load(open(SYM_ENCODER_PATH, 'rb'))
try:    
    SYM_TEMP = pickle.load(open(SYM_TEMP_PATH, 'rb'))
except: 
    SYM_TEMP = 1.0

# NLP Model
nlp_model     = load_model(NLP_MODEL_PATH)
nlp_tokenizer = pickle.load(open(NLP_TOKENIZER_PATH, 'rb'))
nlp_le        = pickle.load(open(NLP_ENCODER_PATH, 'rb'))
NLP_MAXLEN    = 38

# Symptom list (exact training column order)
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
    'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
    'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
]
N_SYMS = len(SYMPTOMS)

# Symptom lookup (case + space/underscore tolerant)
SYM_LOOKUP = {}
for i, s in enumerate(SYMPTOMS):
    SYM_LOOKUP[s.lower()] = i
    SYM_LOOKUP[s.lower().replace('_', ' ')] = i


class PredictRequest(BaseModel):
    symptoms: str


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Health Checker Dual Model Inference API is running."}


@app.post("/predict")
def predict_disease(request: PredictRequest):
    """Run both models locally, return the one with higher confidence."""
    user_input = request.symptoms
    symptoms_raw = [s.strip().lower() for s in user_input.split(',') if s.strip()]
    
    if not symptoms_raw:
        raise HTTPException(status_code=400, detail="No symptoms provided")

    # ─── SYMPTOM MODEL ───────────────────────────────────────────────────
    vec = np.zeros(N_SYMS, dtype=np.float32)
    matched_syms, unmatched_syms = [], []
    for sym in symptoms_raw:
        sym_clean = sym.strip()
        if sym_clean in SYM_LOOKUP:
            vec[SYM_LOOKUP[sym_clean]] = 1.0
            matched_syms.append(sym_clean)
        elif sym_clean.replace(' ', '_') in SYM_LOOKUP:
            vec[SYM_LOOKUP[sym_clean.replace(' ', '_')]] = 1.0
            matched_syms.append(sym_clean)
        else:
            unmatched_syms.append(sym_clean)

    sym_result = None
    if matched_syms:
        X = np.expand_dims(sym_scaler.transform(vec.reshape(1, -1)), -1)
        probs = sym_model.predict(X, verbose=0)[0]
        if SYM_TEMP != 1.0:
            probs = np.exp(np.log(probs + 1e-12) / SYM_TEMP)
            probs = probs / probs.sum()
        top3_idx = np.argsort(probs)[-3:][::-1]
        sym_result = {
            'disease': str(sym_le.classes_[top3_idx[0]]),
            'confidence': round(float(probs[top3_idx[0]]) * 100, 2),
            'matched': matched_syms,
            'unmatched': unmatched_syms,
        }

    # ─── NLP MODEL ───────────────────────────────────────────────────────
    text = user_input.strip().lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    seq = nlp_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=NLP_MAXLEN, padding='post', truncating='post')
    nlp_probs = nlp_model.predict(padded, verbose=0)[0]
    top3_nlp = np.argsort(nlp_probs)[-3:][::-1]
    
    nlp_result = {
        'disease': str(nlp_le.classes_[top3_nlp[0]]),
        'confidence': round(float(nlp_probs[top3_nlp[0]]) * 100, 2),
    }

    # ─── WINNER ──────────────────────────────────────────────────────────
    sym_conf = sym_result['confidence'] if sym_result else 0.0
    nlp_conf = nlp_result['confidence']

    if sym_result and sym_conf >= nlp_conf:
        winner = 'SYMPTOM MODEL'
        final_disease = sym_result['disease']
        final_conf = sym_result['confidence']
    else:
        winner = 'NLP MODEL'
        final_disease = nlp_result['disease']
        final_conf = nlp_result['confidence']

    return {
        'predicted_disease': final_disease,
        'confidence': final_conf,
        'winner': winner,
        'symptoms': symptoms_raw,
        'symptom_model': sym_result,
        'nlp_model': nlp_result,
    }
