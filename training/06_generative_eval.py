"""
================================================================================
NB6: GENERATIVE MODULE EVALUATION (Reviewer 1, Point 5)
================================================================================
Part A: InterceptionLogger — add to your Flask/AyuSeva pipeline
Part B: Clinical evaluation template generator for doctors
Part C: Rating analysis code (after doctors return filled forms)

No GPU needed. Run locally or on Colab.
================================================================================
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, json, os
os.makedirs('/content/results', exist_ok=True)

# ═══════════════════════════════════════════════════
# PART A: INTERCEPTION RATE LOGGER
# ═══════════════════════════════════════════════════
class InterceptionLogger:
    """
    Add to your Flask/AyuSeva pipeline to log every prediction event.
    Tracks: threshold rejections, Gemini activations, hallucination flags, fallbacks.

    USAGE IN YOUR FLASK APP:
        logger = InterceptionLogger()
        # After each prediction:
        logger.log(
            disease="Heart Attack", confidence=0.987,
            input_type="structured",  # or "nlp"
            gemini_activated=True, gemini_flagged=False
        )
        # After session:
        logger.report()
        logger.save('/content/results/interception_log.json')
    """
    def __init__(self, threshold=0.95):
        self.events = []
        self.threshold = threshold

    def log(self, disease, confidence, input_type='structured',
            gemini_activated=False, gemini_flagged=False, fallback_used=False):
        self.events.append({
            'disease': disease, 'confidence': float(confidence),
            'input_type': input_type,
            'passed_threshold': confidence >= self.threshold,
            'gemini_activated': gemini_activated,
            'gemini_flagged': gemini_flagged,
            'fallback_used': fallback_used
        })

    def report(self):
        if not self.events:
            print("No events logged."); return {}
        df = pd.DataFrame(self.events)
        n = len(df)
        r = {
            'total': n,
            'passed': df['passed_threshold'].sum(),
            'rejected': (~df['passed_threshold']).sum(),
            'rejection_rate': (~df['passed_threshold']).mean(),
            'gemini_activated': df['gemini_activated'].sum(),
            'gemini_flagged': df['gemini_flagged'].sum(),
            'flag_rate': df.loc[df['gemini_activated'], 'gemini_flagged'].mean()
                         if df['gemini_activated'].any() else 0,
            'fallback_count': df['fallback_used'].sum(),
            'fallback_rate': df['fallback_used'].mean(),
            'mean_confidence': df['confidence'].mean(),
        }
        for it in ['structured', 'nlp']:
            sub = df[df['input_type']==it]
            if len(sub)>0:
                r[f'{it}_rejection'] = (~sub['passed_threshold']).mean()
                r[f'{it}_mean_conf'] = sub['confidence'].mean()

        print(f"\n{'='*55}")
        print(f"  INTERCEPTION RATE REPORT (θ = {self.threshold})")
        print(f"{'='*55}")
        print(f"  Total predictions:     {r['total']}")
        print(f"  Passed threshold:      {r['passed']} ({1-r['rejection_rate']:.1%})")
        print(f"  Rejected:              {r['rejected']} ({r['rejection_rate']:.1%})")
        print(f"  Gemini flagged:        {r['gemini_flagged']} ({r['flag_rate']:.1%} of activated)")
        print(f"  Fallback used:         {r['fallback_count']} ({r['fallback_rate']:.1%})")
        print(f"  Mean confidence:       {r['mean_confidence']:.4f}")
        if 'structured_rejection' in r:
            print(f"  Structured rejection:  {r['structured_rejection']:.1%}")
        if 'nlp_rejection' in r:
            print(f"  NLP rejection:         {r['nlp_rejection']:.1%}")
        return r

    def save(self, path):
        with open(path, 'w') as f: json.dump(self.events, f, indent=2)
        print(f"  Saved: {path}")

    def visualize(self):
        if not self.events: return
        df = pd.DataFrame(self.events)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Confidence distribution
        axes[0].hist(df['confidence'], bins=20, color='#2E75B6', edgecolor='white', alpha=0.8)
        axes[0].axvline(self.threshold, color='red', ls='--', lw=2, label=f'θ={self.threshold}')
        axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
        axes[0].set_title('Prediction Confidence Distribution'); axes[0].legend()

        # Rejection by input type
        for it in ['structured', 'nlp']:
            sub = df[df['input_type']==it]
            if len(sub)>0:
                rates = [sub['passed_threshold'].mean(), (~sub['passed_threshold']).mean()]
                axes[1].bar([f'{it}\nPassed', f'{it}\nRejected'], rates,
                           color=['#27AE60', '#E74C3C'], alpha=0.7)
        axes[1].set_title('Pass/Reject by Input Type'); axes[1].set_ylabel('Rate')

        # Gemini outcomes
        activated = df['gemini_activated'].sum()
        flagged = df['gemini_flagged'].sum()
        clean = activated - flagged
        axes[2].bar(['Clean\nResponses', 'Flagged\n(Halluc.)', 'Not\nActivated'],
                   [clean, flagged, len(df)-activated],
                   color=['#27AE60', '#E74C3C', '#95A5A6'])
        axes[2].set_title('Gemini Output Outcomes'); axes[2].set_ylabel('Count')

        plt.suptitle('Interception Rate Analysis', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/content/figures/interception_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


# ═══════════════════════════════════════════════════
# PART B: CLINICAL EVALUATION TEMPLATE
# ═══════════════════════════════════════════════════
def generate_evaluation_template(le=None, num_per_severity=10):
    """
    Generate a CSV template for clinician evaluation.
    Distribute to 3 doctors. Each fills in ratings.

    Args:
        le: fitted LabelEncoder (pass from NB1 or NB4)
        num_per_severity: cases per severity level
    """
    categories = {
        'Acute-Critical': ['Heart Attack', 'Paralysis (brain hemorrhage)', 'AIDS'],
        'Acute-Moderate': ['Malaria', 'Dengue', 'Typhoid', 'Hepatitis E', 'Pneumonia',
                           'Tuberculosis', 'Alcoholic Hepatitis', 'Drug Reaction'],
        'Chronic': ['Diabetes ', 'Hypertension ', 'Arthritis', 'Osteoarthritis',
                    'Hypothyroidism', 'Bronchial Asthma', 'GERD', 'Cervical Spondylosis'],
        'Mild': ['Common Cold', 'Acne', 'Allergy', 'Impetigo', 'Fungal Infection',
                 'Migraine', 'Varicose Veins', 'Chicken Pox']
    }

    cases = []
    cid = 1
    for severity, diseases in categories.items():
        if le is not None:
            diseases = [d for d in diseases if d in le.classes_]
        selected = (diseases * ((num_per_severity // len(diseases)) + 1))[:num_per_severity]
        for disease in selected:
            cases.append({
                'Case_ID': f'C{cid:03d}',
                'Disease': disease,
                'Severity': severity,
                'Sample_Symptoms': '[Run AyuSeva and paste symptoms used]',
                'AyuSeva_Response': '[Paste full generated response here]',
                'Medical_Accuracy_1to5': '',
                'Completeness_1to5': '',
                'Empathy_Tone_1to5': '',
                'Safety_1to5': '',
                'Clinical_Utility_1to5': '',
                'Hallucination_YesNo': '',
                'Emergency_Appropriate_YesNoNA': '',
                'Evaluator_Notes': ''
            })
            cid += 1

    df = pd.DataFrame(cases)
    path = '/content/results/clinical_evaluation_template.csv'
    df.to_csv(path, index=False)
    print(f"\n✅ Template saved: {path}")
    print(f"   {len(df)} cases across {len(categories)} severity levels")
    print(f"   Distribute to 3 doctors for independent rating\n")

    # Print rubric
    print("""
╔══════════════════════════════════════════════════════════════╗
║              EVALUATION RUBRIC FOR DOCTORS                   ║
╚══════════════════════════════════════════════════════════════╝

Rate each AyuSeva response on these dimensions:

1. MEDICAL ACCURACY (1-5)
   1=Multiple errors | 3=Mostly accurate | 5=Completely accurate

2. COMPLETENESS (1-5)
   1=Misses most info | 3=Major domains covered | 5=Fully comprehensive

3. EMPATHY & TONE (1-5)
   1=Robotic/alarmist | 3=Appropriate | 5=Excellent emotional intelligence

4. SAFETY (1-5)
   1=Dangerous | 3=Safe but incomplete | 5=Full safety + emergency protocols

5. CLINICAL UTILITY (1-5)
   1=Not useful | 3=Moderate utility | 5=Excellent for triage

6. HALLUCINATION (Yes/No)
   Yes = Contains info NOT linked to the diagnosed condition

7. EMERGENCY APPROPRIATE (Yes/No/NA)
   For acute conditions: Did the system correctly identify emergency?
""")
    return df


# ═══════════════════════════════════════════════════
# PART C: ANALYZE COLLECTED RATINGS
# ═══════════════════════════════════════════════════
def analyze_ratings(csv_paths):
    """
    Analyze ratings from multiple evaluators.
    Args: csv_paths: list of CSV file paths (one per doctor)
    """
    evals = [pd.read_csv(p) for p in csv_paths]
    n_eval = len(evals)
    rcols = ['Medical_Accuracy_1to5','Completeness_1to5','Empathy_Tone_1to5',
             'Safety_1to5','Clinical_Utility_1to5']

    print(f"\n{'='*60}")
    print(f"  CLINICAL EVALUATION RESULTS ({n_eval} Evaluators)")
    print(f"{'='*60}")

    # Mean ± SD per dimension
    print(f"\n  {'Dimension':<25} {'Mean':>8} {'±SD':>10}")
    print(f"  {'-'*45}")
    for col in rcols:
        all_r = np.concatenate([e[col].dropna().values for e in evals])
        print(f"  {col.replace('_1to5',''):<25} {np.mean(all_r):>8.2f} ± {np.std(all_r,ddof=1):>7.2f}")

    # Hallucination rate
    all_h = np.concatenate([(e['Hallucination_YesNo'].str.lower()=='yes').values for e in evals])
    print(f"\n  Hallucination Rate: {np.mean(all_h):.1%} ({np.sum(all_h)}/{len(all_h)})")

    # Inter-rater reliability
    if n_eval >= 2:
        from sklearn.metrics import cohen_kappa_score
        print(f"\n  Inter-Rater Reliability (Weighted Kappa):")
        for i in range(n_eval):
            for j in range(i+1, n_eval):
                kappas = []
                for col in rcols:
                    r1 = evals[i][col].dropna().values.astype(int)
                    r2 = evals[j][col].dropna().values.astype(int)
                    mn = min(len(r1), len(r2))
                    if mn > 0:
                        k = cohen_kappa_score(r1[:mn], r2[:mn], weights='quadratic')
                        kappas.append(k)
                if kappas:
                    print(f"    Evaluator {i+1} vs {j+1}: mean κ = {np.mean(kappas):.3f}")

    # Per-severity
    combined = pd.concat(evals)
    if 'Severity' in combined.columns:
        print(f"\n  Per-Severity Means:")
        for sev in ['Acute-Critical','Acute-Moderate','Chronic','Mild']:
            sub = combined[combined['Severity']==sev]
            if len(sub)>0:
                print(f"    {sev}: " + " | ".join(
                    f"{c.replace('_1to5','')}={sub[c].mean():.1f}" for c in rcols))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmeans = {c.replace('_1to5',''): np.concatenate([e[c].dropna().values for e in evals]).mean()
              for c in rcols}
    cstds = {c.replace('_1to5',''): np.concatenate([e[c].dropna().values for e in evals]).std(ddof=1)
             for c in rcols}
    labels = list(cmeans.keys()); ms = list(cmeans.values()); ss = list(cstds.values())
    cs = ['#2E75B6','#27AE60','#E67E22','#C0392B','#8E44AD']
    ax1.bar(range(len(labels)), ms, yerr=ss, color=cs, capsize=5, edgecolor='white')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([l.replace('_','\n') for l in labels], fontsize=9)
    ax1.set_ylim(0, 5.5); ax1.axhline(4, color='green', ls='--', alpha=0.5)
    ax1.set_title('Clinical Evaluation Ratings', fontweight='bold')
    ax1.set_ylabel('Rating (1-5)'); ax1.grid(alpha=0.3, axis='y')

    if 'Severity' in combined.columns:
        sevs = ['Acute-Critical','Acute-Moderate','Chronic','Mild']
        hrs = []
        for sev in sevs:
            sub = combined[combined['Severity']==sev]
            hrs.append((sub['Hallucination_YesNo'].str.lower()=='yes').mean() if len(sub)>0 else 0)
        ax2.bar(sevs, hrs, color=['#C0392B','#E67E22','#F1C40F','#27AE60'], edgecolor='white')
        ax2.set_title('Hallucination Rate by Severity', fontweight='bold')
        ax2.set_ylabel('Rate'); ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/content/figures/clinical_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

# ═══════ GENERATE OUTPUTS ═══════
print("NB6 loaded. Available functions:")
print("  1. InterceptionLogger() — add to Flask app")
print("  2. generate_evaluation_template(le) — create doctor template")
print("  3. analyze_ratings([path1, path2, path3]) — analyze filled forms")
print()

# Generate template (uncomment when le is available):
# generate_evaluation_template(le, num_per_severity=10)

# Demo the logger:
demo = InterceptionLogger(threshold=0.95)
demo.log("Heart Attack", 0.987, "structured", True, False)
demo.log("Common Cold", 0.72, "nlp", False, False)
demo.log("Dengue", 0.963, "nlp", True, True, True)
demo.log("Arthritis", 0.891, "structured", False, False)
demo.log("AIDS", 0.998, "structured", True, False)
demo.report()
print("\n✅ NB6 ready. Generate template and distribute to doctors.")
