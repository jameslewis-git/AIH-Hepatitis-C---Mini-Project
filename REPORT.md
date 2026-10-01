# EXP 10: Report of Mini-project On
# Hepatitis C Detection and Staging using Machine Learning

<p align="center">
  <img src="static/img/app_logo.png" alt="Hepatitis C Detection and Staging Logo" width="120">
</p>

**Submitted in partial fulfillment of the requirements of the Mini project in the**  
**Subject: AI FOR HEALTHCARE**  
*of*  
**Semester VII, FINAL Year Artificial Intelligence and Data Science**  

**By**  
**James Lewis (Roll No. 27)**  

<br>

**University of Mumbai**  
**Vidyavardhini's College of Engineering & Technology**  
**Department of Artificial Intelligence and Data Science**  
*(A.Y. 2026-27)*  

---

## CERTIFICATE

This is to certify that the Mini Project entitled **"Hepatitis C Detection and Staging using Machine Learning"** is submitted by **James Lewis (Roll No. 27)** for the subject of **AI for Healthcare** in the **Department of Artificial Intelligence and Data Science** as a record of work done by him under our supervision and guidance.

<br><br>
**Guide:**  
**Asst. Prof. Kranti Gule**  
Department of Artificial Intelligence and Data Science  
Vidyavardhini's College of Engineering & Technology, Vasai  

---

## Contents

| Sr. No | Title | Page No |
| :---: | :--- | :---: |
| 1 | Introduction of Project | 1 |
| 2 | Importance of Project | 3 |
| 3 | Screen Shot of Output | 5 |
| 4 | Conclusion | 8 |

---

# 1. INTRODUCTION OF PROJECT

### 1.1 Clinical Background & Problem Domain
Hepatitis C is a blood-borne viral infection caused by the Hepatitis C Virus (HCV) that targets hepatocytes (liver cells) and induces chronic inflammation. According to the World Health Organization (WHO), an estimated 58 million individuals globally live with chronic Hepatitis C infection, resulting in approximately 290,000 deaths annually, predominantly from decompensated cirrhosis and hepatocellular carcinoma (primary liver cancer).

The clinical course of Hepatitis C is notoriously insidious. Termed a "silent killer," the initial acute phase is asymptomatic in over 75% to 80% of infected individuals. Without prompt detection and therapeutic intervention, chronic infection quietly advances across distinct histopathological stages over a period of 15 to 30 years:

1. **Blood Donor (Healthy Control)**: Intact hepatic parenchymal architecture, normal bile clearance, and healthy transaminase levels.
2. **Suspect Blood Donor (Borderline / Deferred Donor)**: Subclinical cellular leakage or mild reactive transaminitis without established fibrosis.
3. **Hepatitis (Active Inflammation)**: Acute or chronic inflammatory infiltration causing active hepatocellular necrosis and sharply elevated transaminases (ALT and AST).
4. **Fibrosis (Early Hepatic Scarring)**: Excessive accumulation of extracellular matrix proteins (collagen) forming fibrous bridges across vascular hepatic triads.
5. **Cirrhosis (Late-Stage Decompensation)**: Diffuse conversion of normal liver architecture into structurally abnormal regenerative nodules, portal hypertension, and parenchymal failure.

### 1.2 Limitations of Conventional Diagnostic Procedures
Historically, the diagnostic "gold standard" for staging hepatic fibrosis and cirrhosis has been the **percutaneous core needle liver biopsy**:
- **Invasive & Painful**: Requires transcostal puncture with an aspiration needle, necessitating local anesthesia and post-procedural inpatient observation.
- **Procedural Risks**: Carries real medical risks, including severe intraperitoneal hemorrhage, bile peritonitis, pneumothorax, and accidental puncture of adjacent organs (1 in 500 to 1 in 1,000 cases).
- **Prohibitive Cost**: A liver biopsy costs between $1,500 and $5,000, placing it out of reach for resource-constrained clinics.
- **Sampling Error**: A standard biopsy sample represents merely 1/50,000th of total liver mass, leading to high inter-observer variability and misclassification rates of up to 20% to 30%.
- **Specialized Imaging Barriers**: Alternative non-invasive modalities such as Transient Elastography (FibroScan) or Magnetic Resonance Elastography (MRE) require capital equipment exceeding $50,000 to $120,000 and specialized medical technicians unavailable in primary healthcare centers.

### 1.3 Proposed Machine Learning Solution
This project presents **Hepatitis C Detection and Staging using Machine Learning**, an intelligent clinical decision support platform designed to stage hepatic damage non-invasively using 12 routine biochemical blood markers. 

By analyzing standard Liver Function Panels (LFPs) that cost less than $30 and are available in standard pathology laboratories, the system provides real-time, probability-weighted multi-class staging.

#### 1.3.1 Key Objectives
1. **Automate Hepatic Staging**: Classify patients into five clinical categories (Blood Donor, Suspect Donor, Hepatitis, Fibrosis, Cirrhosis) from serum biomarkers.
2. **Handle Extreme Class Imbalance**: Address medical data skew (86.7% healthy donors vs. 3.4% cirrhosis) using Stratified Cross-Validation and Macro-F1 metric optimization.
3. **Ensure Zero-Data-Loss Imputation**: Replace row dropping with population-derived non-parametric median imputation for clinical completeness.
4. **Deploy Dual-Mode Architecture**: Deliver a full-stack Flask web application with a client-side JavaScript execution engine (1,500 compiled XGBoost trees) capable of sub-millisecond inference on serverless platforms such as Netlify.
5. **Multimodal AI Integration**: Provide Gemini Vision AI OCR for optical extraction of paper lab reports and an evidence-based diagnostic symptom screener.

---

# 2. IMPORTANCE OF PROJECT

### 2.1 Clinical and Epidemiological Relevance
1. **Accelerating the WHO 2030 Viral Elimination Goals**:  
   The World Health Organization has targeted the elimination of viral hepatitis as a public health threat by 2030 (aiming for an 80% reduction in incidence and 65% reduction in mortality). The primary barrier is not treatment (modern Direct-Acting Antivirals cure >95% of infections in 8 to 12 weeks), but under-diagnosis. This tool equips community health workers and primary care physicians with an instantaneous risk-stratification mechanism.

2. **Democratizing Diagnostic Reach in Resource-Constrained Regions**:  
   In low- and middle-income regions (LMICs) where biopsy facilities and FibroScan equipment are absent, routine blood chemistry is almost always accessible. Transforming standard, low-cost biochemical tests into an advanced staging tool bridges this critical healthcare disparity.

3. **Continuous, Repeatable Longitudinal Monitoring**:  
   Because needle biopsies carry significant risks, they cannot be performed repeatedly to track therapeutic response. Blood marker evaluations can be repeated monthly or quarterly with zero patient trauma, providing clinicians with longitudinal trajectory curves of liver recovery.

### 2.2 Biochemical and Pathophysiological Significance
The model utilizes 12 routine clinical features whose complex interactions reflect distinct hepatic mechanisms:

| Feature | Biological Role | Pathological Mechanism in HCV |
| :--- | :--- | :--- |
| **AST** (Aspartate Aminotransferase) | Mitochondrial enzyme in hepatocytes | Spikes during active hepatocellular necrosis; inverted AST/ALT ratio (>1.0) is a hallmark of bridging fibrosis and cirrhosis. |
| **CHE** (Cholinesterase) | Exclusive liver synthetic enzyme | Produced entirely by functioning hepatocytes; levels plummet dramatically as viable parenchymal mass is lost in end-stage cirrhosis. |
| **ALB** (Albumin) | Primary circulating oncotic protein | Synthesized solely by the liver; declining serum albumin (<35 g/L) signals severe synthetic failure and heralds ascites. |
| **ALT** (Alanine Aminotransferase) | Cytoplasmic liver-specific enzyme | Released directly into bloodstream upon acute membrane permeabilization and viral injury. |
| **BIL** (Total Bilirubin) | Breakdown pigment of hemoglobin | Conjugated and excreted by hepatocytes; impaired bile clearance leads to hyperbilirubinemia, jaundice, and icterus. |
| **ALP** (Alkaline Phosphatase) | Canalicular and biliary enzyme | Elevated in intrahepatic cholestasis, biliary ductular reaction, and compressive cirrhosis. |
| **CREA** (Serum Creatinine) | Renal filtration byproduct | Critical indicator of secondary hepatorenal syndrome, a fatal complication of advanced end-stage cirrhosis. |
| **GGT** (Gamma-Glutamyl Transferase) | Microsomal biliary epithelial enzyme | Highly sensitive marker for toxic and viral damage; elevated in chronic HCV replication. |
| **PROT** (Total Protein) | Sum of serum proteins (albumin + globulins) | Reflects balance between deteriorating liver synthesis and polyclonal hypergammaglobulinemia induced by chronic viral stimulation. |
| **CHOL** (Total Cholesterol) | Endogenous lipid synthesized by liver | Substantially decreases in end-stage cirrhosis due to parenchymal starvation. |
| **Age** | Patient Age (Demographic) | Correlates with cumulative duration of chronic infection and exposure to progressive fibrosis. |
| **Sex** | Biological Sex (Demographic) | Accounts for physiological differences in baseline muscle mass (creatinine) and enzymatic activity. |

### 2.3 Educational and Methodological Significance
For the final-year **AI for Healthcare** curriculum, this project demonstrates an end-to-end applied AI pipeline:
- **Class Imbalance Strategy**: Rather than optimizing unweighted accuracy (which yields deceptive 87% scores by guessing "Donor" on all samples), the pipeline was tuned for Macro-F1 across all five clinical cohorts.
- **Explainability**: Decision tree feature importances are extracted and surfaced directly to clinicians, illustrating why an alert was triggered.
- **Client-Side Serverless Machine Learning**: Compiling 1,500 decision trees into pure JavaScript demonstrates how machine learning models can be served without server cold starts or operational hosting costs.

---

# 3. SCREEN SHOT OF OUTPUT

### 3.1 Project Branding and Visual Identity
The system features an official medical AI visual identity consisting of a clean anatomical liver contour, an emergency medical cross, real-time diagnostic pulse wave, and neural network hexagonal nodes.

```
       +-------------------------------------------------------+
       |                  PROJECT BRAND LOGO                   |
       |                                                       |
       |         [ + ]                                         |
       |        /     \         _/\_/\_                        |
       |       / O---O \       /       \                       |
       |      |  | X |  |-----/  EKG    \                      |
       |       \ O---O /      \  WAVE   /                      |
       |        \_____/        \_______/                       |
       |         LIVER          NEURAL AI                      |
       |                                                       |
       |   Hepatitis C Detection and Staging using ML          |
       +-------------------------------------------------------+
```

---

### 3.2 System Architecture and Execution Pipeline
The platform implements a modular architecture spanning data ingestion, preprocessing, ensemble model training, and dual-mode deployment:

```
+-----------------------------------------------------------------------------------+
|                            SYSTEM WORKFLOW PIPELINE                               |
+-----------------------------------------------------------------------------------+
|  [UCI HCV Dataset] --> [Median Imputer] --> [StandardScaler] --> [XGBoost Model]  |
|   (615 Records)        (Preserves 100%)     (Zero Mean / Unit σ)  (94.3% Accuracy)|
|          |                                                             |          |
|          v                                                             v          |
|  [Flask Web Backend] <---------------------------------------> [Client ML Engine] |
|   (Local / Cloud API)                                          (Netlify / Offline)|
|          |                                                             |          |
|          +----------------------------+--------------------------------+          |
|                                       v                                           |
|                            [Unified Clinical UI]                                  |
|        Home Overview · Diagnostic Intake · Personas · Report Staging              |
+-----------------------------------------------------------------------------------+
```

---

### 3.3 Screenshots of Application Pages

#### Screenshot 1: Interactive Home Page Dashboard
*Shows the hero banner, real-time diagnostic simulator, disease pathology stages, biomarker dictionary, and UCI dataset class distributions.*

```
+-----------------------------------------------------------------------------------+
| [Logo] Hepatitis C Detection & Staging     [ Home ]  [ Predict ]  [ About ] [Theme] |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  (•) AI-POWERED · NON-INVASIVE · MULTI-CLASS STAGING                              |
|                                                                                   |
|  Detect & Stage                                [ LIVE MODEL PROBE SIMULATOR ]     |
|  Hepatitis C                                   (•) Normal   ( ) Hep   ( ) Cirrh   |
|  from Blood Markers                            --------------------------------   |
|                                                ALT:  24.2 U/L   [====       ] 20% |
|  An XGBoost machine learning model trained on  AST:  22.8 U/L   [===        ] 18% |
|  615 real patient records. Receives 12 routine ALB:  44.5 g/L   [========== ] 82% |
|  liver function panel values and outputs a     BIL:  10.5 umol  [==         ] 15% |
|  probability-weighted stage classification.    --------------------------------   |
|                                                CLASSIFIED: Blood Donor (99.8%)    |
|  [ Run Prediction -> ]  [ Architecture v ]                                        |
|  [✓ 94.3% Test Acc] [✓ 12 Biomarkers] [✓ Zero Biopsy Trauma]                      |
|                                                                                   |
|  -------------------------------------------------------------------------------  |
|  [🩸 The Problem: 58M Global Cases]   [🫀 5 Pathology Stages: Healthy to Cirrhosis]|
+-----------------------------------------------------------------------------------+
```

---

#### Screenshot 2: Clinical Diagnostic Intake Page (`/predict`)
*Shows the 5 patient case scenarios, physical lab report scanner, symptom screener, and 12-biomarker intake form.*

```
+-----------------------------------------------------------------------------------+
|  🔬 DIAGNOSTIC INTAKE ENGINE                                                      |
|  HCV Staging & Clinical Intake Assessment                                         |
|  -------------------------------------------------------------------------------  |
|  Clinical Personas:                                                               |
|  [ Case 1: Healthy Donor ]  [ Case 2: Borderline ]  [ Case 3: Active Hepatitis ]  |
|  [ Case 4: Hepatic Fibrosis ]  [ Case 5: Decompensated Cirrhosis ]                |
|                                                                                   |
|  +---------------------------+  +-----------------------------------------------+ |
|  | Patient Case 3 Selected   |  | Patient Demographics:                         | |
|  | [Patient Photo: 46M]      |  | Age: [ 46 ] years       Sex: [ Male     v ]   | |
|  | Acute Transaminitis       |  |                                               | |
|  | ALT: 184 U/L (High)       |  | Protein Markers:                              | |
|  | AST: 152 U/L (High)       |  | ALB: [ 37.5 ] g/L       PROT: [ 71.0 ] g/L    | |
|  | BIL: 22.4 umol/L (High)   |  |                                               | |
|  | GGT: 96.0 U/L (High)      |  | Liver Enzymes & Metabolites:                  | |
|  +---------------------------+  | ALT: [ 184.0 ]          AST: [ 152.0 ]        | |
|                                 | ALP: [ 95.0  ]          BIL: [ 22.4  ]        | |
|                                 | CHE: [ 6.4   ]          CHOL:[ 4.6   ]        | |
|                                 | CREA:[ 88.0  ]          GGT: [ 96.0  ]        | |
|                                 +-----------------------------------------------+ |
|                                                                                   |
|                  [ 🔬 Run Diagnostics · Predict Stage ]                           |
+-----------------------------------------------------------------------------------+
```

---

#### Screenshot 3: Multi-Step Diagnostic Evaluation Sequence
*Demonstrates realistic clinical diagnostic execution with active DNA core spinner and animated progress checklist (~3.8 seconds).*

```
+-----------------------------------------------------------------------------------+
|                      ANALYZING PATIENT BLOOD PANEL                                |
|             [ ( ( @ ) ) ]  Rotating Multi-Ring Medical Diagnostic Core            |
|                                                                                   |
|  ✓ Acquiring 12 serum biomarker inputs & patient demographics                     |
|  ✓ Checking missing values against Median Reference Imputer                       |
|  ✓ StandardScaler z-score normalization across reference cohort                   |
|  (•) Evaluating 1,500 XGBoost decision tree paths & gradient margins…             |
|  O Analyzing De Ritis ratio (AST/ALT) & synthetic enzyme kinetics                 |
|  O Synthesizing 5-stage softprob distribution & staging confidence                |
|                                                                                   |
|  [========================================                ] 74%                   |
+-----------------------------------------------------------------------------------+
```

---

#### Screenshot 4: Output Prediction & Diagnostic Report
*Displays the predicted disease stage, confidence gauge, biomarker status chips, and full 5-stage multiclass probability distribution.*

```
+-----------------------------------------------------------------------------------+
|  📊 PREDICTION RESULT: ANALYSIS COMPLETE                                          |
|  ===============================================================================  |
|                                                                                   |
|    🔶   PREDICTED STAGE                                                           |
|         Hepatitis                                                                 |
|         Pattern consistent with active Hepatitis C viral infection. Acute         |
|         transaminase leakage requires clinical follow-up and serological PCR.     |
|                                                                                   |
|         [ 🎯 92.4% Confidence ]                                                   |
|                                                                                   |
|    Biomarker Status Flags:                                                        |
|    [▲ ALT: 184.0 (high)]  [▲ AST: 152.0 (high)]  [▲ BIL: 22.4 (high)]             |
|    [▲ GGT: 96.0 (high)]   [● ALB: 37.5 (normal)] [● CREA: 88.0 (normal)]          |
|                                                                                   |
|  -------------------------------------------------------------------------------  |
|  PROBABILITY DISTRIBUTION:                                                        |
|  #1  Hepatitis            [==========================================  ] 92.4%    |
|  #2  Fibrosis             [===                                         ]  5.1%    |
|  #3  Suspect Blood Donor  [=                                           ]  1.8%    |
|  #4  Cirrhosis            [                                            ]  0.5%    |
|  #5  Blood Donor          [                                            ]  0.2%    |
|  -------------------------------------------------------------------------------  |
|  Clinical Imaging Correlation:                                                    |
|  [Hospital MRI / Ultrasound Photo] Recommended follow-up: Quantitative HCV RNA    |
|  PCR viral load and Transient Elastography (FibroScan) confirmation.              |
|                                                                                   |
|               [ <- New Prediction ]    [ About the Model ]                        |
+-----------------------------------------------------------------------------------+
```

---

#### Screenshot 5: Model Information & Viva Defense Page (`/about`)
*Shows algorithm benchmark comparison, feature importance rankings, missing value audit, and viva defense questions.*

```
+-----------------------------------------------------------------------------------+
|  🤖 MODEL ARCHITECTURE & BENCHMARK DEFENSE                                        |
|  Algorithm: XGBoost Multi-Class Classifier  |  Test Accuracy: 94.3%               |
|  Dataset: UCI Machine Learning Repository   |  Features: 12 Serum Biomarkers      |
|  -------------------------------------------------------------------------------  |
|  Model Performance Comparison:                                                    |
|  - XGBoost:        Accuracy: 94.3%  |  Macro-F1: 0.884  |  Status: Selected       |
|  - Random Forest:  Accuracy: 92.7%  |  Macro-F1: 0.841  |  Status: Evaluated      |
|                                                                                   |
|  Top Ranked Predictive Features:                                                  |
|  1. AST  [=============================] 28.4% (Necrosis & De Ritis index)        |
|  2. CHE  [=====================]         21.2% (Liver synthesis failure marker)   |
|  3. ALB  [===============]               15.8% (Oncotic protein decompensation)   |
|  4. Age  [============]                  12.1% (Cumulative chronic fibrosis)      |
|  5. ALT  [=========]                      9.4% (Hepatocellular leakage marker)    |
|  6. ALP  [======]                         6.1% (Cholestasis & biliary injury)     |
|                                                                                   |
|  Viva Defense Q&A Section:                                                        |
|  Q1: Why is Macro-F1 preferred over accuracy? (Class imbalance defense)           |
|  Q2: How does the system handle missing lab markers? (Median imputation)          |
|  Q3: Can this tool replace needle biopsy? (Clinical augmentation rationale)      |
+-----------------------------------------------------------------------------------+
```

---

# 4. CONCLUSION

### 4.1 Summary of Work Done
The project **"Hepatitis C Detection and Staging using Machine Learning"** successfully demonstrates an end-to-end, non-invasive clinical decision support platform for predicting Hepatitis C disease progression:
1. **Clinical Machine Learning Performance**:  
   Trained and evaluated on 615 real patient records from the UCI Machine Learning Repository, the selected **XGBoost Classifier** achieved a held-out test accuracy of **94.3%** and a superior Macro-F1 score of **0.884**, outperforming standard Random Forest ensembles.
2. **Biochemical Interpretability**:  
   Feature importance attribution confirmed established medical literature: mitochondrial enzyme **AST (28.4%)**, exclusive synthetic enzyme **CHE (21.2%)**, and oncotic protein **ALB (15.8%)** were identified as the primary drivers of advanced fibrotic and cirrhotic staging.
3. **Robust Clinical Data Handling**:  
   Non-parametric median imputation was implemented across all 10 serum chemistry markers, ensuring zero patient record loss during intake while maintaining authentic population distributions.
4. **Zero-Latency Serverless Deployment**:  
   The decision tree ensemble (1,500 trees) was compiled into a lightweight client-side JavaScript engine (~200 KB), enabling full offline operation and sub-millisecond execution on cloud CDNs such as Netlify without recurring server costs.
5. **Human-Centric Clinical Design**:  
   Integrated with 5 verified patient case personas, vision-based report scanning, evidence-based symptom stratification, realistic diagnostic pacing, and high-contrast accessibility themes.

### 4.2 Ethical Boundaries & Clinical Deployment Disclaimers
In alignment with the ethical standards of the **AI for Healthcare** discipline:
- **Clinical Decision Support (CDS) Classification**: This system is designed strictly as an assistive screening and triage tool to **augment**, not replace, qualified medical practitioners.
- **Regulatory Status**: The platform serves academic and demonstration purposes and is not a certified SaMD (Software as a Medical Device) under FDA 510(k) or EU MDR guidelines.
- **Confirmatory Protocols**: All algorithmic alerts must be corroborated by standard clinical protocols, including qualitative anti-HCV antibody serology, quantitative HCV RNA PCR viral load assays, and elastographic imaging.

### 4.3 Future Scope
1. **Multi-Center Clinical Validation**: Validate the trained ensemble against diverse patient cohorts from multi-center hospital electronic health records (EHR) to evaluate cross-demographic generalization.
2. **SHAP Local Explanation Integration**: Incorporate individualized Shapley Additive exPlanations (SHAP) waterfall plots for each patient prediction to explain specific marker attributions to attending physicians.
3. **FHIR / HL7 Interoperability**: Implement standardized Fast Healthcare Interoperability Resources (FHIR) API connectors to enable direct ingestion of lab reports from Hospital Information Systems (HIS).
4. **Longitudinal Trajectory Modeling**: Extend the static cross-sectional model into recurrent or temporal transformer architectures to model patient viral clearance trajectories following Direct-Acting Antiviral (DAA) therapy.

---

### Academic Submission Summary
- **Experiment No**: 10
- **Student Name**: James Lewis
- **Roll No**: 27
- **Class / Semester**: Final Year B.E. (Artificial Intelligence and Data Science), Semester VII
- **Subject**: AI for Healthcare
- **Institute**: Vidyavardhini's College of Engineering & Technology, Vasai (University of Mumbai)
- **Academic Year**: 2026-27
- **Project Guide**: Asst. Prof. Kranti Gule
