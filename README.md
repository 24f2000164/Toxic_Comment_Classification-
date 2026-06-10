Toxic Comment Classification
Multi-label toxicity detection using classical NLP + ensemble blending
MLP Project — IIT Madras | Kaggle Competition | April 2026

Overview
This project builds a multi-class text classification pipeline to detect toxic comments across four severity levels. The approach combines classical NLP (TF-IDF) with 55 handcrafted linguistic features, trained with Logistic Regression and LightGBM, and blended using optimized probability weights. The final model achieves an 81.87% Macro F1-score on a heavily imbalanced 198K-row dataset.

Problem Statement
Given a comment along with metadata (upvotes, downvotes, emoticons, identity signals), predict the toxicity label:
LabelCategoryTrain CountShare0Non-toxic114,17357.7%1Identity-based toxicity15,9188.0%2General toxic62,44031.5%3Threat / severe5,4692.8%
Class imbalance is severe — Label 3 represents under 3% of training data.

Dataset
SplitRowsColumnsTrain198,00015Test102,00014
Key columns: comment, label, upvote, downvote, emoticon_1/2/3, if_1, if_2, race, religion, gender, disability, created_date
Competition: Comment Category Prediction Challenge

Methodology
1. Text Preprocessing

Unicode normalization and accent stripping
Regex-based cleaning (URLs, special characters, repeated whitespace)
Custom clean column produced and used for all vectorization

2. Exploratory Data Analysis

Label distribution and class imbalance analysis
Comment length distribution by class
Missing value profiling (race, religion, gender columns have significant NaN rates)
Upvote/downvote patterns by toxicity label
Identity column encoding (race, religion, gender → binary presence flags)

3. Feature Engineering (55 dense features)
Text-based features:

char_count, word_count, unique_words
caps_count, caps_ratio, caps_word_count
exclamation_count, question_count, repeated_chars

Toxicity signal features:

threat_count, has_threat, threat_ratio
toxic_count, has_toxic, toxic_ratio
slur_count, has_slur
identity_word_count, has_identity_word

Engagement features:

total_votes, vote_ratio, has_downvote, dv_uv_ratio
vote_silence, is_controversial
Bucketed encodings of if_1 and if_2 interaction features

Identity flags:

race_present, religion_present, gender_present
any_identity_flag, identity_col_count, disability_flag

Temporal features:

time_morning, time_afternoon, time_evening, time_night

Cross features:

threat_x_short, slur_x_threat, identity_no_toxic
caps_x_threat, caps_x_toxic

4. TF-IDF Vectorization
Two vectorizers fit on the combined train + test corpus to avoid vocabulary gaps:
VectorizerConfigFeaturesWord-levelunigrams + bigrams, max_features=20000, min_df=320,000Character-level3–5 grams (char_wb), max_features=15000, min_df=515,000
Total feature matrix: 35,055 dimensions
5. Feature Matrix Assembly
For Logistic Regression:
  X_lr = [word_tfidf | char_tfidf | dense_scaled]

For LightGBM:
  X_lgb = [word_tfidf | char_tfidf | dense_unscaled]
Dense features are scaled with StandardScaler for LR (sensitive to magnitude) but kept raw for LightGBM (tree-based, scale-invariant).
6. Class Imbalance Handling
Inverse-frequency class weights were computed, then manually adjusted based on per-class validation performance:
LabelInverse-freq weightCustom weight00.4340.43413.1102.17720.7930.79339.05116.292
Label 3 receives a 16× boost to compensate for severe under-representation.

Models
SVM (Baseline)

LinearSVC with CalibratedClassifierCV (sigmoid calibration)
L1 penalty, C=0.1, OVR multi-class
Single train/val split

MetricScoreLog Loss0.3785Accuracy0.8800Macro F10.7569
Per-class: L0=0.9575, L1=0.7670, L2=0.8415, L3=0.4616
Logistic Regression (3-Fold CV)

L1 penalty, C=0.5, liblinear solver, OVR
3-fold stratified cross-validation

FoldMacro F1AccuracyLog Loss10.80850.90650.294520.81480.90970.286830.81330.90900.2903Mean0.8122 ± 0.0033——

Threshold tuning via Nelder-Mead optimization was attempted but reduced Macro F1 (0.8122 → 0.7814), so raw thresholds were retained for LR.

LightGBM (3-Fold CV + Threshold Tuning)

n_estimators=800, learning_rate=0.05, num_leaves=31, max_depth=8
subsample=0.8, colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=0.5
Out-of-fold (OOF) predictions used for unbiased evaluation
Test predictions averaged across all folds

FoldMacro F1AccuracyBest Iter10.81350.909180020.81560.910780030.81370.9100800
After threshold tuning (Nelder-Mead):
MetricBeforeAfterΔOOF Macro F10.81430.8212+0.0070Accuracy0.90990.9141+0.0042
Tuned class weights: [1.000, 0.552, 0.862, 0.485]
Post-tuning classification report:
LabelPrecisionRecallF100.97910.94510.961810.76180.82100.790320.86560.90490.884830.65480.64140.6481

Final Blend
The final submission uses a weighted probability blend of LR and LightGBM:
pythontest_proba_blend = best_w_lr * test_proba_lr + best_w_lgb * test_proba_lgb
test_proba_scaled = test_proba_blend * best_thresholds
final_preds = np.argmax(test_proba_scaled, axis=1)
Optimal blend weights and per-class thresholds were determined via OOF probability optimization. The final model bundle (final_blend_model.joblib) packages both models, weights, and thresholds.
Final submission distribution (102,000 test samples):
LabelCountShare056,54355.4%19,0148.8%232,77332.1%33,6703.6%
Final Macro F1: 0.8187

Repository Structure
├── notebook.ipynb              # Main training + inference notebook
├── submission.csv              # Final Kaggle submission
└── README.md

Saved model artifacts (Kaggle Model Hub):
├── svm_single_model.joblib     # Baseline SVM (calibrated)
├── lr_full_pipeline.joblib     # LR models + OOF probas + CV results
├── lgb_full_bundle.joblib      # LightGBM models + OOF probas + tuned weights
└── final_blend_model.joblib    # Final blend: LR + LGB + optimal weights + thresholds

Requirements
numpy
pandas
scikit-learn
lightgbm
scipy
matplotlib
seaborn
regex
unicodedata
joblib
Install via:
bashpip install numpy pandas scikit-learn lightgbm scipy matplotlib seaborn regex joblib

How to Run
This notebook was built and executed on Kaggle with the competition dataset attached. To reproduce:

Go to the competition page and accept the rules.
Open the notebook on Kaggle and add the competition dataset as a data source.
Optionally add the saved model artifacts from the Kaggle Model Hub (f2000164/*) to skip retraining.
Run all cells in order. The final cell generates submission.csv.


Training time: The full LightGBM 3-fold CV takes approximately 130 minutes on Kaggle CPU. The LR pipeline takes ~15 minutes. Use the pre-saved .joblib bundles to skip training and go straight to inference.


Key Design Decisions
Why TF-IDF over transformers? The dataset has 198K rows and runs on CPU-only Kaggle kernels. TF-IDF with character n-grams captures sub-word toxicity patterns (deliberate misspellings, partial slurs) efficiently without GPU requirements.
Why blend LR and LightGBM? LR is fast and excels at high-dimensional sparse inputs. LightGBM adds non-linear interaction capture across features. Their prediction errors are partially uncorrelated, so blending reduces variance.
Why not threshold-tune LR? OOF experiments showed Nelder-Mead tuning degraded LR's Macro F1 from 0.8122 to 0.7814, suggesting the model was already well-calibrated. Threshold tuning helps LightGBM (+0.007) because its raw probabilities are less calibrated on imbalanced classes.
Handling Label 3 (rare threats): Custom class weight of 16× combined with post-hoc threshold reweighting. Without this, the model collapses Label 3 F1 below 0.46 (SVM baseline).

Results Summary
ModelMacro F1AccuracySVM (baseline, single fold)0.75690.8800Logistic Regression (3-fold CV)0.81220.9084LightGBM (3-fold CV, before tuning)0.81430.9099LightGBM (after threshold tuning)0.82120.9141Final blend (submission)0.8187—

Author
Sahil Kumar
B.Tech ECE — NIT Uttarakhand | B.Sc. Data Science — IIT Madras
GitHub · LinkedIn