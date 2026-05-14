"""Generate the CyberSentinel experiment notebook."""
import json, os

def cell(src, ct="code"):
    lines = (src.strip() + "\n").split("\n")
    lines = [l + "\n" for l in lines]
    if ct == "markdown":
        return {"cell_type":"markdown","metadata":{},"source":lines}
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":lines}

cells = []

# ── 0 ──
cells.append(cell("# 🛡️ CyberSentinel — Model Experiment Notebook\n> Run each model & stage individually. All tunable parameters live in `config/model_params.json`.", "markdown"))

# ── 1  Imports ──
cells.append(cell("""import json, os, sys, warnings, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))
print("✓ imports loaded")"""))

# ── 2  Config ──
cells.append(cell("## ⚙️ Configuration — edit `config/model_params.json` to tune parameters", "markdown"))

cells.append(cell("""# ╔══════════════════════════════════════════════════════════════╗
# ║  SELECT MODEL — uncomment exactly ONE line below           ║
# ╚══════════════════════════════════════════════════════════════╝

SELECTED_MODEL = "xgboost"
# SELECTED_MODEL = "randomforest"
# SELECTED_MODEL = "svm"

# ╔══════════════════════════════════════════════════════════════╗
# ║  SELECT DATASET                                            ║
# ╚══════════════════════════════════════════════════════════════╝

TRAIN_CSV = os.path.join("data", "train", "KDDTrain+.csv")
# TRAIN_CSV = os.path.join("data", "train", "KDDTrain+_20Percent.csv")  # faster

TEST_CSV  = os.path.join("data", "test", "KDDTest-21.csv")
# TEST_CSV  = os.path.join("data", "test", "KDDTest+.csv")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SELECT MODE — train, test, or both                        ║
# ╚══════════════════════════════════════════════════════════════╝

RUN_TRAIN = True
RUN_TEST  = True

print(f"Model : {SELECTED_MODEL}")
print(f"Train : {TRAIN_CSV}")
print(f"Test  : {TEST_CSV}")"""))

# ── 3  Load params ──
cells.append(cell("""# ── Load tunable parameters from JSON ──
with open(os.path.join("config", "model_params.json")) as f:
    PARAMS = json.load(f)

GENERAL      = PARAMS["general"]                         # val_size, random_state, etc.
MODEL_PARAMS = PARAMS[SELECTED_MODEL]                    # per-model stage1/stage2 hyperparams
CASCADE_TH   = PARAMS["cascade_thresholds"]              # threshold grid ranges
DROP_COLS    = PARAMS["drop_columns"]                     # columns to remove

print("\\n── General ──")
for k,v in GENERAL.items():
    if not k.startswith("_"): print(f"  {k}: {v}")

print(f"\\n── {SELECTED_MODEL} Stage 1 hyperparams ──")
for k,v in MODEL_PARAMS["stage1"].items():
    if not k.startswith("_"): print(f"  {k}: {v}")

print(f"\\n── {SELECTED_MODEL} Stage 2 hyperparams ──")
for k,v in MODEL_PARAMS["stage2"].items():
    if not k.startswith("_"): print(f"  {k}: {v}")

print("\\n── Cascade threshold grid ──")
for k,v in CASCADE_TH.items():
    if not k.startswith("_"): print(f"  {k}: {v}")"""))

# ── 4  Helpers ──
cells.append(cell("## 🔧 Helper Functions", "markdown"))

cells.append(cell("""CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

ATTACK_MAP = {
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS",
    "teardrop":"DoS","apache2":"DoS","mailbomb":"DoS","processtable":"DoS",
    "udpstorm":"DoS","worm":"DoS",
    "satan":"Probe","ipsweep":"Probe","nmap":"Probe","portsweep":"Probe",
    "mscan":"Probe","saint":"Probe",
    "guess_passwd":"R2L","ftp_write":"R2L","imap":"R2L","phf":"R2L",
    "multihop":"R2L","warezmaster":"R2L","warezclient":"R2L","spy":"R2L",
    "xlock":"R2L","xsnoop":"R2L","snmpguess":"R2L","snmpgetattack":"R2L",
    "httptunnel":"R2L","sendmail":"R2L","named":"R2L",
    "buffer_overflow":"U2R","loadmodule":"U2R","rootkit":"U2R","perl":"U2R",
    "sqlattack":"U2R","xterm":"U2R","ps":"U2R",
}

def load_and_clean(path):
    df = pd.read_csv(path).drop_duplicates()
    for c in df.select_dtypes("number").columns:
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes("object").columns:
        df[c].fillna(df[c].mode()[0], inplace=True)
    drop = [c for c in DROP_COLS if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df

def to_binary(labels):
    return np.array([0 if str(v).strip().lower()=="normal" else 1 for v in labels])

def to_category(labels):
    return pd.Series([ATTACK_MAP.get(str(v).strip().lower(), "R2L") for v in labels],
                     index=labels.index)

def make_preprocessor(X):
    cats = [c for c in CATEGORICAL_COLS if c in X.columns]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)],
                             remainder="passthrough")

def scale_pos_weight(y):
    neg = int((y==0).sum()); pos = int((y==1).sum())
    return max(1.0, neg/pos) if pos else 1.0

def build_pipeline(stage, X, y_bin=None):
    pre = make_preprocessor(X)
    p = MODEL_PARAMS[stage]
    m = SELECTED_MODEL.lower()
    if m == "randomforest":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
              min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"],
              class_weight=p["class_weight"], random_state=GENERAL["random_state"], n_jobs=-1)
        return Pipeline([("preprocessor",pre),("classifier",clf)])
    if m == "svm":
        clf = LinearSVC(C=p["C"], class_weight=p["class_weight"], dual=p["dual"],
              max_iter=p["max_iter"], random_state=GENERAL["random_state"])
        return Pipeline([("preprocessor",pre),("scaler",StandardScaler(with_mean=False)),("classifier",clf)])
    if m == "xgboost":
        kw = dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
              learning_rate=p["learning_rate"], subsample=p["subsample"],
              colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
              gamma=p["gamma"], reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
              random_state=GENERAL["random_state"], n_jobs=-1)
        if stage == "stage1":
            spw = p.get("scale_pos_weight","auto")
            kw["scale_pos_weight"] = scale_pos_weight(y_bin) if spw=="auto" else spw
            kw["eval_metric"] = "logloss"
        else:
            kw["eval_metric"] = "mlogloss"
            kw["objective"] = "multi:softprob"
        clf = XGBClassifier(**kw)
        return Pipeline([("preprocessor",pre),("classifier",clf)])
    raise ValueError(f"Unknown model: {m}")

print("✓ helpers defined")"""))

# ── 5  Load data ──
cells.append(cell("## 📂 Load & Prepare Data", "markdown"))

cells.append(cell("""df_full = load_and_clean(TRAIN_CSV)
print(f"Training data: {len(df_full)} rows, {df_full.shape[1]} cols")

y_binary_all = to_binary(df_full["label"])
train_df, val_df = train_test_split(df_full, test_size=GENERAL["val_size"],
                                     random_state=GENERAL["random_state"],
                                     stratify=y_binary_all)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print(f"Train split: {len(train_df)}  |  Val split: {len(val_df)}")"""))

# ── 6  Stage 1 ──
cells.append(cell("## 🔷 Stage 1 — Binary Classification (Normal vs Attack)", "markdown"))

cells.append(cell("""if RUN_TRAIN:
    X_tr1 = train_df.drop(columns=["label","attack_category"], errors="ignore")
    y_tr1 = to_binary(train_df["label"])
    X_v1  = val_df.drop(columns=["label","attack_category"], errors="ignore")
    y_v1  = to_binary(val_df["label"])

    pipe1 = build_pipeline("stage1", X_tr1, y_tr1)
    pipe1.fit(X_tr1, y_tr1)

    pred1 = pipe1.predict(X_v1)
    acc1  = accuracy_score(y_v1, pred1)
    cm1   = confusion_matrix(y_v1, pred1, labels=[0,1])
    tn,fp,fn,tp = cm1.ravel()

    print(f"\\n{'='*55}")
    print(f"  STAGE 1 RESULTS — {SELECTED_MODEL.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy     : {acc1*100:.2f}%")
    print(f"  F1           : {f1_score(y_v1, pred1):.4f}")
    print(f"  Attack Recall: {tp/(tp+fn)*100:.2f}%")
    print(f"  FPR (Normal) : {fp/(fp+tn)*100:.4f}%")
    print(f"  Confusion    : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_v1, pred1, target_names=["Normal","Attack"]))

    # Save stage 1
    out_dir = os.path.join("model","cascade",SELECTED_MODEL.lower())
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe1, os.path.join(out_dir,"stage1_model.pkl"))
    print(f"✓ Stage 1 model saved → {out_dir}/stage1_model.pkl")
else:
    print("⏭ Skipping Stage 1 training (RUN_TRAIN=False)")"""))

# ── 7  Stage 2 ──
cells.append(cell("## 🔶 Stage 2 — Attack Category Classification (DoS / Probe / R2L / U2R)", "markdown"))

cells.append(cell("""if RUN_TRAIN:
    y_tr_bin = to_binary(train_df["label"])
    y_v_bin  = to_binary(val_df["label"])

    atk_tr = train_df[y_tr_bin==1].copy()
    atk_v  = val_df[y_v_bin==1].copy()

    atk_tr["attack_category"] = to_category(atk_tr["label"])
    atk_v["attack_category"]  = to_category(atk_v["label"])

    X_tr2 = atk_tr.drop(columns=["label","attack_category"], errors="ignore")
    X_v2  = atk_v.drop(columns=["label","attack_category"], errors="ignore")

    le2 = LabelEncoder()
    y_tr2 = le2.fit_transform(atk_tr["attack_category"])
    y_v2  = le2.transform(atk_v["attack_category"])

    pipe2 = build_pipeline("stage2", X_tr2)

    # Fit with sample weights for class balancing
    classes, counts = np.unique(y_tr2, return_counts=True)
    cw = {c: len(y_tr2)/(len(classes)*n) for c,n in zip(classes,counts)}
    sw = np.array([cw[v] for v in y_tr2])
    pipe2.fit(X_tr2, y_tr2, classifier__sample_weight=sw)

    pred2 = pipe2.predict(X_v2)
    acc2  = accuracy_score(y_v2, pred2)
    f1_m  = f1_score(y_v2, pred2, average="macro", zero_division=0)

    print(f"\\n{'='*55}")
    print(f"  STAGE 2 RESULTS — {SELECTED_MODEL.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc2*100:.2f}%")
    print(f"  Macro F1 : {f1_m:.4f}")
    print(f"  Classes  : {list(le2.classes_)}")
    print(classification_report(y_v2, pred2, target_names=le2.classes_, zero_division=0))

    # Save stage 2
    joblib.dump(pipe2, os.path.join(out_dir,"stage2_model.pkl"))
    joblib.dump(le2,   os.path.join(out_dir,"stage2_label_encoder.pkl"))
    print(f"✓ Stage 2 model + encoder saved → {out_dir}")
else:
    print("⏭ Skipping Stage 2 training (RUN_TRAIN=False)")"""))

# ── 8  Calibration ──
cells.append(cell("## 🎯 Calibration — Probability Calibration for Cascade", "markdown"))

cells.append(cell("""if RUN_TRAIN:
    method = GENERAL["calibration_method"]        # ◄── TUNABLE: "sigmoid" or "isotonic"

    def calibrate(base, X_cal, y_cal, method):
        try:
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=None)
        except Exception:
            cal = CalibratedClassifierCV(base, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal

    # Calibrate Stage 1
    cal1 = calibrate(pipe1, X_v1, y_v1, method)
    joblib.dump(cal1, os.path.join(out_dir,"stage1_calibrated.pkl"))

    # Calibrate Stage 2
    cal2 = calibrate(pipe2, X_v2, y_v2, method)
    joblib.dump(cal2, os.path.join(out_dir,"stage2_calibrated.pkl"))

    print(f"✓ Both stages calibrated ({method}) and saved")
else:
    print("⏭ Skipping calibration (RUN_TRAIN=False)")"""))

# ── 9  Threshold tuning ──
cells.append(cell("## 📊 Threshold Tuning — Grid Search for Cascade Decision Boundaries", "markdown"))

cells.append(cell("""if RUN_TRAIN:
    # ◄── ALL TUNABLE via config/model_params.json ──►
    t1_low_grid    = np.arange(CASCADE_TH["stage1_low"]["start"],
                               CASCADE_TH["stage1_low"]["stop"],
                               CASCADE_TH["stage1_low"]["step"])
    t1_strong_grid = np.arange(CASCADE_TH["stage1_strong"]["start"],
                               CASCADE_TH["stage1_strong"]["stop"],
                               CASCADE_TH["stage1_strong"]["step"])
    t2_conf_grid   = np.arange(CASCADE_TH["stage2_conf"]["start"],
                               CASCADE_TH["stage2_conf"]["stop"],
                               CASCADE_TH["stage2_conf"]["step"])
    t2_margin_grid = np.arange(CASCADE_TH["stage2_margin"]["start"],
                               CASCADE_TH["stage2_margin"]["stop"],
                               CASCADE_TH["stage2_margin"]["step"])

    min_recall = GENERAL["min_attack_recall_for_threshold_search"]  # ◄── TUNABLE

    # Re-compute probabilities on val set with calibrated models
    p_atk  = cal1.predict_proba(X_v1)[:,1]
    p_cat  = cal2.predict_proba(X_v2)

    # Extend stage2 probs to full val set (normal samples get zeros)
    p_cat_full = np.zeros((len(X_v1), p_cat.shape[1]))
    atk_mask = y_v1 == 1

    # We need stage2 predictions for ALL samples (even normal) for threshold search
    p_cat_all = cal2.predict_proba(X_v1)
    top1   = np.max(p_cat_all, axis=1)
    top2   = np.partition(p_cat_all, -2, axis=1)[:,-2]
    margin = top1 - top2

    rows = []
    for tl in t1_low_grid:
        for ts in t1_strong_grid:
            if ts <= tl: continue
            for tc in t2_conf_grid:
                for tm in t2_margin_grid:
                    fb = np.zeros(len(X_v1), dtype=int)
                    fb[p_atk >= ts] = 1
                    weak = (p_atk >= tl) & (p_atk < ts)
                    fb[weak & (top1 >= tc) & (margin >= tm)] = 1

                    tn,fp,fn,tp = confusion_matrix(y_v1, fb, labels=[0,1]).ravel()
                    ar = tp/(tp+fn) if (tp+fn) else 0
                    fpr = fp/(fp+tn) if (fp+tn) else 0
                    bf1 = f1_score(y_v1, fb, zero_division=0)
                    rows.append(dict(stage1_low=round(float(tl),4), stage1_strong=round(float(ts),4),
                                     stage2_conf=round(float(tc),4), stage2_margin=round(float(tm),4),
                                     attack_recall=ar, fpr_normal=fpr, binary_f1=bf1,
                                     accuracy=accuracy_score(y_v1,fb)))

    # Select best
    valid = [r for r in rows if r["attack_recall"] >= min_recall]
    if valid:
        best = min(valid, key=lambda r: (r["fpr_normal"], -r["attack_recall"]))
    else:
        best = max(rows, key=lambda r: (r["attack_recall"], -r["fpr_normal"]))

    cascade_cfg = {
        "model": SELECTED_MODEL,
        "thresholds": {k: best[k] for k in ["stage1_low","stage1_strong","stage2_conf","stage2_margin"]},
        "validation_metrics": {k: best[k] for k in ["attack_recall","fpr_normal","binary_f1","accuracy"]}
    }

    with open(os.path.join(out_dir,"cascade_config.json"),"w") as f:
        json.dump(cascade_cfg, f, indent=2)

    print(f"\\n{'='*55}")
    print(f"  BEST THRESHOLDS — {SELECTED_MODEL.upper()}")
    print(f"{'='*55}")
    for k,v in cascade_cfg["thresholds"].items():
        print(f"  {k:20s}: {v}")
    print()
    for k,v in cascade_cfg["validation_metrics"].items():
        print(f"  {k:20s}: {v:.6f}")
    print(f"\\n✓ cascade_config.json saved → {out_dir}")
    print(f"  (searched {len(rows)} threshold combos)")
else:
    print("⏭ Skipping threshold tuning (RUN_TRAIN=False)")"""))

# ── 10  Test inference ──
cells.append(cell("## 🧪 Test Inference — Full Cascade on Test Dataset", "markdown"))

cells.append(cell("""if RUN_TEST:
    # Load models (works whether we just trained or are loading saved ones)
    out_dir = os.path.join("model","cascade", SELECTED_MODEL.lower())
    s1 = joblib.load(os.path.join(out_dir,"stage1_calibrated.pkl"))
    s2 = joblib.load(os.path.join(out_dir,"stage2_calibrated.pkl"))
    le = joblib.load(os.path.join(out_dir,"stage2_label_encoder.pkl"))
    with open(os.path.join(out_dir,"cascade_config.json")) as f:
        cfg = json.load(f)
    th = cfg["thresholds"]

    df_test = load_and_clean(TEST_CSV)
    X_test  = df_test.drop(columns=["label","attack_category"], errors="ignore")
    print(f"Test data: {len(df_test)} rows from {TEST_CSV}")

    p_atk = s1.predict_proba(X_test)[:,1]
    p_cat = s2.predict_proba(X_test)
    top1  = np.max(p_cat, axis=1)
    top2  = np.partition(p_cat, -2, axis=1)[:,-2]
    marg  = top1 - top2
    pred_cat = le.inverse_transform(np.argmax(p_cat, axis=1))

    tl, ts = th["stage1_low"], th["stage1_strong"]
    tc, tm = th["stage2_conf"], th["stage2_margin"]

    final, path = [], []
    for i in range(len(X_test)):
        if p_atk[i] < tl:
            final.append("normal"); path.append("stage1_normal")
        elif p_atk[i] >= ts:
            final.append(pred_cat[i]); path.append("stage2_strong")
        elif top1[i] < tc or marg[i] < tm:
            final.append("normal"); path.append("stage2_reverted")
        else:
            final.append(pred_cat[i]); path.append("stage2_confident")

    result = df_test.copy()
    result["stage1_p_attack"]      = p_atk
    result["stage2_pred_category"] = pred_cat
    result["stage2_top1_conf"]     = top1
    result["stage2_margin"]        = marg
    result["final_prediction"]     = final
    result["decision_path"]        = path

    if "label" in df_test.columns:
        yt = to_binary(df_test["label"])
        yp = np.array([0 if v=="normal" else 1 for v in final])
        tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        acc = accuracy_score(yt, yp)
        ar  = tp/(tp+fn) if (tp+fn) else 0
        fpr = fp/(fp+tn) if (fp+tn) else 0

        print(f"\\n{'='*55}")
        print(f"  CASCADE TEST RESULTS — {SELECTED_MODEL.upper()}")
        print(f"{'='*55}")
        print(f"  Accuracy     : {acc*100:.2f}%")
        print(f"  Attack Recall: {ar*100:.2f}%")
        print(f"  FPR Normal   : {fpr*100:.4f}%")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print(f"\\n  Decision path breakdown:")
    for p_name, cnt in pd.Series(path).value_counts().items():
        print(f"    {p_name:25s}: {cnt}")

    out_csv = TEST_CSV.replace(".csv", f"_cascade_{SELECTED_MODEL}.csv")
    result.to_csv(out_csv, index=False)
    print(f"\\n✓ Predictions saved → {out_csv}")
else:
    print("⏭ Skipping test inference (RUN_TEST=False)")"""))

# ── 11  Quick compare ──
cells.append(cell("## 📈 Quick Single-Sample Inspection", "markdown"))

cells.append(cell("""if RUN_TEST:
    sample = result.sample(5, random_state=42)[["label","final_prediction","decision_path",
              "stage1_p_attack","stage2_pred_category","stage2_top1_conf","stage2_margin"]]
    display(sample)
else:
    print("Run test inference first")"""))

# ── Build notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out_path = os.path.join("CyberSentinel", "cybersentinel_experiments.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"OK - Notebook written to {out_path}")
