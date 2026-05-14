"""Patch cybersentinel_experiments.ipynb to fix critical issues."""
import json, copy

NB_PATH = "cybersentinel_experiments.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Cell index map (0-based) ──
# 0: markdown title
# 1: code imports
# 2: markdown config
# 3: code model/dataset selection
# 4: code load params
# 5: markdown helpers
# 6: code helpers
# 7: markdown load data
# 8: code load & prepare data
# 9: markdown stage 1
# 10: code stage 1
# 11: markdown stage 2
# 12: code stage 2
# 13: markdown calibration
# 14: code calibration
# 15: markdown threshold
# 16: code threshold tuning
# 17: markdown test
# 18: code test inference
# 19: markdown inspection
# 20: code inspection

# ═══════════════════════════════════════════════════════════════
# FIX 3: Move out_dir to config cell (cell 3) so it's always available
# ═══════════════════════════════════════════════════════════════
cells[3]["source"].insert(-1, "# Output directory for saved models\n")
cells[3]["source"].insert(-1, 'out_dir = os.path.join("model","cascade",SELECTED_MODEL.lower())\n')
cells[3]["source"].insert(-1, "os.makedirs(out_dir, exist_ok=True)\n")

# ═══════════════════════════════════════════════════════════════
# FIX 1: 3-way split to prevent data leakage (cell 8)
# ═══════════════════════════════════════════════════════════════
cells[8]["source"] = [
    'df_full = load_and_clean(TRAIN_CSV)\n',
    'print(f"Training data: {len(df_full)} rows, {df_full.shape[1]} cols")\n',
    '\n',
    'y_binary_all = to_binary(df_full["label"])\n',
    '\n',
    '# 3-way split: train / calibration / validation\n',
    '# Calibration set is used ONLY for probability calibration\n',
    '# Validation set is used ONLY for metrics & threshold tuning (no leakage)\n',
    'train_df, temp_df = train_test_split(df_full, test_size=GENERAL["val_size"],\n',
    '                                     random_state=GENERAL["random_state"],\n',
    '                                     stratify=y_binary_all)\n',
    'y_temp_bin = to_binary(temp_df["label"])\n',
    'cal_df, val_df = train_test_split(temp_df, test_size=0.5,\n',
    '                                   random_state=GENERAL["random_state"],\n',
    '                                   stratify=y_temp_bin)\n',
    '\n',
    'train_df = train_df.reset_index(drop=True)\n',
    'cal_df   = cal_df.reset_index(drop=True)\n',
    'val_df   = val_df.reset_index(drop=True)\n',
    'print(f"Train: {len(train_df)}  |  Cal: {len(cal_df)}  |  Val: {len(val_df)}")\n',
    '\n',
]

# ═══════════════════════════════════════════════════════════════
# FIX 3 cont: Remove out_dir from Stage 1 cell (cell 10)
# ═══════════════════════════════════════════════════════════════
cells[10]["source"] = [
    'if RUN_TRAIN:\n',
    '    X_tr1 = train_df.drop(columns=["label","attack_category"], errors="ignore")\n',
    '    y_tr1 = to_binary(train_df["label"])\n',
    '    X_v1  = val_df.drop(columns=["label","attack_category"], errors="ignore")\n',
    '    y_v1  = to_binary(val_df["label"])\n',
    '\n',
    '    pipe1 = build_pipeline("stage1", X_tr1, y_tr1)\n',
    '    pipe1.fit(X_tr1, y_tr1)\n',
    '\n',
    '    pred1 = pipe1.predict(X_v1)\n',
    '    acc1  = accuracy_score(y_v1, pred1)\n',
    '    cm1   = confusion_matrix(y_v1, pred1, labels=[0,1])\n',
    '    tn,fp,fn,tp = cm1.ravel()\n',
    '\n',
    '    print(f"\\n{\'=\'*55}")\n',
    '    print(f"  STAGE 1 RESULTS \\u2014 {SELECTED_MODEL.upper()}")\n',
    '    print(f"{\'=\'*55}")\n',
    '    print(f"  Accuracy     : {acc1*100:.2f}%")\n',
    '    print(f"  F1           : {f1_score(y_v1, pred1):.4f}")\n',
    '    ar1 = tp/(tp+fn) if (tp+fn) else 0\n',
    '    fpr1 = fp/(fp+tn) if (fp+tn) else 0\n',
    '    print(f"  Attack Recall: {ar1*100:.2f}%")\n',
    '    print(f"  FPR (Normal) : {fpr1*100:.4f}%")\n',
    '    print(f"  Confusion    : TN={tn} FP={fp} FN={fn} TP={tp}")\n',
    '    print(classification_report(y_v1, pred1, target_names=["Normal","Attack"]))\n',
    '\n',
    '    # Save stage 1\n',
    '    joblib.dump(pipe1, os.path.join(out_dir,"stage1_model.pkl"))\n',
    '    print(f"\\u2713 Stage 1 model saved \\u2192 {out_dir}/stage1_model.pkl")\n',
    'else:\n',
    '    print("\\u23ed Skipping Stage 1 training (RUN_TRAIN=False)")\n',
    '\n',
]

# ═══════════════════════════════════════════════════════════════
# FIX 1 cont: Calibration uses cal_df (cell 14)
# ═══════════════════════════════════════════════════════════════
cells[14]["source"] = [
    'if RUN_TRAIN:\n',
    '    method = GENERAL["calibration_method"]\n',
    '\n',
    '    def calibrate(base, X_cal, y_cal, method):\n',
    '        try:\n',
    '            from sklearn.frozen import FrozenEstimator\n',
    '            cal = CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=None)\n',
    '        except Exception:\n',
    '            cal = CalibratedClassifierCV(base, method=method, cv="prefit")\n',
    '        cal.fit(X_cal, y_cal)\n',
    '        return cal\n',
    '\n',
    '    # ── Calibrate Stage 1 using dedicated calibration split ──\n',
    '    X_cal1 = cal_df.drop(columns=["label","attack_category"], errors="ignore")\n',
    '    y_cal1 = to_binary(cal_df["label"])\n',
    '    cal1 = calibrate(pipe1, X_cal1, y_cal1, method)\n',
    '    joblib.dump(cal1, os.path.join(out_dir,"stage1_calibrated.pkl"))\n',
    '\n',
    '    # ── Calibrate Stage 2 — attack-only samples from cal split ──\n',
    '    atk_cal = cal_df[to_binary(cal_df["label"]) == 1].copy()\n',
    '    atk_cal["attack_category"] = to_category(atk_cal["label"])\n',
    '    X_cal2 = atk_cal.drop(columns=["label","attack_category"], errors="ignore")\n',
    '    y_cal2 = le2.transform(atk_cal["attack_category"])\n',
    '    cal2 = calibrate(pipe2, X_cal2, y_cal2, method)\n',
    '    joblib.dump(cal2, os.path.join(out_dir,"stage2_calibrated.pkl"))\n',
    '\n',
    '    print(f"\\u2713 Both stages calibrated ({method}) using dedicated cal split")\n',
    '    print(f"  Cal set: {len(cal_df)} total, {len(atk_cal)} attacks")\n',
    'else:\n',
    '    print("\\u23ed Skipping calibration (RUN_TRAIN=False)")\n',
    '\n',
]

# ═══════════════════════════════════════════════════════════════
# FIX 2: Threshold tuning — stage 2 only on candidates (cell 16)
# ═══════════════════════════════════════════════════════════════
cells[16]["source"] = [
    'if RUN_TRAIN:\n',
    '    t1_low_grid    = np.arange(CASCADE_TH["stage1_low"]["start"],\n',
    '                               CASCADE_TH["stage1_low"]["stop"],\n',
    '                               CASCADE_TH["stage1_low"]["step"])\n',
    '    t1_strong_grid = np.arange(CASCADE_TH["stage1_strong"]["start"],\n',
    '                               CASCADE_TH["stage1_strong"]["stop"],\n',
    '                               CASCADE_TH["stage1_strong"]["step"])\n',
    '    t2_conf_grid   = np.arange(CASCADE_TH["stage2_conf"]["start"],\n',
    '                               CASCADE_TH["stage2_conf"]["stop"],\n',
    '                               CASCADE_TH["stage2_conf"]["step"])\n',
    '    t2_margin_grid = np.arange(CASCADE_TH["stage2_margin"]["start"],\n',
    '                               CASCADE_TH["stage2_margin"]["stop"],\n',
    '                               CASCADE_TH["stage2_margin"]["step"])\n',
    '\n',
    '    min_recall = GENERAL["min_attack_recall_for_threshold_search"]\n',
    '\n',
    '    # Stage 1 probabilities on the held-out val set (clean — not used for calibration)\n',
    '    p_atk = cal1.predict_proba(X_v1)[:,1]\n',
    '\n',
    '    # ── FIX: Only run Stage 2 on samples that could ever reach it ──\n',
    '    # Any sample with p_atk < min(t1_low_grid) is always classified normal,\n',
    '    # so stage 2 predictions are never used for those.\n',
    '    min_tl = float(t1_low_grid.min())\n',
    '    s2_mask = p_atk >= min_tl\n',
    '    top1   = np.zeros(len(X_v1))\n',
    '    margin = np.zeros(len(X_v1))\n',
    '\n',
    '    if s2_mask.any():\n',
    '        p_cat_s2 = cal2.predict_proba(X_v1[s2_mask])\n',
    '        top1_s2  = np.max(p_cat_s2, axis=1)\n',
    '        top2_s2  = np.partition(p_cat_s2, -2, axis=1)[:,-2]\n',
    '        top1[s2_mask]   = top1_s2\n',
    '        margin[s2_mask] = top1_s2 - top2_s2\n',
    '    print(f"Stage 2 applied to {s2_mask.sum()}/{len(X_v1)} val samples")\n',
    '\n',
    '    rows = []\n',
    '    for tl in t1_low_grid:\n',
    '        for ts in t1_strong_grid:\n',
    '            if ts <= tl: continue\n',
    '            for tc in t2_conf_grid:\n',
    '                for tm in t2_margin_grid:\n',
    '                    fb = np.zeros(len(X_v1), dtype=int)\n',
    '                    fb[p_atk >= ts] = 1\n',
    '                    weak = (p_atk >= tl) & (p_atk < ts)\n',
    '                    fb[weak & (top1 >= tc) & (margin >= tm)] = 1\n',
    '\n',
    '                    tn,fp,fn,tp = confusion_matrix(y_v1, fb, labels=[0,1]).ravel()\n',
    '                    ar = tp/(tp+fn) if (tp+fn) else 0\n',
    '                    fpr = fp/(fp+tn) if (fp+tn) else 0\n',
    '                    bf1 = f1_score(y_v1, fb, zero_division=0)\n',
    '                    rows.append(dict(stage1_low=round(float(tl),4), stage1_strong=round(float(ts),4),\n',
    '                                     stage2_conf=round(float(tc),4), stage2_margin=round(float(tm),4),\n',
    '                                     attack_recall=ar, fpr_normal=fpr, binary_f1=bf1,\n',
    '                                     accuracy=accuracy_score(y_v1,fb)))\n',
    '\n',
    '    # Select best\n',
    '    valid = [r for r in rows if r["attack_recall"] >= min_recall]\n',
    '    if valid:\n',
    '        best = min(valid, key=lambda r: (r["fpr_normal"], -r["attack_recall"]))\n',
    '    else:\n',
    '        best = max(rows, key=lambda r: (r["attack_recall"], -r["fpr_normal"]))\n',
    '\n',
    '    cascade_cfg = {\n',
    '        "model": SELECTED_MODEL,\n',
    '        "thresholds": {k: best[k] for k in ["stage1_low","stage1_strong","stage2_conf","stage2_margin"]},\n',
    '        "validation_metrics": {k: best[k] for k in ["attack_recall","fpr_normal","binary_f1","accuracy"]}\n',
    '    }\n',
    '\n',
    '    with open(os.path.join(out_dir,"cascade_config.json"),"w") as f:\n',
    '        json.dump(cascade_cfg, f, indent=2)\n',
    '\n',
    '    print(f"\\n{\'=\'*55}")\n',
    '    print(f"  BEST THRESHOLDS \\u2014 {SELECTED_MODEL.upper()}")\n',
    '    print(f"{\'=\'*55}")\n',
    '    for k,v in cascade_cfg["thresholds"].items():\n',
    '        print(f"  {k:20s}: {v}")\n',
    '    print()\n',
    '    for k,v in cascade_cfg["validation_metrics"].items():\n',
    '        print(f"  {k:20s}: {v:.6f}")\n',
    '    print(f"\\n\\u2713 cascade_config.json saved \\u2192 {out_dir}")\n',
    '    print(f"  (searched {len(rows)} threshold combos)")\n',
    'else:\n',
    '    print("\\u23ed Skipping threshold tuning (RUN_TRAIN=False)")\n',
    '\n',
]

# ═══════════════════════════════════════════════════════════════
# FIX 2 cont: Test inference — stage 2 only on candidates (cell 18)
# ═══════════════════════════════════════════════════════════════
cells[18]["source"] = [
    'if RUN_TEST:\n',
    '    # Load models\n',
    '    out_dir = os.path.join("model","cascade", SELECTED_MODEL.lower())\n',
    '    s1 = joblib.load(os.path.join(out_dir,"stage1_calibrated.pkl"))\n',
    '    s2 = joblib.load(os.path.join(out_dir,"stage2_calibrated.pkl"))\n',
    '    le = joblib.load(os.path.join(out_dir,"stage2_label_encoder.pkl"))\n',
    '    with open(os.path.join(out_dir,"cascade_config.json")) as f:\n',
    '        cfg = json.load(f)\n',
    '    th = cfg["thresholds"]\n',
    '\n',
    '    df_test = load_and_clean(TEST_CSV)\n',
    '    X_test  = df_test.drop(columns=["label","attack_category"], errors="ignore")\n',
    '    print(f"Test data: {len(df_test)} rows from {TEST_CSV}")\n',
    '\n',
    '    # Stage 1: binary attack probability for all samples\n',
    '    p_atk = s1.predict_proba(X_test)[:,1]\n',
    '    tl, ts = th["stage1_low"], th["stage1_strong"]\n',
    '    tc, tm = th["stage2_conf"], th["stage2_margin"]\n',
    '\n',
    '    # ── FIX: Only run Stage 2 on samples that need it (p_atk >= tl) ──\n',
    '    s2_mask  = p_atk >= tl\n',
    '    pred_cat = np.full(len(X_test), "", dtype=object)\n',
    '    top1     = np.zeros(len(X_test))\n',
    '    marg     = np.zeros(len(X_test))\n',
    '\n',
    '    if s2_mask.any():\n',
    '        X_s2  = X_test[s2_mask]\n',
    '        p_cat = s2.predict_proba(X_s2)\n',
    '        top1[s2_mask] = np.max(p_cat, axis=1)\n',
    '        top2_vals     = np.partition(p_cat, -2, axis=1)[:,-2]\n',
    '        marg[s2_mask] = top1[s2_mask] - top2_vals\n',
    '        pred_cat[s2_mask] = le.inverse_transform(np.argmax(p_cat, axis=1))\n',
    '    print(f"  Stage 2 applied to {s2_mask.sum()}/{len(X_test)} samples")\n',
    '\n',
    '    final, path = [], []\n',
    '    for i in range(len(X_test)):\n',
    '        if p_atk[i] < tl:\n',
    '            final.append("normal"); path.append("stage1_normal")\n',
    '        elif p_atk[i] >= ts:\n',
    '            final.append(pred_cat[i]); path.append("stage2_strong")\n',
    '        elif top1[i] < tc or marg[i] < tm:\n',
    '            final.append("normal"); path.append("stage2_reverted")\n',
    '        else:\n',
    '            final.append(pred_cat[i]); path.append("stage2_confident")\n',
    '\n',
    '    result = df_test.copy()\n',
    '    result["stage1_p_attack"]      = p_atk\n',
    '    result["stage2_pred_category"] = pred_cat\n',
    '    result["stage2_top1_conf"]     = top1\n',
    '    result["stage2_margin"]        = marg\n',
    '    result["final_prediction"]     = final\n',
    '    result["decision_path"]        = path\n',
    '\n',
    '    if "label" in df_test.columns:\n',
    '        yt = to_binary(df_test["label"])\n',
    '        yp = np.array([0 if v=="normal" else 1 for v in final])\n',
    '        tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()\n',
    '        acc = accuracy_score(yt, yp)\n',
    '        ar  = tp/(tp+fn) if (tp+fn) else 0\n',
    '        fpr = fp/(fp+tn) if (fp+tn) else 0\n',
    '\n',
    '        print(f"\\n{\'=\'*55}")\n',
    '        print(f"  CASCADE TEST RESULTS \\u2014 {SELECTED_MODEL.upper()}")\n',
    '        print(f"{\'=\'*55}")\n',
    '        print(f"  Accuracy     : {acc*100:.2f}%")\n',
    '        print(f"  Attack Recall: {ar*100:.2f}%")\n',
    '        print(f"  FPR Normal   : {fpr*100:.4f}%")\n',
    '        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")\n',
    '\n',
    '    print(f"\\n  Decision path breakdown:")\n',
    '    for p_name, cnt in pd.Series(path).value_counts().items():\n',
    '        print(f"    {p_name:25s}: {cnt}")\n',
    '\n',
    '    out_csv = TEST_CSV.replace(".csv", f"_cascade_{SELECTED_MODEL}.csv")\n',
    '    result.to_csv(out_csv, index=False)\n',
    '    print(f"\\n\\u2713 Predictions saved \\u2192 {out_csv}")\n',
    'else:\n',
    '    print("\\u23ed Skipping test inference (RUN_TEST=False)")\n',
    '\n',
]

# ── Write patched notebook ──
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook patched successfully")
print("  Changes:")
print("  1. 3-way split (train/cal/val) — fixes data leakage")
print("  2. Stage 2 only runs on attack candidates — not normal samples")
print("  3. out_dir moved to config cell — always available")
print("  4. Zero-division guard added to Stage 1 metrics")
