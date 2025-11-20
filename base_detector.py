import os
import json
import argparse
import numpy as np
import pandas as pd
import hashlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from feature_loader import load_linguistic_features, load_llm_enhanced_features


JUDGMENT_FIELDS = {
    "helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    "helpsteer3": ["score"],
    "neurips": ["rating", "confidence", "soundness", "presentation", "contribution"],
    "antique": ["ranking"]
}


data_root = "data/dataset_detection"
results = []


def load_dataset(data_root, dataset, split):
    """
    Opens the exact file for each dataset/split pair based on your folder structure.
    No dynamic joining or guessing — hardcoded where necessary.
    """

    mapping = {
        "helpsteer2": {
            "train": f"{data_root}/gpt-4o-2024-08-06_helpsteer2_train_sampled_1_grouped/gpt-4o-2024-08-06_helpsteer2_train_sampled_groups.json",
            "test": f"{data_root}/gpt-4o-2024-08-06_helpsteer2_test_1_grouped/gpt-4o-2024-08-06_helpsteer2_test_groups.json"
        },
        "helpsteer3": {
            "train": f"{data_root}/gpt-4o-2024-08-06_helpsteer3_train_1_grouped/gpt-4o-2024-08-06_helpsteer3_train_groups.json",
            "test": f"{data_root}/gpt-4o-2024-08-06_helpsteer3_test_1_grouped/gpt-4o-2024-08-06_helpsteer3_test_groups.json"
        },
        "antique": {
            "train": f"{data_root}/gpt-4o-2024-08-06_antique_train_1_grouped/gpt-4o-2024-08-06_antique_train_groups.json",
            "test": f"{data_root}/gpt-4o-2024-08-06_antique_test_1_grouped/gpt-4o-2024-08-06_antique_test_groups.json"
        },
        "neurips": {
            "train": f"{data_root}/gpt-4o-2024-08-06_neurips_train_1_grouped/gpt-4o-2024-08-06_neurips_train_groups.json",
            "test": f"{data_root}/gpt-4o-2024-08-06_neurips_test_1_grouped/gpt-4o-2024-08-06_neurips_test_groups.json"
        }
    }

    if dataset not in mapping or split not in mapping[dataset]:
        raise ValueError(f"❌ No mapping found for dataset={dataset}, split={split}")

    data_path = mapping[dataset][split]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    print(f"✅ Loading {dataset} [{split}] → {data_path}")
    with open(data_path, "r",encoding="utf-8") as f:
        return json.load(f)

    
def extract_features_labels(data, dataset):
    dims = JUDGMENT_FIELDS[dataset]
    X, y = [], []

    for item in data:
        # --- Label extraction (top-level) ---
        label = 1 if str(item.get("label", "")).lower() == "llm" else 0

        # --- Features extraction ---
        examples = item.get("examples", [])
        if not examples:
            X.append([0.0] * len(dims))
            y.append(label)
            continue

        ex = examples[0]  # use first example in the group

        feat = []
        for d in dims:
            val = ex.get(d, 0.0)

            # Handle list-type fields (e.g., ranking)
            if isinstance(val, list):
                try:
                    val = np.mean(val)
                except:
                    val = 0.0
            feat.append(float(val))

        X.append(feat)
        y.append(label)

    X = np.nan_to_num(np.array(X, dtype=float))
    y = np.array(y, dtype=int)
    return X, y

            
def run_for_dataset(dataset: str, use_ling: bool, use_llm: bool, classifier: str, debug: bool = False, hs3_full_join: bool = False) -> None:
    # Special path for helpsteer3: build directly from feature files to enable augmentation reliably
    if dataset == "helpsteer3" and (use_ling or use_llm) and not hs3_full_join:
        # Load both splits' LLM-enhanced JSON to get consistent feature columns
        with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_train_Qwen3-8B.json"), "r", encoding="utf-8") as f:
            recs_tr = json.load(f)
        with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_test_Qwen3-8B.json"), "r", encoding="utf-8") as f:
            recs_te = json.load(f)

        # Build label/score map from grouped JSON via canonicalized context key
        def canon_ctx_key(ctx_list):
            parts = []
            for t in (ctx_list or []):
                role = str(t.get("role", "")).strip()
                content = str(t.get("content", "")).strip()
                content = " ".join(content.split())  # collapse whitespace
                parts.append(role + "|" + content)
            return "||".join(parts)

        def load_group_map(split: str):
            grouped = load_dataset(data_root, "helpsteer3", split)
            gmap = {}
            for item in grouped:
                ex = (item.get("examples") or [{}])[0]
                key = canon_ctx_key(ex.get("context", []))
                lbl = 1 if str(item.get("label", "")).lower() == "llm" else 0
                score = float(ex.get("score", 0.0))
                gmap[key] = (lbl, score)
            return gmap

        gmap_tr = load_group_map("train")
        gmap_te = load_group_map("test")

        def filter_and_attach(recs, gmap):
            out = []
            matched, total = 0, len(recs)
            for r in recs:
                key = canon_ctx_key(r.get("context", []))
                if key in gmap:
                    lbl, score = gmap[key]
                    r["__canon_key__"] = key
                    r["__label__"] = lbl
                    r["__score__"] = score
                    out.append(r)
                    matched += 1
            return out, matched, total

        recs_tr, mtr, ttr = filter_and_attach(recs_tr, gmap_tr)
        recs_te, mte, tte = filter_and_attach(recs_te, gmap_te)
        if debug:
            print(f"   [debug hs3-align] grouped match train: {mtr}/{ttr}, test: {mte}/{tte}")

        def extract_meta(recs):
            keys = ["||".join(sorted([str(r.get("response1", "")), str(r.get("response2", ""))])) for r in recs]
            y = [int(r.get("__label__", 0)) for r in recs]
            base = [float(r.get("__score__", 0.0)) for r in recs]
            return keys, np.array(y, dtype=int), np.array(base, dtype=float).reshape(-1, 1)

        keys_tr, y_train, base_tr = extract_meta(recs_tr)
        keys_te, y_test, base_te = extract_meta(recs_te)

        # Union of LLM numeric keys across both splits for consistent dims
        r1_keys, r2_keys = set(), set()
        for r in recs_tr + recs_te:
            r1 = r.get("llm_enhanced_feature_r1", {}) or {}
            r2 = r.get("llm_enhanced_feature_r2", {}) or {}
            r1_keys.update([k for k, v in r1.items() if isinstance(v, (int, float))])
            r2_keys.update([k for k, v in r2.items() if isinstance(v, (int, float))])
        r1_order = sorted(r1_keys)
        r2_order = sorted(r2_keys)

        def build_llm_mat(recs):
            rows = []
            for r in recs:
                r1 = r.get("llm_enhanced_feature_r1", {}) or {}
                r2 = r.get("llm_enhanced_feature_r2", {}) or {}
                row = [float(r1.get(k, 0.0)) if isinstance(r1.get(k, 0), (int, float)) else 0.0 for k in r1_order]
                row += [float(r2.get(k, 0.0)) if isinstance(r2.get(k, 0), (int, float)) else 0.0 for k in r2_order]
                rows.append(row)
            return np.array(rows, dtype=float) if use_llm and (len(r1_order) + len(r2_order) > 0) else np.zeros((len(recs), 0), dtype=float)

        llm_tr = build_llm_mat(recs_tr)
        llm_te = build_llm_mat(recs_te)

        # Linguistic CSVs with consistent numeric columns across splits
        if use_ling:
            df_tr = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer3_train.csv"))
            df_te = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer3_test.csv"))
            for df in (df_tr, df_te):
                resp_df = df[["response1", "response2"]].astype(str)
                df["__key__"] = resp_df.apply(lambda r: "||".join(sorted([r.iloc[0], r.iloc[1]])), axis=1)
            num_tr = set(df_tr.select_dtypes(include=["number"]).columns)
            num_te = set(df_te.select_dtypes(include=["number"]).columns)
            num_cols = sorted(list(num_tr.intersection(num_te)))
            ling_tr_g = df_tr.groupby("__key__")[num_cols].mean()
            ling_te_g = df_te.groupby("__key__")[num_cols].mean()
            if debug:
                m_tr = sum(1 for k in keys_tr if k in ling_tr_g.index)
                m_te = sum(1 for k in keys_te if k in ling_te_g.index)
                print(f"   [debug hs3-ling] matched train: {m_tr}/{len(keys_tr)}, matched test: {m_te}/{len(keys_te)}")
            ling_tr = ling_tr_g.reindex(keys_tr).fillna(0.0).to_numpy(dtype=float)
            ling_te = ling_te_g.reindex(keys_te).fillna(0.0).to_numpy(dtype=float)
        else:
            ling_tr = np.zeros((len(keys_tr), 0), dtype=float)
            ling_te = np.zeros((len(keys_te), 0), dtype=float)

        # Concatenate base + llm + linguistic
        X_train = np.concatenate([base_tr, llm_tr, ling_tr], axis=1)
        X_test = np.concatenate([base_te, llm_te, ling_te], axis=1)

        print(f"\n📊 HELPSTEER3 AUGMENTED DATA CHECK")
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        print(f"Human (0): {np.sum(y_train==0)}, LLM (1): {np.sum(y_train==1)}")
        print(f"Base score dim: 1, LLM dims: {llm_tr.shape[1]}, Linguistic dims: {ling_tr.shape[1]}")

        # Final cleanup
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Model selection
        if classifier == "logistic":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=200, random_state=42)

        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("classifier", clf)
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"=== Results for helpsteer3 (augmented via features, ling={use_ling}, llm={use_llm}, clf={classifier}) ===")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        results.append({
            "dataset": dataset,
            "mode": "hs3-feature-source",
            "use_linguistic": use_ling,
            "use_llm_enhanced": use_llm,
            "classifier": classifier,
            "accuracy": float(acc),
            "f1": float(f1)
        })
        return
    train_data = load_dataset(data_root, dataset, "train")
    test_data = load_dataset(data_root, dataset, "test")

    X_train, y_train = extract_features_labels(train_data, dataset)
    X_test, y_test = extract_features_labels(test_data, dataset)

    print(f"\n📊 {dataset.upper()} BASE DATA CHECK")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Human (0): {np.sum(y_train==0)}, LLM (1): {np.sum(y_train==1)}")
    print(f"Base feature dim: {X_train.shape[1]}")

    # Helper: build matching keys and join features for helpsteer2
    def _hs2_keys(data_items):
        keys = []
        for it in data_items:
            ex = (it.get("examples") or [{}])[0]
            ctx = ex.get("context", [])
            # find last assistant and its preceding user prompt
            last_assistant = None
            last_user_before = None
            for i in range(len(ctx) - 1, -1, -1):
                if ctx[i].get("role") == "assistant":
                    last_assistant = ctx[i].get("content", "")
                    # find previous user
                    for j in range(i - 1, -1, -1):
                        if ctx[j].get("role") == "user":
                            last_user_before = ctx[j].get("content", "")
                            break
                    break
            prompt = last_user_before or ""
            response = last_assistant or ""
            keys.append(prompt + "\n" + response)
        return keys

    def _antique_keys(data_items):
        keys = []
        for it in data_items:
            ex = (it.get("examples") or [{}])[0]
            q = str(ex.get("query", ""))
            docs = ex.get("docs", []) or []
            docs = sorted([str(d) for d in docs])
            key = q + "\n" + "||".join(docs)
            keys.append(key)
        return keys

    # Augment with linguistic features
    if use_ling:
        try:
            if dataset == "helpsteer2":
                # Join by prompt/response text key
                keys_train = _hs2_keys(train_data)
                keys_test = _hs2_keys(test_data)
                ling_train_df = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer2_train.csv"))
                ling_test_df = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer2_test.csv"))
                for df in (ling_train_df, ling_test_df):
                    df["__key__"] = (df["prompt"].astype(str) + "\n" + df["response"].astype(str))
                # Aggregate duplicates by mean on numeric cols
                num_cols = ling_train_df.select_dtypes(include=["number"]).columns
                ling_train_g = ling_train_df.groupby("__key__")[list(num_cols)].mean()
                ling_test_g = ling_test_df.groupby("__key__")[list(num_cols)].mean()
                if debug:
                    m_train = sum(1 for k in keys_train if k in ling_train_g.index)
                    m_test = sum(1 for k in keys_test if k in ling_test_g.index)
                    print(f"   [debug hs2-ling] matched train: {m_train}/{len(keys_train)}, matched test: {m_test}/{len(keys_test)}")
                # Reindex by our keys; missing -> 0
                ling_train_mat = ling_train_g.reindex(keys_train).fillna(0.0).to_numpy(dtype=float)
                ling_test_mat = ling_test_g.reindex(keys_test).fillna(0.0).to_numpy(dtype=float)
                X_train = np.concatenate([X_train, ling_train_mat], axis=1)
                X_test = np.concatenate([X_test, ling_test_mat], axis=1)
                print(f"➕ Added linguistic features (helpsteer2 join): +{ling_train_mat.shape[1]} dims")
            elif dataset == "antique":
                keys_train = _antique_keys(train_data)
                keys_test = _antique_keys(test_data)
                ling_train_df = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "ANTIQUE_train.csv"))
                ling_test_df = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "ANTIQUE_test.csv"))
                for df in (ling_train_df, ling_test_df):
                    resp_df = df[["response1", "response2", "response3"]].astype(str)
                    df["__resp_sorted_join__"] = resp_df.apply(lambda r: "||".join(sorted([r.iloc[0], r.iloc[1], r.iloc[2]])), axis=1)
                    df["__key__"] = df["query"].astype(str) + "\n" + df["__resp_sorted_join__"]
                num_cols_tr = ling_train_df.select_dtypes(include=["number"]).columns
                num_cols_te = ling_test_df.select_dtypes(include=["number"]).columns
                # Use intersection of numeric columns to be safe
                common_cols = [c for c in num_cols_tr if c in set(num_cols_te)]
                ling_train_g = ling_train_df.groupby("__key__")[common_cols].mean()
                ling_test_g = ling_test_df.groupby("__key__")[common_cols].mean()
                # Debug: alignment rate
                if debug:
                    m_train = sum(1 for k in keys_train if k in ling_train_g.index)
                    m_test = sum(1 for k in keys_test if k in ling_test_g.index)
                    print(f"   [debug antique-ling] matched train: {m_train}/{len(keys_train)}, matched test: {m_test}/{len(keys_test)}")
                ling_train_mat = ling_train_g.reindex(keys_train).fillna(0.0).to_numpy(dtype=float)
                ling_test_mat = ling_test_g.reindex(keys_test).fillna(0.0).to_numpy(dtype=float)
                X_train = np.concatenate([X_train, ling_train_mat], axis=1)
                X_test = np.concatenate([X_test, ling_test_mat], axis=1)
                print(f"➕ Added linguistic features (antique join): +{ling_train_mat.shape[1]} dims")
            elif dataset == "helpsteer3" and hs3_full_join:
                # Use context-key join via LLM JSON to retrieve response1/2, then map to CSV features
                def canon_ctx_key(ctx_list):
                    parts = []
                    for t in (ctx_list or []):
                        role = str(t.get("role", "")).strip()
                        content = str(t.get("content", "")).strip()
                        content = " ".join(content.split())
                        parts.append(role + "|" + content)
                    return "||".join(parts)

                # Build context keys for grouped data
                g_keys_tr = [canon_ctx_key(((it.get("examples") or [{}])[0]).get("context", [])) for it in train_data]
                g_keys_te = [canon_ctx_key(((it.get("examples") or [{}])[0]).get("context", [])) for it in test_data]

                # Load LLM JSON and map context -> sorted response key
                with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_train_Qwen3-8B.json"), "r", encoding="utf-8") as f:
                    recs_tr = json.load(f)
                with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_test_Qwen3-8B.json"), "r", encoding="utf-8") as f:
                    recs_te = json.load(f)
                def respkey(r):
                    return "||".join(sorted([str(r.get("response1", "")), str(r.get("response2", ""))]))
                def ctxkey(r):
                    return canon_ctx_key(r.get("context", []))
                resp_by_ctx_tr = {ctxkey(r): respkey(r) for r in recs_tr}
                resp_by_ctx_te = {ctxkey(r): respkey(r) for r in recs_te}

                # Load linguistic CSVs and aggregate
                df_tr = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer3_train.csv"))
                df_te = pd.read_csv(os.path.join("data", "features", "linguistic_feature", "helpsteer3_test.csv"))
                for df in (df_tr, df_te):
                    rdf = df[["response1", "response2"]].astype(str)
                    df["__key__"] = rdf.apply(lambda r: "||".join(sorted([r.iloc[0], r.iloc[1]])), axis=1)
                num_cols = sorted(list(set(df_tr.select_dtypes(include=["number"]).columns).intersection(set(df_te.select_dtypes(include=["number"]).columns))))
                ling_tr_g = df_tr.groupby("__key__")[num_cols].mean()
                ling_te_g = df_te.groupby("__key__")[num_cols].mean()

                # Build per-example keys mapped to response keys
                rk_tr = [resp_by_ctx_tr.get(k, None) for k in g_keys_tr]
                rk_te = [resp_by_ctx_te.get(k, None) for k in g_keys_te]
                if debug:
                    mt = sum(1 for k in rk_tr if k is not None)
                    me = sum(1 for k in rk_te if k is not None)
                    print(f"   [debug hs3-ling-fulljoin] have resp keys train: {mt}/{len(rk_tr)}, test: {me}/{len(rk_te)}")

                # Reindex with fallback zeros where response key missing
                def mat_from_respkeys(g, keys):
                    rows = []
                    for k in keys:
                        if k is None or k not in g.index:
                            rows.append([0.0] * len(num_cols))
                        else:
                            rows.append(list(g.loc[k, num_cols]))
                    return np.array(rows, dtype=float)

                ling_train_mat = mat_from_respkeys(ling_tr_g, rk_tr)
                ling_test_mat = mat_from_respkeys(ling_te_g, rk_te)
                X_train = np.concatenate([X_train, ling_train_mat], axis=1)
                X_test = np.concatenate([X_test, ling_test_mat], axis=1)
                print(f"➕ Added linguistic features (helpsteer3 full-join): +{ling_train_mat.shape[1]} dims")
            else:
                # Fallback to naive row-wise concat when sizes match
                ling_train, ling_cols = load_linguistic_features(dataset, "train")
                ling_test, _ = load_linguistic_features(dataset, "test")
                if ling_train.shape[0] == X_train.shape[0] and ling_test.shape[0] == X_test.shape[0]:
                    X_train = np.concatenate([X_train, ling_train], axis=1)
                    X_test = np.concatenate([X_test, ling_test], axis=1)
                    print(f"➕ Added linguistic features: +{len(ling_cols)} dims")
                else:
                    print(f"⚠️ Size mismatch for linguistic features on {dataset}. Skipping.")
        except Exception as e:
            print(f"⚠️ Failed to load/join linguistic features for {dataset}: {e}")

    # Augment with LLM-enhanced features
    if use_llm:
        try:
            if dataset == "helpsteer2":
                # Build mapping by prompt/response key
                keys_train = _hs2_keys(train_data)
                keys_test = _hs2_keys(test_data)
                llm_path_tr = os.path.join("data", "features", "llm_enhanced_features", "helpsteer2_train_Qwen3-8B.json")
                llm_path_te = os.path.join("data", "features", "llm_enhanced_features", "helpsteer2_test_Qwen3-8B.json")
                with open(llm_path_tr, "r", encoding="utf-8") as f:
                    recs_tr = json.load(f)
                with open(llm_path_te, "r", encoding="utf-8") as f:
                    recs_te = json.load(f)

                def build_llm_mat(recs, keys):
                    # Determine numeric keys once
                    all_keys = set()
                    for r in recs:
                        d = r.get("llm_enhanced_feature", {}) or {}
                        for k, v in d.items():
                            if isinstance(v, (int, float)):
                                all_keys.add(k)
                    ordered = sorted(all_keys)
                    # Map key -> vector
                    rows = {}
                    for r in recs:
                        k = (r.get("prompt", "") + "\n" + r.get("response", ""))
                        d = r.get("llm_enhanced_feature", {}) or {}
                        rows[k] = [float(d.get(k2, 0.0)) if isinstance(d.get(k2, 0), (int, float)) else 0.0 for k2 in ordered]
                    # Build matrix aligned to keys
                    mat = np.array([rows.get(k, [0.0] * len(ordered)) for k in keys], dtype=float)
                    return mat, ordered

                llm_train_mat, llm_cols = build_llm_mat(recs_tr, keys_train)
                llm_test_mat, _ = build_llm_mat(recs_te, keys_test)
                if debug:
                    # Approximate match by building available keys set
                    keys_avail_tr = set([r.get("prompt", "") + "\n" + r.get("response", "") for r in recs_tr])
                    keys_avail_te = set([r.get("prompt", "") + "\n" + r.get("response", "") for r in recs_te])
                    m_train = sum(1 for k in keys_train if k in keys_avail_tr)
                    m_test = sum(1 for k in keys_test if k in keys_avail_te)
                    print(f"   [debug hs2-llm] matched train: {m_train}/{len(keys_train)}, matched test: {m_test}/{len(keys_test)}")
                X_train = np.concatenate([X_train, llm_train_mat], axis=1)
                X_test = np.concatenate([X_test, llm_test_mat], axis=1)
                print(f"➕ Added LLM-enhanced features (helpsteer2 join): +{len(llm_cols)} dims")
            elif dataset == "antique":
                keys_train = _antique_keys(train_data)
                keys_test = _antique_keys(test_data)
                llm_path_tr = os.path.join("data", "features", "llm_enhanced_features", "ANTIQUE_train_Qwen3-8B.json")
                llm_path_te = os.path.join("data", "features", "llm_enhanced_features", "ANTIQUE_test_Qwen3-8B.json")
                with open(llm_path_tr, "r", encoding="utf-8") as f:
                    recs_tr = json.load(f)
                with open(llm_path_te, "r", encoding="utf-8") as f:
                    recs_te = json.load(f)

                def build_llm_mat_antique(recs, keys):
                    # Collect numeric keys and whether any record has a Ranking list
                    all_keys = set()
                    include_ranking = False
                    for r in recs:
                        d = r.get("llm_enhanced_feature", {}) or {}
                        for k, v in d.items():
                            if isinstance(v, (int, float)):
                                all_keys.add(k)
                            elif isinstance(v, list):
                                include_ranking = True
                    ordered = sorted(all_keys)
                    # Prepare per-record vectors
                    rows = {}
                    for r in recs:
                        k = (r.get("query", "") + "\n" + "||".join([str(x) for x in (r.get("docs") or [])]))
                        d = r.get("llm_enhanced_feature", {}) or {}
                        vec = [float(d.get(k2, 0.0)) if isinstance(d.get(k2, 0), (int, float)) else 0.0 for k2 in ordered]
                        # Append ranking list if present, else zeros (assume top-3)
                        if include_ranking:
                            rk = d.get("Ranking")
                            if isinstance(rk, list):
                                vec.extend([float(x) for x in rk])
                            else:
                                vec.extend([0.0, 0.0, 0.0])
                        rows[k] = vec
                    width = len(ordered) + (3 if include_ranking else 0)
                    mat = np.array([rows.get(k, [0.0] * width) for k in keys], dtype=float)
                    return mat, width

                llm_train_mat, w_tr = build_llm_mat_antique(recs_tr, keys_train)
                llm_test_mat, w_te = build_llm_mat_antique(recs_te, keys_test)
                X_train = np.concatenate([X_train, llm_train_mat], axis=1)
                X_test = np.concatenate([X_test, llm_test_mat], axis=1)
                print(f"➕ Added LLM-enhanced features (antique join): +{llm_train_mat.shape[1]} dims")
            elif dataset == "helpsteer3" and hs3_full_join:
                # Map grouped examples via canonicalized context to LLM-enhanced feature vectors
                def canon_ctx_key(ctx_list):
                    parts = []
                    for t in (ctx_list or []):
                        role = str(t.get("role", "")).strip()
                        content = str(t.get("content", "")).strip()
                        content = " ".join(content.split())
                        parts.append(role + "|" + content)
                    return "||".join(parts)

                g_keys_tr = [canon_ctx_key(((it.get("examples") or [{}])[0]).get("context", [])) for it in train_data]
                g_keys_te = [canon_ctx_key(((it.get("examples") or [{}])[0]).get("context", [])) for it in test_data]

                with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_train_Qwen3-8B.json"), "r", encoding="utf-8") as f:
                    recs_tr = json.load(f)
                with open(os.path.join("data", "features", "llm_enhanced_features", "helpsteer3_test_Qwen3-8B.json"), "r", encoding="utf-8") as f:
                    recs_te = json.load(f)

                # Union numeric keys across splits
                r1_keys, r2_keys = set(), set()
                for r in recs_tr + recs_te:
                    r1 = r.get("llm_enhanced_feature_r1", {}) or {}
                    r2 = r.get("llm_enhanced_feature_r2", {}) or {}
                    r1_keys.update([k for k, v in r1.items() if isinstance(v, (int, float))])
                    r2_keys.update([k for k, v in r2.items() if isinstance(v, (int, float))])
                r1_order = sorted(r1_keys)
                r2_order = sorted(r2_keys)

                def ctxkey(r):
                    return canon_ctx_key(r.get("context", []))
                def vec_from_rec(r):
                    r1 = r.get("llm_enhanced_feature_r1", {}) or {}
                    r2 = r.get("llm_enhanced_feature_r2", {}) or {}
                    row = [float(r1.get(k, 0.0)) if isinstance(r1.get(k, 0), (int, float)) else 0.0 for k in r1_order]
                    row += [float(r2.get(k, 0.0)) if isinstance(r2.get(k, 0), (int, float)) else 0.0 for k in r2_order]
                    return row

                map_tr = {ctxkey(r): vec_from_rec(r) for r in recs_tr}
                map_te = {ctxkey(r): vec_from_rec(r) for r in recs_te}
                if debug:
                    mtr = sum(1 for k in g_keys_tr if k in map_tr)
                    mte = sum(1 for k in g_keys_te if k in map_te)
                    print(f"   [debug hs3-llm-fulljoin] matched train: {mtr}/{len(g_keys_tr)}, matched test: {mte}/{len(g_keys_te)}")

                def mat_from_ctx(mapper, keys):
                    if not (use_llm and (len(r1_order)+len(r2_order)>0)):
                        return np.zeros((len(keys), 0), dtype=float)
                    cols = len(r1_order) + len(r2_order)
                    rows = []
                    for k in keys:
                        rows.append(mapper.get(k, [0.0]*cols))
                    return np.array(rows, dtype=float)

                llm_train_mat = mat_from_ctx(map_tr, g_keys_tr)
                llm_test_mat = mat_from_ctx(map_te, g_keys_te)
                X_train = np.concatenate([X_train, llm_train_mat], axis=1)
                X_test = np.concatenate([X_test, llm_test_mat], axis=1)
                print(f"➕ Added LLM-enhanced features (helpsteer3 full-join): +{llm_train_mat.shape[1]} dims")
            else:
                # Fallback to naive row-wise concat when sizes match
                llm_train, llm_cols = load_llm_enhanced_features(dataset, "train")
                llm_test, _ = load_llm_enhanced_features(dataset, "test")
                if llm_train.shape[0] == X_train.shape[0] and llm_test.shape[0] == X_test.shape[0]:
                    X_train = np.concatenate([X_train, llm_train], axis=1)
                    X_test = np.concatenate([X_test, llm_test], axis=1)
                    print(f"➕ Added LLM-enhanced features: +{len(llm_cols)} dims")
                else:
                    print(f"⚠️ Size mismatch for LLM-enhanced features on {dataset}. Skipping.")
        except Exception as e:
            print(f"⚠️ Failed to load/join LLM-enhanced features for {dataset}: {e}")

    # Final cleanup to avoid NaN/Inf from external feature files
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Total feature dim for {dataset}: {X_train.shape[1]}")

    # Model selection
    if classifier == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("classifier", clf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"=== Results for {dataset} (ling={use_ling}, llm={use_llm}, clf={classifier}) ===")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    results.append({
        "dataset": dataset,
        "use_linguistic": use_ling,
        "use_llm_enhanced": use_llm,
        "classifier": classifier,
        "accuracy": float(acc),
        "f1": float(f1)
    })


def main():
    parser = argparse.ArgumentParser(description="Base and augmented detector for LLM-judgement detectability")
    parser.add_argument("--datasets", nargs="*", default=["helpsteer2", "helpsteer3", "antique","neurips"],
                        help="Datasets to run: helpsteer2, helpsteer3, antique")
    parser.add_argument("--use_linguistic", action="store_true", help="Augment with linguistic features")
    parser.add_argument("--use_llm_enhanced", action="store_true", help="Augment with LLM-enhanced features")
    parser.add_argument("--classifier", choices=["rf", "logistic"], default="rf", help="Classifier type")
    parser.add_argument("--save_results", type=str, default="",
                        help="Optional path to save aggregated results as CSV")#Might need adjusting, havent tested
    parser.add_argument("--debug", action="store_true", help="Print feature alignment stats and summaries")
    parser.add_argument("--hs3_full_join", action="store_true", help="For helpsteer3, join features onto the full grouped dataset using context keys (recommended)")
    args = parser.parse_args()

    for ds in args.datasets:
        run_for_dataset(ds, args.use_linguistic, args.use_llm_enhanced, "logistic" if args.classifier == "logistic" else "rf", debug=args.debug, hs3_full_join=args.hs3_full_join)

    if args.save_results:
        out_path = args.save_results
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"💾 Saved results to {out_path}")


if __name__ == "__main__":
    main()

