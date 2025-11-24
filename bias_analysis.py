import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from feature_loader import load_linguistic_features, load_llm_enhanced_features

# Canonical judgment fields (intrinsic features)
JUDGMENT_FIELDS = {
    "helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    "neurips": ["rating", "confidence", "soundness", "presentation", "contribution"],
    "helpsteer3": ["score"],  # Not used in default bias run
    "antique": ["ranking"],   # Not used in default bias run
}

DATA_ROOT = "data/dataset_detection"
DATA_MAPPING = {
    "helpsteer2": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer2_train_sampled_1_grouped/gpt-4o-2024-08-06_helpsteer2_train_sampled_groups.json",
    },
    "neurips": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_neurips_train_1_grouped/gpt-4o-2024-08-06_neurips_train_groups.json",
    }
}


def load_train(dataset: str) -> List[Dict]:
    if dataset not in DATA_MAPPING or "train" not in DATA_MAPPING[dataset]:
        raise ValueError(f"No mapping for dataset {dataset}")
    path = DATA_MAPPING[dataset]["train"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_intrinsic_features(records: List[Dict], dataset: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dims = JUDGMENT_FIELDS[dataset]
    X_rows, y_vals = [], []
    for item in records:
        label = 1 if str(item.get("label", "")).lower() == "llm" else 0
        exs = item.get("examples", []) or []
        if not exs:
            X_rows.append([0.0]*len(dims))
            y_vals.append(label)
            continue
        ex = exs[0]
        row = []
        for d in dims:
            val = ex.get(d, 0.0)
            if isinstance(val, list):
                val = float(np.mean(val)) if len(val) else 0.0
            row.append(float(val))
        X_rows.append(row)
        y_vals.append(label)
    X = np.nan_to_num(np.array(X_rows, dtype=float))
    y = np.array(y_vals, dtype=int)
    return X, y, dims


def join_augmented_features(dataset: str, X_intrinsic: np.ndarray, records: List[Dict]) -> Tuple[np.ndarray, List[str], Dict[str,str]]:
    """Load linguistic + llm-enhanced features naively (row-wise) if dimensions align.
    Returns augmented matrix, feature names list (ordered), and category mapping per feature.
    """
    feature_names: List[str] = []
    category: Dict[str,str] = {}
    X_parts = [X_intrinsic]

    # Alignment for helpsteer2 
    def _canon_text(s: str) -> str:
        s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
        # Collapse internal whitespace per line but keep line structure
        s = "\n".join(" ".join(line.split()) for line in s.split("\n"))
        return s

    def _hs2_keys(data_items):
        keys = []
        for it in data_items:
            ex = (it.get("examples") or [{}])[0]
            # helpsteer2 grouped JSON stores prompt/response at top-level
            prompt = str(ex.get("prompt", "")).strip()
            response = str(ex.get("response", "")).strip()
            keys.append(prompt + "\n" + response)
        return keys

    if dataset == "helpsteer2":
        keys = _hs2_keys(records)
        # Linguistic CSV join
        try:
            import pandas as pd  # ensure available
            # Use TRAIN feature file (aligned with chosen grouped file)
            ling_path = os.path.join("data", "features", "linguistic_feature", "helpsteer2_train.csv")
            ling_df = pd.read_csv(ling_path)
            # Canonicalize to match grouped records
            ling_df["prompt"] = ling_df["prompt"].astype(str).apply(_canon_text)
            ling_df["response"] = ling_df["response"].astype(str).apply(_canon_text)
            ling_df["__key__"] = (ling_df["prompt"] + "\n" + ling_df["response"]) 
            num_cols = list(ling_df.select_dtypes(include=["number"]).columns)
            ling_g = ling_df.groupby("__key__")[num_cols].mean()
            # Diagnostic: key match rate before reindex
            try:
                match_count = sum(1 for k in keys if k in ling_g.index)
                total = len(keys)
                match_ratio = (match_count / total) if total else 0.0
                print(f"ℹ️ helpsteer2 linguistic key match: {match_count}/{total} ({match_ratio:.1%})")
            except Exception:
                pass
            ling_mat = ling_g.reindex(keys).fillna(0.0).to_numpy(dtype=float)
            X_parts.append(ling_mat)
            for n in num_cols:
                feature_names.append(n)
                category[n] = "linguistic"
        except Exception as e:
            print(f"⚠️ Failed aligned linguistic join for helpsteer2: {e}")
        # LLM-enhanced JSON join
        try:
            llm_path = os.path.join("data", "features", "llm_enhanced_features", "helpsteer2_train_Qwen3-8B.json")
            with open(llm_path, "r", encoding="utf-8") as f:
                recs = json.load(f)
            # determine union of numeric keys
            all_keys = set()
            for r in recs:
                d = r.get("llm_enhanced_feature", {}) or {}
                for k,v in d.items():
                    if isinstance(v, (int, float)):
                        all_keys.add(k)
            ordered = sorted(all_keys)
            # map key -> vector
            row_map: Dict[str, List[float]] = {}
            for r in recs:
                k = (_canon_text(r.get("prompt", "")) + "\n" + _canon_text(r.get("response", "")))
                d = r.get("llm_enhanced_feature", {}) or {}
                vec = [float(d.get(col, 0.0)) if isinstance(d.get(col, 0), (int, float)) else 0.0 for col in ordered]
                row_map[k] = vec
            # Diagnostic: key match rate
            try:
                match_count = sum(1 for k in keys if k in row_map)
                total = len(keys)
                match_ratio = (match_count / total) if total else 0.0
                print(f"ℹ️ helpsteer2 llm-enhanced key match: {match_count}/{total} ({match_ratio:.1%})")
            except Exception:
                pass
            llm_mat = np.array([row_map.get(k, [0.0]*len(ordered)) for k in keys], dtype=float)
            X_parts.append(llm_mat)
            for n in ordered:
                name = f"llm::{n}"
                feature_names.append(name)
                category[name] = "llm_enhanced"
        except Exception as e:
            print(f"⚠️ Failed aligned LLM-enhanced join for helpsteer2: {e}")
    else:
        # Fallback: naive row-wise concat when counts match
        try:
            ling_X, ling_names = load_linguistic_features(dataset, "train")
            if ling_X.shape[0] == X_intrinsic.shape[0]:
                X_parts.append(ling_X)
                for n in ling_names:
                    feature_names.append(n)
                    category[n] = "linguistic"
            else:
                print(f"⚠️ Linguistic row count mismatch for {dataset}; skipping.")
        except Exception as e:
            print(f"⚠️ Failed linguistic features for {dataset}: {e}")

        try:
            llm_X, llm_names = load_llm_enhanced_features(dataset, "train")
            if llm_X.shape[0] == X_intrinsic.shape[0]:
                X_parts.append(llm_X)
                for n in llm_names:
                    feature_names.append(n)
                    category[n] = "llm_enhanced"
            else:
                print(f"⚠️ LLM-enhanced row count mismatch for {dataset}; skipping.")
        except Exception as e:
            print(f"⚠️ Failed llm-enhanced features for {dataset}: {e}")

    # Intrinsic names first
    intrinsic_names = JUDGMENT_FIELDS[dataset]
    full_names = intrinsic_names + feature_names
    for n in intrinsic_names:
        category[n] = "judgment_intrinsic"

    X_aug = np.concatenate(X_parts, axis=1)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)
    return X_aug, full_names, category


def train_model(X: np.ndarray, y: np.ndarray, model_type: str) -> Pipeline:
    model_type = (model_type or "random_forest").lower()
    if model_type in ("logistic", "logreg", "lr"):
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
        ])
    else:
        # RandomForest does not need scaling
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
        model = Pipeline([
            ("clf", clf)
        ])
    model.fit(X, y)
    return model


def extract_top_coefficients(model: Pipeline, feature_names: List[str], category_map: Dict[str,str], top_k: int) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        # Logistic Regression path
        coefs = clf.coef_[0]
        abs_coefs = np.abs(coefs)
    elif hasattr(clf, "feature_importances_"):
        # Tree-based importance
        coefs = getattr(clf, "feature_importances_")
        abs_coefs = np.array(coefs, dtype=float)
    else:
        raise ValueError("Unsupported classifier for importance extraction")
    order = np.argsort(-abs_coefs)[:top_k]
    rows = []
    for idx in order:
        fname = feature_names[idx]
        rows.append({
            "feature": fname,
            "coef": float(coefs[idx]),
            "abs_coef": float(abs_coefs[idx]),
            "category": category_map.get(fname, "other"),
        })
    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    return df


def compute_intrinsic_bias_stats(X_intrinsic: np.ndarray, y: np.ndarray, intrinsic_names: List[str]) -> pd.DataFrame:
    rows = []
    human_mask = (y == 0)
    llm_mask = (y == 1)
    n_h = human_mask.sum(); n_l = llm_mask.sum(); n_total = len(y)
    for i, name in enumerate(intrinsic_names):
        vals = X_intrinsic[:, i]
        mean_h = float(vals[human_mask].mean())
        mean_l = float(vals[llm_mask].mean())
        var_h = float(vals[human_mask].var(ddof=1)) if n_h > 1 else 0.0
        var_l = float(vals[llm_mask].var(ddof=1)) if n_l > 1 else 0.0
        # Pooled std for Cohen's d
        if n_h > 1 and n_l > 1:
            pooled_std = np.sqrt(((n_h - 1)*var_h + (n_l - 1)*var_l) / (n_h + n_l - 2))
        else:
            pooled_std = 0.0
        cohen_d = (mean_l - mean_h) / pooled_std if pooled_std > 1e-12 else 0.0
        # Point-biserial correlation
        std_all = vals.std(ddof=1)
        r_pb = ((mean_l - mean_h) / std_all) * np.sqrt((n_h * n_l) / (n_total**2)) if std_all > 1e-12 else 0.0
        rows.append({
            "dimension": name,
            "mean_human": mean_h,
            "mean_llm": mean_l,
            "diff": mean_l - mean_h,
            "cohen_d": cohen_d,
            "point_biserial_r": r_pb,
        })
    df = pd.DataFrame(rows)
    # Sort by absolute diff descending
    df = df.reindex(df.index[np.argsort(-np.abs(df["diff"]))])
    return df


def plot_top_features(df: pd.DataFrame, out_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        colors = {
            "judgment_intrinsic": "#1f77b4",
            "linguistic": "#ff7f0e",
            "llm_enhanced": "#2ca02c",
            "other": "#7f7f7f"
        }
        df = df.sort_values("abs_coef", ascending=True)  # for horizontal bar (small at bottom)
        plt.figure(figsize=(7, 6))
        bars = plt.barh(range(len(df)), df["abs_coef"], color=[colors.get(c, "#7f7f7f") for c in df["category"]])
        plt.yticks(range(len(df)), df["feature"], fontsize=8)
        plt.xlabel("|Coefficient| (importance)")
        plt.title(title)
        # Legend (unique categories)
        seen = {}
        for b, cat in zip(bars, df["category"]):
            if cat not in seen:
                seen[cat] = b
        plt.legend(seen.values(), seen.keys(), loc="lower right", frameon=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
    except ImportError:
        print("⚠️ matplotlib not installed; skipping plots.")


def run_bias_analysis(datasets: List[str], top_k: int, output_dir: str, model_type: str = "random_forest"):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for ds in datasets:
        print(f"\n=== Bias Analysis: {ds} ===")
        records = load_train(ds)
        X_intrinsic, y, intrinsic_names = extract_intrinsic_features(records, ds)
        X_aug, feat_names, category = join_augmented_features(ds, X_intrinsic, records)
        print(f"Samples: {len(y)}, intrinsic dims: {X_intrinsic.shape[1]}, augmented dims: {X_aug.shape[1]}")
        print(f"Human: {int((y==0).sum())}, LLM: {int((y==1).sum())}")
        model = train_model(X_aug, y, model_type)
        top_df = extract_top_coefficients(model, feat_names, category, top_k)
        bias_stats_df = compute_intrinsic_bias_stats(X_intrinsic, y, intrinsic_names)
        # Save outputs
        top_path = os.path.join(output_dir, f"bias_top_features_{ds}.csv")
        stats_path = os.path.join(output_dir, f"bias_intrinsic_stats_{ds}.csv")
        top_df.to_csv(top_path, index=False)
        bias_stats_df.to_csv(stats_path, index=False)
        print(f"Saved top feature importances → {top_path}")
        print(f"Saved intrinsic bias stats → {stats_path}")
        plot_top_features(top_df, os.path.join(output_dir, f"bias_top_features_{ds}.png"), f"Top {top_k} Important Features ({ds})")
        # Record summary for README or aggregation
        summary_rows.append({
            "dataset": ds,
            "n_samples": len(y),
            "n_intrinsic_dims": X_intrinsic.shape[1],
            "n_augmented_dims": X_aug.shape[1],
            "top_features": ";".join(list(top_df.head(5).feature))
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "bias_summary.csv"), index=False)
    print(f"\n📄 Saved summary → {os.path.join(output_dir, 'bias_summary.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Bias Quantification Using Interpretability (Section 6.2 style)")
    parser.add_argument("--datasets", nargs="*", default=["helpsteer2", "neurips"], help="Datasets for bias analysis (default: helpsteer2 neurips)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features by |coef| to store")
    parser.add_argument("--output_dir", type=str, default="bias_outputs", help="Directory to write results")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "logistic"], help="Model to use for importance (default: random_forest)")
    args = parser.parse_args()
    run_bias_analysis(args.datasets, args.top_k, args.output_dir, model_type=args.model)

if __name__ == "__main__":
    main()
