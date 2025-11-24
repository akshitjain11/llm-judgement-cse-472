import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

"""Detectability Analysis (Step 4)

Implements:
  1. Group Size Analysis (k = 1,2,4,8,16)
  2. Rating Scale Analysis (helpsteer2 & helpsteer3)
  3. Judgment Dimension Number Analysis (helpsteer2 & neurips)
  4. Visualization of accuracy/F1 vs parameter.

Constraints: Keeps pipeline within previously defined steps (load data → instance model → group aggregation).
"""

DATA_ROOT = "data/dataset_detection"

# Canonical judgment fields
JUDGMENT_FIELDS = {
    "helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    "helpsteer3": ["score"],
    "antique": ["ranking"],
    "neurips": ["rating", "confidence", "soundness", "presentation", "contribution"],
}

DATA_MAPPING = {
    "helpsteer2": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer2_train_sampled_1_grouped/gpt-4o-2024-08-06_helpsteer2_train_sampled_groups.json",
        "test": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer2_test_1_grouped/gpt-4o-2024-08-06_helpsteer2_test_groups.json",
    },
    "helpsteer3": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer3_train_1_grouped/gpt-4o-2024-08-06_helpsteer3_train_groups.json",
        "test": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer3_test_1_grouped/gpt-4o-2024-08-06_helpsteer3_test_groups.json",
    },
    "antique": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_antique_train_1_grouped/gpt-4o-2024-08-06_antique_train_groups.json",
        "test": f"{DATA_ROOT}/gpt-4o-2024-08-06_antique_test_1_grouped/gpt-4o-2024-08-06_antique_test_groups.json",
    },
    "neurips": {
        "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_neurips_train_1_grouped/gpt-4o-2024-08-06_neurips_train_groups.json",
        "test": f"{DATA_ROOT}/gpt-4o-2024-08-06_neurips_test_1_grouped/gpt-4o-2024-08-06_neurips_test_groups.json",
    },
}

def load_json(dataset: str, split: str):
    if dataset not in DATA_MAPPING or split not in DATA_MAPPING[dataset]:
        raise ValueError(f"No mapping for {dataset}/{split}")
    path = DATA_MAPPING[dataset][split]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_grouped_json(dataset: str, group_size: int):
    # Test grouped sets only (as per existing structure)
    dir_name = f"gpt-4o-2024-08-06_{dataset}_test_{group_size}_grouped"
    json_name = f"gpt-4o-2024-08-06_{dataset}_test_groups.json"
    path = os.path.join(DATA_ROOT, dir_name, json_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_rating_scale_helpsteer3(value: float, variant: str):
    # Original score assumed in [-3..3]
    if variant == "continuous":
        return float(value)
    if variant == "ternary":  # merge negatives to -1, positives to 1, keep 0
        if value <= -1: return -1.0
        if value >= 1: return 1.0
        return 0.0
    if variant == "binary":  # negatives & zero -> 0, positives -> 1
        return 1.0 if value > 0 else 0.0
    raise ValueError(f"Unknown helpsteer3 scale variant: {variant}")

def apply_rating_scale_helpsteer2(value: float, variant: str):
    # Original scale assumed 1..5
    if variant == "continuous":
        return float(value)
    if variant == "binary":  # 1/2 -> 0, 3/4/5 ->1
        return 1.0 if value >= 3 else 0.0
    if variant == "3point":  # 1/2 ->1, 3 ->2, 4/5 ->3
        if value <= 2: return 1.0
        if value == 3: return 2.0
        return 3.0
    raise ValueError(f"Unknown helpsteer2 scale variant: {variant}")

def extract_instance_features(records, dataset: str, dims_subset=None, scale_variant=None):
    dims = JUDGMENT_FIELDS[dataset]
    if dims_subset is not None:
        dims = dims[:dims_subset]
    X, y = [], []
    for item in records:
        label = 1 if str(item.get("label", "")).lower() == "llm" else 0
        ex_list = item.get("examples", []) or []
        if not ex_list:
            X.append([0.0] * len(dims))
            y.append(label)
            continue
        ex = ex_list[0]
        feat = []
        for d in dims:
            val = ex.get(d, 0.0)
            if isinstance(val, list):
                val = float(np.mean(val)) if len(val) else 0.0
            # Apply scale transforms
            if dataset == "helpsteer3" and d == "score" and scale_variant:
                val = apply_rating_scale_helpsteer3(val, scale_variant)
            if dataset == "helpsteer2" and scale_variant:
                # Apply to all judgment fields for simplicity
                val = apply_rating_scale_helpsteer2(val, scale_variant)
            feat.append(float(val))
        X.append(feat)
        y.append(label)
    X = np.nan_to_num(np.array(X, dtype=float))
    y = np.array(y, dtype=int)
    return X, y

def train_instance_model(X_train, y_train, classifier: str):
    if classifier == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", clf)
    ])
    model.fit(X_train, y_train)
    return model

def group_level_evaluate(model, grouped_records, dataset: str, dims_subset=None, scale_variant=None):
    # Build instance feature matrix for all examples across groups, tracking ids
    dims = JUDGMENT_FIELDS[dataset]
    if dims_subset is not None:
        dims = dims[:dims_subset]
    inst_feats = []
    group_ids = []
    group_labels = []
    for gid, item in enumerate(grouped_records):
        label = 1 if str(item.get("label", "")).lower() == "llm" else 0
        group_labels.append(label)
        exs = item.get("examples", []) or []
        for ex in exs:
            feat = []
            for d in dims:
                val = ex.get(d, 0.0)
                if isinstance(val, list):
                    val = float(np.mean(val)) if len(val) else 0.0
                if dataset == "helpsteer3" and d == "score" and scale_variant:
                    val = apply_rating_scale_helpsteer3(val, scale_variant)
                if dataset == "helpsteer2" and scale_variant:
                    val = apply_rating_scale_helpsteer2(val, scale_variant)
                feat.append(float(val))
            inst_feats.append(feat)
            group_ids.append(gid)
    X_inst = np.nan_to_num(np.array(inst_feats, dtype=float))
    group_ids = np.array(group_ids, dtype=int)
    group_labels = np.array(group_labels, dtype=int)
    proba = model.predict_proba(X_inst)[:, 1]
    scores = np.zeros_like(group_labels, dtype=float)
    counts = np.zeros_like(group_labels, dtype=int)
    for p, gid in zip(proba, group_ids):
        scores[gid] += p
        counts[gid] += 1
    counts[counts == 0] = 1
    scores /= counts  # mean aggregation
    y_pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(group_labels, y_pred)
    f1 = f1_score(group_labels, y_pred)
    try:
        auroc = roc_auc_score(group_labels, scores)
    except ValueError:
        auroc = float('nan')
    try:
        auprc = average_precision_score(group_labels, scores)
    except ValueError:
        auprc = float('nan')
    return acc, f1, auroc, auprc

def run_group_size_analysis(dataset: str, group_sizes, classifier: str):
    train = load_json(dataset, "train")
    test1 = load_json(dataset, "test")  # group_size=1 baseline
    Xtr, ytr = extract_instance_features(train, dataset)
    Xte, yte = extract_instance_features(test1, dataset)
    model = train_instance_model(Xtr, ytr, classifier)
    inst_pred = model.predict(Xte)
    inst_scores = model.predict_proba(Xte)[:,1]
    inst_acc = accuracy_score(yte, inst_pred)
    inst_f1 = f1_score(yte, inst_pred)
    try:
        inst_auroc = roc_auc_score(yte, inst_scores)
    except ValueError:
        inst_auroc = float('nan')
    try:
        inst_auprc = average_precision_score(yte, inst_scores)
    except ValueError:
        inst_auprc = float('nan')
    results = [
        {"dataset": dataset, "analysis": "group_size", "group_size": 1, "accuracy": inst_acc, "f1": inst_f1, "auroc": inst_auroc, "auprc": inst_auprc}
    ]
    for k in group_sizes:
        if k == 1:
            continue
        try:
            grouped = load_grouped_json(dataset, k)
        except FileNotFoundError:
            continue
        acc, f1, auroc, auprc = group_level_evaluate(model, grouped, dataset)
        results.append({"dataset": dataset, "analysis": "group_size", "group_size": k, "accuracy": acc, "f1": f1, "auroc": auroc, "auprc": auprc})
    return results

def run_rating_scale_analysis(dataset: str, variants, classifier: str):
    if dataset not in ("helpsteer2", "helpsteer3"):
        return []
    train = load_json(dataset, "train")
    test = load_json(dataset, "test")
    results = []
    for var in variants:
        Xtr, ytr = extract_instance_features(train, dataset, scale_variant=var)
        Xte, yte = extract_instance_features(test, dataset, scale_variant=var)
        model = train_instance_model(Xtr, ytr, classifier)
        y_pred = model.predict(Xte)
        scores = model.predict_proba(Xte)[:,1]
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred)
        try:
            auroc = roc_auc_score(yte, scores)
        except ValueError:
            auroc = float('nan')
        try:
            auprc = average_precision_score(yte, scores)
        except ValueError:
            auprc = float('nan')
        results.append({"dataset": dataset, "analysis": "rating_scale", "scale_variant": var, "accuracy": acc, "f1": f1, "auroc": auroc, "auprc": auprc})
    return results

def run_dimension_count_analysis(dataset: str, classifier: str):
    if dataset not in ("helpsteer2", "neurips"):
        return []
    train = load_json(dataset, "train")
    test = load_json(dataset, "test")
    total_dims = len(JUDGMENT_FIELDS[dataset])
    results = []
    for n in range(1, total_dims + 1):
        Xtr, ytr = extract_instance_features(train, dataset, dims_subset=n)
        Xte, yte = extract_instance_features(test, dataset, dims_subset=n)
        model = train_instance_model(Xtr, ytr, classifier)
        y_pred = model.predict(Xte)
        scores = model.predict_proba(Xte)[:,1]
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred)
        try:
            auroc = roc_auc_score(yte, scores)
        except ValueError:
            auroc = float('nan')
        try:
            auprc = average_precision_score(yte, scores)
        except ValueError:
            auprc = float('nan')
        results.append({"dataset": dataset, "analysis": "dimension_count", "dims_used": n, "accuracy": acc, "f1": f1, "auroc": auroc, "auprc": auprc})
    return results

def try_plot(df: pd.DataFrame, out_dir: str):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        # Group Size plots (Accuracy & AUROC)
        gs = df[df.analysis == "group_size"]
        if not gs.empty:
            for ds in gs.dataset.unique():
                sub = gs[gs.dataset == ds].sort_values("group_size")
                plt.figure(figsize=(5,3))
                plt.plot(sub.group_size, sub.accuracy, marker="o")
                plt.title(f"Group Size vs Accuracy ({ds})")
                plt.xlabel("Group Size")
                plt.ylabel("Accuracy")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"group_size_accuracy_{ds}.png"))
                plt.close()
        # Consolidated multi-panel (Accuracy & F1) replacing AUROC figure
        if not df.empty:
            try:
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                mpl.rcParams['axes.spines.top'] = False
                mpl.rcParams['axes.spines.right'] = False
                fig, axes = plt.subplots(2, 3, figsize=(14,8))
                # Helper to plot metric across datasets
                def plot_metric(subdf, x_col, metric, ax, title):
                    if subdf.empty:
                        ax.set_axis_off(); return
                    for ds in subdf.dataset.unique():
                        sub = subdf[subdf.dataset == ds].sort_values(x_col)
                        ax.plot(sub[x_col], sub[metric], marker='o', label=ds)
                    ax.set_xlabel(x_col.replace('_',' ').title())
                    ax.set_ylabel(metric.title())
                    ax.set_title(title)
                    ax.grid(alpha=0.3)
                gs = df[df.analysis == 'group_size']
                rs = df[df.analysis == 'rating_scale']
                dc = df[df.analysis == 'dimension_count']
                # Order rating scales
                if not rs.empty:
                    scale_order = ['continuous','binary','ternary','3point']
                    rs['__order'] = rs.scale_variant.apply(lambda x: scale_order.index(x) if x in scale_order else 999)
                    rs = rs.sort_values('__order')
                # Accuracy row
                plot_metric(gs, 'group_size', 'accuracy', axes[0,0], 'Accuracy vs Group Size')
                if not rs.empty:
                    for ds in rs.dataset.unique():
                        sub = rs[rs.dataset == ds]
                        axes[0,1].plot(sub.scale_variant, sub.accuracy, marker='o', label=ds)
                    axes[0,1].set_xlabel('Rating Scale Variant')
                    axes[0,1].set_ylabel('Accuracy')
                    axes[0,1].set_title('Accuracy vs Rating Scale')
                    axes[0,1].grid(alpha=0.3)
                plot_metric(dc, 'dims_used', 'accuracy', axes[0,2], 'Accuracy vs Dimension Count')
                # F1 row
                plot_metric(gs, 'group_size', 'f1', axes[1,0], 'F1 vs Group Size')
                if not rs.empty:
                    for ds in rs.dataset.unique():
                        sub = rs[rs.dataset == ds]
                        axes[1,1].plot(sub.scale_variant, sub.f1, marker='o', label=ds)
                    axes[1,1].set_xlabel('Rating Scale Variant')
                    axes[1,1].set_ylabel('F1')
                    axes[1,1].set_title('F1 vs Rating Scale')
                    axes[1,1].grid(alpha=0.3)
                plot_metric(dc, 'dims_used', 'f1', axes[1,2], 'F1 vs Dimension Count')
                # Shared legend
                handles, labels = [], []
                for ax in axes.flatten():
                    h,l = ax.get_legend_handles_labels()
                    handles.extend(h); labels.extend(l)
                if handles:
                    dedup = {}
                    for h,l in zip(handles, labels):
                        if l not in dedup:
                            dedup[l] = h
                fig.tight_layout(rect=(0,0,1,0.90))
                fig.suptitle('LLM Judgment Detectability (Accuracy & F1)', y=0.98)
                # Single legend top-left (avoid title overlap)
                if handles:
                    fig.legend(dedup.values(), dedup.keys(), loc='upper left', bbox_to_anchor=(0.01,0.99), frameon=False, ncol=min(len(dedup),4))
                out_path = os.path.join(out_dir, 'figure6_style_accuracy_f1.png')
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Failed consolidated Accuracy/F1 plot: {e}")
                if 'auroc' in sub.columns:
                    plt.figure(figsize=(5,3))
                    plt.plot(sub.group_size, sub.auroc, marker="o", color='orange')
                    plt.title(f"Group Size vs AUROC ({ds})")
                    plt.xlabel("Group Size")
                    plt.ylabel("AUROC")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"group_size_auroc_{ds}.png"))
                    plt.close()
        # Rating scale plots
        rs = df[df.analysis == "rating_scale"]
        if not rs.empty:
            for ds in rs.dataset.unique():
                sub = rs[rs.dataset == ds]
                plt.figure(figsize=(5,3))
                plt.bar(sub.scale_variant, sub.accuracy)
                plt.title(f"Rating Scale vs Accuracy ({ds})")
                plt.xlabel("Scale Variant")
                plt.ylabel("Accuracy")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"rating_scale_accuracy_{ds}.png"))
                plt.close()
                if 'auroc' in sub.columns:
                    plt.figure(figsize=(5,3))
                    plt.bar(sub.scale_variant, sub.auroc, color='orange')
                    plt.title(f"Rating Scale vs AUROC ({ds})")
                    plt.xlabel("Scale Variant")
                    plt.ylabel("AUROC")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"rating_scale_auroc_{ds}.png"))
                    plt.close()
        # Dimension count plots
        dc = df[df.analysis == "dimension_count"]
        if not dc.empty:
            for ds in dc.dataset.unique():
                sub = dc[dc.dataset == ds]
                plt.figure(figsize=(5,3))
                plt.plot(sub.dims_used, sub.accuracy, marker="o")
                plt.title(f"Dims Used vs Accuracy ({ds})")
                plt.xlabel("Dimensions Used")
                plt.ylabel("Accuracy")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"dimension_count_accuracy_{ds}.png"))
                plt.close()
                if 'auroc' in sub.columns:
                    plt.figure(figsize=(5,3))
                    plt.plot(sub.dims_used, sub.auroc, marker="o", color='orange')
                    plt.title(f"Dims Used vs AUROC ({ds})")
                    plt.xlabel("Dimensions Used")
                    plt.ylabel("AUROC")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"dimension_count_auroc_{ds}.png"))
                    plt.close()
    except ImportError:
        print("⚠️ matplotlib not installed; skipping plots. Install via: pip install matplotlib")

def main():
    parser = argparse.ArgumentParser(description="Detectability analysis experiments")
    parser.add_argument("--datasets", nargs="*", default=["helpsteer2", "helpsteer3", "neurips", "antique"], help="Datasets to include (antique added by default)")
    parser.add_argument("--group_sizes", nargs="*", type=int, default=[2,4,8,16], help="Group sizes for group-level analysis (k>1). k=1 always included")
    parser.add_argument("--rating_scales_helpsteer2", nargs="*", default=["continuous","binary","3point"], help="Scale variants for helpsteer2")
    parser.add_argument("--rating_scales_helpsteer3", nargs="*", default=["continuous","binary","ternary"], help="Scale variants for helpsteer3")
    parser.add_argument("--classifier", choices=["rf","logistic"], default="rf")
    parser.add_argument("--run_group_size", action="store_true")
    parser.add_argument("--run_rating_scale", action="store_true")
    parser.add_argument("--run_dimension_count", action="store_true")
    parser.add_argument("--plots_dir", type=str, default="analysis_plots", help="Directory to save plots")
    parser.add_argument("--save_csv", type=str, default="", help="Optional path to save combined results CSV")
    args = parser.parse_args()

    all_results = []
    clf = args.classifier

    for ds in args.datasets:
        print(f"\n=== Dataset: {ds} ===")
        if args.run_group_size:
            try:
                gs_res = run_group_size_analysis(ds, args.group_sizes, clf)
                all_results.extend(gs_res)
                print("Group size analysis done.")
            except Exception as e:
                print(f"⚠️ Group size analysis failed for {ds}: {e}")
        if args.run_rating_scale and ds in ("helpsteer2","helpsteer3"):
            variants = args.rating_scales_helpsteer2 if ds == "helpsteer2" else args.rating_scales_helpsteer3
            try:
                rs_res = run_rating_scale_analysis(ds, variants, clf)
                all_results.extend(rs_res)
                print("Rating scale analysis done.")
            except Exception as e:
                print(f"⚠️ Rating scale analysis failed for {ds}: {e}")
        if args.run_dimension_count and ds in ("helpsteer2","neurips"):
            try:
                dc_res = run_dimension_count_analysis(ds, clf)
                all_results.extend(dc_res)
                print("Dimension count analysis done.")
            except Exception as e:
                print(f"⚠️ Dimension count analysis failed for {ds}: {e}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nCombined Results Preview:")
        print(df.head())
        try_plot(df, args.plots_dir)
        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
            df.to_csv(args.save_csv, index=False)
            print(f"💾 Saved results CSV to {args.save_csv}")
    else:
        print("No results generated. Check flags.")

if __name__ == "__main__":
    main()
