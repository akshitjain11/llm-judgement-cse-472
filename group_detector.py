import os
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


JUDGMENT_FIELDS = {
    # Corrected spellings and dataset field mappings to match base_detector.py
    # helpsteer2 has five scalar judgment fields
    "helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    # helpsteer3 single score per example
    "helpsteer3": ["score"],
    # antique provides a ranking list; we reduce via mean in feature extraction
    "antique": ["ranking"],
    # neurips uses panel-style multi-dimension review fields
    "neurips": ["rating", "confidence", "soundness", "presentation", "contribution"],
}

DATA_ROOT = "data/dataset_detection"

def load_instance_dataset(dataset: str, split: str):
    """
    Load group_size=1 data for training an instance-level model.
    (Same group-of-1 JSON you already used in base_detector.py)
    """
    mapping = {
        "helpsteer2": {
            "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer2_train_sampled_1_grouped/"
                     f"gpt-4o-2024-08-06_helpsteer2_train_sampled_groups.json",
            "test":  f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer2_test_1_grouped/"
                     f"gpt-4o-2024-08-06_helpsteer2_test_groups.json",
        },
        "helpsteer3": {
            "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer3_train_1_grouped/"
                     f"gpt-4o-2024-08-06_helpsteer3_train_groups.json",
            "test":  f"{DATA_ROOT}/gpt-4o-2024-08-06_helpsteer3_test_1_grouped/"
                     f"gpt-4o-2024-08-06_helpsteer3_test_groups.json",
        },
        "antique": {
            "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_antique_train_1_grouped/"
                     f"gpt-4o-2024-08-06_antique_train_groups.json",
            "test":  f"{DATA_ROOT}/gpt-4o-2024-08-06_antique_test_1_grouped/"
                     f"gpt-4o-2024-08-06_antique_test_groups.json",
        },
        "neurips": {
            "train": f"{DATA_ROOT}/gpt-4o-2024-08-06_neurips_train_1_grouped/"
                     f"gpt-4o-2024-08-06_neurips_train_groups.json",
            "test":  f"{DATA_ROOT}/gpt-4o-2024-08-06_neurips_test_1_grouped/"
                     f"gpt-4o-2024-08-06_neurips_test_groups.json",
        },
    }

    if dataset not in mapping or split not in mapping[dataset]:
        raise ValueError(f"No mapping for dataset={dataset}, split={split}")

    path = mapping[dataset][split]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance dataset not found: {path}")

    print(f"✅ Loading instance-level {dataset} [{split}] → {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    


def extract_base_features_labels_from_groups(data, dataset: str):
    """
    For group_size=1 JSON:
    -Each item has one example in item ["examples"][0]
    -We extract base judgment fields as features
    """

    dims = JUDGMENT_FIELDS[dataset]
    x, y = [], []

    for item in data:
        label = 1 if str(item.get("label","")) .lower()=="llm" else 0
        examples = item.get("examples", []) or []

        if not examples:
            # Handle empty by zero vector (dimension decided below per dataset)
            if dataset == "antique":
                x.append([0.0]*11)  # mean,std,min,max,range,entropy,median,skew,kurtosis,top_minus_bottom,top_minus_median
            else:
                x.append([0.0] * len(dims))
            y.append(label)
            continue

        ex = examples[0]
        feat = []
        for d in dims:
            val = ex.get(d, 0.0)
            if dataset == "antique" and d == "ranking" and isinstance(val, list):
                arr = np.array(val, dtype=float) if len(val) > 0 else np.array([0.0])
                # Sort for positional contrasts
                sorted_arr = np.sort(arr)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                mn = float(sorted_arr[0])
                mx = float(sorted_arr[-1])
                rng = mx - mn
                median = float(np.median(arr))
                # Skewness and kurtosis (Fisher definition, excess kurtosis)
                if std > 1e-12 and arr.size > 2:
                    centered = arr - mean
                    m3 = np.mean(centered**3)
                    m4 = np.mean(centered**4)
                    skew = float(m3 / (std**3))
                    kurt = float(m4 / (std**4) - 3.0)
                else:
                    skew = 0.0
                    kurt = 0.0
                top_minus_bottom = mx - mn  # same as range kept for clarity
                top_minus_median = mx - median
                eps = 1e-9
                pos = np.abs(arr) + eps
                probs = pos / np.sum(pos)
                entropy = float(-np.sum(probs * np.log(probs + eps)))
                feat.extend([mean, std, mn, mx, rng, entropy, median, skew, kurt, top_minus_bottom, top_minus_median])
            else:
                if isinstance(val,list):
                    val = float(np.mean(val)) if len(val) > 0 else 0.0
                feat.append(float(val))

        # Ensure consistent dimensionality for antique (exact 11 features)
        if dataset == "antique" and len(feat) != 11:
            while len(feat) < 11:
                feat.append(0.0)

        x.append(feat)
        y.append(label)

    x = np.nan_to_num(np.array(x,dtype=float))
    y = np.array(y,dtype=int)
    return x, y 

def train_instance_model(dataset: str, classifier:str = "rf",show_instance_metrics: bool = True):
    train_data = load_instance_dataset(dataset, "train")
    test_data = load_instance_dataset(dataset, "test")

    x_train, y_train = extract_base_features_labels_from_groups(train_data, dataset)
    x_test, y_test = extract_base_features_labels_from_groups(test_data, dataset)

    print(f"\n {dataset.upper()} Instance-Level Model Training (group_size=1) CHECK")
    print(f"Train: {len(x_train)} examples, Test: {len(x_test)} examples")
    print(f"Human (0): {np.sum(y_train==0)}, LLM (1): {np.sum(y_train==1)}")
    print(f"Base features: {x_train.shape[1]}")

    if classifier == "logistic":
        clf = LogisticRegression(max_iter=2000,class_weight="balanced", random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200,random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("classifier", clf)
    ])

    model.fit(x_train, y_train)

    if show_instance_metrics:
        y_pred = model.predict(x_test)
        print("\nInstance-Level Metrics:")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    return model


def load_grouped_dataset(dataset:str, group_size:int):
    """
    Load grouped dataset for training a group-level model.
    """
    dir_name = f"gpt-4o-2024-08-06_{dataset}_test_{group_size}_grouped"
    json_name = f"gpt-4o-2024-08-06_{dataset}_test_groups.json"
    path = os.path.join(DATA_ROOT, dir_name, json_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Grouped dataset not found for {dataset}: {path}")
    
    print(f"✅ Loading grouped {dataset} (group_size={group_size}) → {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def extract_instance_features_from_grouped(data, dataset: str):
    """
    For grouped JSON (group_size >= 1):
    - Each item is a group containing multiple examples.
    - We return:
        X_instances: feature matrix for each example
        group_ids: group index for each example (0..num_groups-1)
        group_labels: label per group (0=human, 1=llm)
    """
    dims = JUDGMENT_FIELDS[dataset]
    X_instances = []
    group_ids = []
    group_labels = []

    for g_idx, item in enumerate(data):
        label = 1 if str(item.get("label", "")).lower() == "llm" else 0
        group_labels.append(label)

        examples = item.get("examples", []) or []
        for ex in examples:
            feat = []
            for d in dims:
                val = ex.get(d, 0.0)
                if dataset == "antique" and d == "ranking" and isinstance(val, list):
                    arr = np.array(val, dtype=float) if len(val) > 0 else np.array([0.0])
                    sorted_arr = np.sort(arr)
                    mean = float(np.mean(arr))
                    std = float(np.std(arr))
                    mn = float(sorted_arr[0])
                    mx = float(sorted_arr[-1])
                    rng = mx - mn
                    median = float(np.median(arr))
                    if std > 1e-12 and arr.size > 2:
                        centered = arr - mean
                        m3 = np.mean(centered**3)
                        m4 = np.mean(centered**4)
                        skew = float(m3 / (std**3))
                        kurt = float(m4 / (std**4) - 3.0)
                    else:
                        skew = 0.0
                        kurt = 0.0
                    top_minus_bottom = mx - mn
                    top_minus_median = mx - median
                    eps = 1e-9
                    pos = np.abs(arr) + eps
                    probs = pos / np.sum(pos)
                    entropy = float(-np.sum(probs * np.log(probs + eps)))
                    feat.extend([mean, std, mn, mx, rng, entropy, median, skew, kurt, top_minus_bottom, top_minus_median])
                else:
                    if isinstance(val, list):
                        val = float(np.mean(val)) if len(val) > 0 else 0.0
                    feat.append(float(val))
            if dataset == "antique" and len(feat) != 11:
                while len(feat) < 11:
                    feat.append(0.0)
            X_instances.append(feat)
            group_ids.append(g_idx)

    X_instances = np.nan_to_num(np.array(X_instances, dtype=float))
    group_ids = np.array(group_ids, dtype=int)
    group_labels = np.array(group_labels, dtype=int)

    return X_instances, group_ids, group_labels


def evaluate_group_level(model,dataset:str,group_size:int, agg_method: str = "mean"):
    data = load_grouped_dataset(dataset, group_size)
    if data is None:
        print(f"⚠️ No grouped data for {dataset} (group_size={group_size})")
        return
    
    x_inst,group_ids,group_labels = extract_instance_features_from_grouped(data, dataset)

    proba = model.predict_proba(x_inst)[:,1]

    num_groups = group_labels.shape[0]
    group_scores = np.zeros(num_groups,dtype=float)
    counts = np.zeros(num_groups,dtype=int)

    for p,gid in zip(proba, group_ids):
        group_scores[gid] += p
        counts[gid] += 1

    counts[counts==0] = 1
    if agg_method == "mean":
        group_scores /= counts
        threshold = 0.5  # standard probability threshold
    elif agg_method == "sum":
        # Sum of probabilities; adapt decision threshold proportionally to group size.
        # Rationale: if individual probs are calibrated around 0.5, a sum > 0.5*size indicates majority leaning LLM.
        threshold = 0.5 * counts  # vector threshold per group
    else:
        raise ValueError(f"Unsupported aggregation method: {agg_method}")

    y_true = group_labels
    if agg_method == "sum":
        # threshold is an array when using sum
        y_pred = (group_scores >= threshold).astype(int)
    else:
        y_pred = (group_scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{dataset.upper()} Group-Level Evaluation (group_size={group_size}):")
    print(f"Groups: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "dataset": dataset,
        "group_size": group_size,
        "agg_method": agg_method,
        "accuracy": float(acc),
        "f1_score": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Group-level detector: aggregate instance-level logits to classify groups."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["helpsteer2", "helpsteer3", "antique", "neurips"],
        help="Datasets to run: helpsteer2, helpsteer3, antique, neurips",
    )
    parser.add_argument(
        "--group_sizes",
        nargs="*",
        type=int,
        default=[2, 4, 8, 16],
        help="Group sizes to evaluate at test time (e.g., 2 4 8 16)",
    )
    parser.add_argument(
        "--classifier",
        choices=["rf", "logistic"],
        default="rf",
        help="Classifier type for instance-level model",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default="",
        help="Optional path to save aggregated group-level results as CSV",
    )
    parser.add_argument(
        "--agg_method",
        choices=["mean", "sum"],
        default="mean",
        help="Aggregation of instance probabilities per group (mean or sum)",
    )

    args = parser.parse_args()

    all_results = []

    for ds in args.datasets:
        print("\n" + "=" * 80)
        print(f"🧪 Dataset: {ds}")
        print("=" * 80)

        # 1. Train instance-level model on group_size=1
        model = train_instance_model(ds, classifier=args.classifier, show_instance_metrics=True)

        # 2. Evaluate group-level performance for each k in group_sizes
        for k in args.group_sizes:
            res = evaluate_group_level(model, ds, k, agg_method=args.agg_method)
            if res is not None:
                all_results.append(res)

    if args.save_results:
        out_path = args.save_results
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n💾 Saved group-level results to {out_path}")


if __name__ == "__main__":
    main()
