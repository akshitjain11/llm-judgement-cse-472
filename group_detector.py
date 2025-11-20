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
    "helpsteer2": ["helpfullness","correctness","coherence","complexiity","verbosity"],
    "helpsteer3": ["score"],
    "antique": ["rating","confidence","soundness","presentation","contribution"],
    "neurips": ["ranking"]
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
        label = 1 if str(item.get("label","")).lower()=="llm" else 0
        examples = item.get("examples", []) or []

        if not examples:
            x.append([0.0] * len(dims))
            y.append(label)
            continue

        ex = examples[0]
        feat = []
        for d in dims:
            val = ex.get(d, 0.0)
            if isinstance(val,list):
                val = float(np.mean(val)) if len(val) > 0 else 0.0
            feat.append(float(val))

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
                if isinstance(val, list):
                    val = float(np.mean(val)) if len(val) > 0 else 0.0
                feat.append(float(val))
            X_instances.append(feat)
            group_ids.append(g_idx)

    X_instances = np.nan_to_num(np.array(X_instances, dtype=float))
    group_ids = np.array(group_ids, dtype=int)
    group_labels = np.array(group_labels, dtype=int)

    return X_instances, group_ids, group_labels


def evaluate_group_level(model,dataset:str,group_size:int):
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
    group_scores /= counts

    y_true = group_labels
    y_pred = (group_scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{dataset.upper()} Group-Level Evaluation (group_size={group_size}):")
    print(f"Groups: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "dataset": dataset,
        "group_size": group_size,
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
            res = evaluate_group_level(model, ds, k)
            if res is not None:
                all_results.append(res)

    if args.save_results:
        out_path = args.save_results
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n💾 Saved group-level results to {out_path}")


if __name__ == "__main__":
    main()
