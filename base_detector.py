import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


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

    # Explicit mapping based on your screenshot
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
        # # Add later when available
        # "neurips": {
        #     "train": f"{data_root}/gpt-4o-2024-08-06_neurips_train_1_grouped/gpt-4o-2024-08-06_neurips_train_groups.json",
        #     "test": f"{data_root}/gpt-4o-2024-08-06_neurips_test_1_grouped/gpt-4o-2024-08-06_neurips_test_groups.json"
        # }
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

            

for dataset in ["helpsteer2","helpsteer3","antique"]:
    train_data=load_dataset(data_root, dataset, "train")
    test_data=load_dataset(data_root, dataset, "test")

    X_train,y_train = extract_features_labels(train_data, dataset)
    X_test,y_test = extract_features_labels(test_data, dataset)

    print(f"\n📊 {dataset.upper()} DATA CHECK")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Human (0): {np.sum(y_train==0)}, LLM (1): {np.sum(y_train==1)}")

    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    print("Feature means:", X_means)
    print("Feature stds :", X_stds)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(f"=== Results for {dataset} ===")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test,y_pred))

    results.append({
        "dataset": dataset,
        "accuracy": acc,
        "f1": f1
    })

