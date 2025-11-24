import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List


LINGUISTIC_DIR = os.path.join("data", "features", "linguistic_feature")
LLM_ENHANCED_DIR = os.path.join("data", "features", "llm_enhanced_features")


_LING_MAP = {
    "helpsteer2": {"train": "helpsteer2_train.csv", "test": "helpsteer2_test.csv"},
    "helpsteer3": {"train": "helpsteer3_train.csv", "test": "helpsteer3_test.csv"},
    "antique": {"train": "ANTIQUE_train.csv", "test": "ANTIQUE_test.csv"},
    "neurips": {"train": "neurips_train.csv", "test": "neurips_test.csv"},
}

_LLM_MAP = {
    "helpsteer2": {"train": "helpsteer2_train_Qwen3-8B.json", "test": "helpsteer2_test_Qwen3-8B.json"},
    "helpsteer3": {"train": "helpsteer3_train_Qwen3-8B.json", "test": "helpsteer3_test_Qwen3-8B.json"},
    "antique": {"train": "ANTIQUE_train_Qwen3-8B.json", "test": "ANTIQUE_test_Qwen3-8B.json"},
    "neurips": {"train": "neurips_train_Qwen3-8B.json", "test": "neurips_test_Qwen3-8B.json"},
}


def _ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")


def load_linguistic_features(dataset: str, split: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load precomputed linguistic features for a dataset/split from CSV.

    - Drops obvious text columns like prompt/response/response1/response2.
    - Keeps only numeric columns.
    - Returns (X, feature_names).
    """
    if dataset not in _LING_MAP or split not in _LING_MAP[dataset]:
        raise ValueError(f"No linguistic features mapping for {dataset}/{split}")

    path = os.path.join(LINGUISTIC_DIR, _LING_MAP[dataset][split])
    _ensure_exists(path)

    # Read CSV. Large files are expected; pandas handles chunking internally if needed.
    df = pd.read_csv(path)

    # Drop obvious text columns if present
    drop_cols = [c for c in df.columns if c.lower() in {"prompt", "response", "response1", "response2", "context"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep numeric only
    num_df = df.select_dtypes(include=["number"]).copy()
    feature_names = list(num_df.columns)
    X = num_df.to_numpy(dtype=float)
    return X, feature_names


def load_llm_enhanced_features(dataset: str, split: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load LLM-enhanced features from JSON.

    - For pairwise datasets (e.g., helpsteer3), use llm_enhanced_feature_r1 / _r2.
    - For single-response datasets (e.g., helpsteer2-style, neurips, antique),
      use llm_enhanced_feature.
    """
    if dataset not in _LLM_MAP or split not in _LLM_MAP[dataset]:
        raise ValueError(f"No LLM-enhanced features mapping for {dataset}/{split}")

    path = os.path.join(LLM_ENHANCED_DIR, _LLM_MAP[dataset][split])
    _ensure_exists(path)

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Detect structure: r1/r2 vs single-dict
    has_r1r2 = any(
        ("llm_enhanced_feature_r1" in rec) or ("llm_enhanced_feature_r2" in rec)
        for rec in records
    )
    has_single = any("llm_enhanced_feature" in rec for rec in records)

    X_list: List[List[float]] = []
    feature_names: List[str] = []

    if has_r1r2 and not has_single:
        # === helpsteer3-style: r1 and r2 dicts ===
        r1_keys = set()
        r2_keys = set()
        for rec in records:
            r1 = rec.get("llm_enhanced_feature_r1", {}) or {}
            r2 = rec.get("llm_enhanced_feature_r2", {}) or {}
            r1_keys.update([k for k, v in r1.items() if isinstance(v, (int, float))])
            r2_keys.update([k for k, v in r2.items() if isinstance(v, (int, float))])

        r1_keys = sorted(r1_keys)
        r2_keys = sorted(r2_keys)
        feature_names = [f"r1::{k}" for k in r1_keys] + [f"r2::{k}" for k in r2_keys]

        for rec in records:
            r1 = rec.get("llm_enhanced_feature_r1", {}) or {}
            r2 = rec.get("llm_enhanced_feature_r2", {}) or {}
            row = []
            for k in r1_keys:
                v = r1.get(k, 0.0)
                row.append(float(v) if isinstance(v, (int, float)) else 0.0)
            for k in r2_keys:
                v = r2.get(k, 0.0)
                row.append(float(v) if isinstance(v, (int, float)) else 0.0)
            X_list.append(row)

    elif has_single:
        # === helpsteer2 / neurips-style: single llm_enhanced_feature dict ===
        all_keys = set()
        for rec in records:
            d = rec.get("llm_enhanced_feature", {}) or {}
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    all_keys.add(k)

        ordered = sorted(all_keys)
        feature_names = list(ordered)

        for rec in records:
            d = rec.get("llm_enhanced_feature", {}) or {}
            row = [
                float(d.get(k, 0.0)) if isinstance(d.get(k, 0), (int, float)) else 0.0
                for k in ordered
            ]
            X_list.append(row)
    else:
        # No usable numeric features – return empty matrix
        X = np.zeros((len(records), 0), dtype=float)
        return X, []

    X = np.array(X_list, dtype=float)
    return X, feature_names

