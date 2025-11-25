# On the Detectability of LLM-generated Judgments

This project implements a comprehensive framework for detecting LLM-generated vs. human-generated judgments across multiple datasets. It includes base classifiers, augmented feature-based detectors, group-level detection, detectability analysis across various parameters, and bias quantification.

## Project Overview

### Tasks

1. **Base Detector Implementation** - Classify judgments using only judgment dimensions
2. **Augmented Detector** - Enhance detection with linguistic and LLM-enhanced features
3. **Group-Level Detector** - Classify groups of judgments based on aggregated logits
4. **Detectability Analysis** - Analyze how detectability changes with:
   - Group size (k = 1, 2, 4, 8, 16)
   - Rating scale variations (binary, ternary, continuous)
   - Number of judgment dimensions
5. **Bias Quantification** (Bonus) - Analyze GPT-4o's judgment bias using model interpretability

### Datasets

- **Helpsteer2** - 5 judgment dimensions: helpfulness, correctness, coherence, complexity, verbosity
- **Helpsteer3** - Single score dimension
- **NEURIPS** - 5 dimensions: rating, confidence, soundness, presentation, contribution
- **ANTIQUE** - Ranking list dimension

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone or download the project
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Project476/
├── base_detector.py              # Task 1 & 2: Base and augmented classifiers
├── group_detector.py             # Task 3: Group-level detection
├── detectability_analysis.py     # Task 4: Parameter sensitivity analysis
├── bias_analysis.py              # Task 5: Bias quantification
├── feature_loader.py             # Utility: Load linguistic and LLM-enhanced features
├── requirements.txt              # Python dependencies
├── README.md                      # This file
└── data/
    ├── dataset_detection/        # Grouped judgment datasets
    │   ├── gpt-4o-2024-08-06_helpsteer2_*/
    │   ├── gpt-4o-2024-08-06_helpsteer3_*/
    │   ├── gpt-4o-2024-08-06_neurips_*/
    │   └── gpt-4o-2024-08-06_antique_*/
    └── features/
        ├── linguistic_feature/   # Linguistic features (CSV)
        └── llm_enhanced_features/# LLM-enhanced features (JSON)
```

## Quick Start

### Task 1: Base Detector

Run the base classifier using only judgment dimensions:

```bash
python base_detector.py
```

Options:
```bash
# Specify datasets
python base_detector.py --datasets helpsteer2 helpsteer3 neurips antique

# Use Logistic Regression (default: Random Forest)
python base_detector.py --classifier logistic

# Save results
python base_detector.py --save_results results_base.csv
```

**Output:** Accuracy and F1 scores for each dataset

---

### Task 2: Augmented Detector

Enhance detection with linguistic and LLM-enhanced features:

```bash
# Add linguistic features
python base_detector.py --use_linguistic

# Add LLM-enhanced features
python base_detector.py --use_llm_enhanced

# Add both
python base_detector.py --use_linguistic --use_llm_enhanced

# With debugging
python base_detector.py --use_linguistic --use_llm_enhanced --debug

# Full options for helpsteer3
python base_detector.py --datasets helpsteer3 --use_linguistic --use_llm_enhanced --hs3_full_join
```

**Output:** Improved accuracy/F1 with augmented features + comparison with base detector

---

### Task 3: Group-Level Detector

Classify groups of judgments using instance-level predictions:

```bash
# Default: trains on group_size=1, evaluates on 2,4,8,16
python group_detector.py

# Specify datasets and group sizes
python group_detector.py --datasets helpsteer2 neurips --group_sizes 2 4 8 16

# Use Logistic Regression
python group_detector.py --classifier logistic

# Use sum aggregation (default: mean)
python group_detector.py --agg_method sum

# Save results
python group_detector.py --save_results group_level_results.csv
```

**Output:**
- Instance-level metrics (baseline for k=1)
- Group-level accuracy/F1 for each group size
- Classification reports

---

### Task 4: Detectability Analysis

Comprehensive analysis of detectability across parameters:

```bash
# Run all three analyses
python detectability_analysis.py --run_group_size --run_rating_scale --run_dimension_count

# Group size analysis only
python detectability_analysis.py --run_group_size --group_sizes 2 4 8 16

# Rating scale analysis (helpsteer2 & helpsteer3 only)
python detectability_analysis.py --run_rating_scale

# Dimension count analysis (helpsteer2 & neurips only)
python detectability_analysis.py --run_dimension_count

# With plotting and CSV output
python detectability_analysis.py --run_group_size --run_rating_scale --run_dimension_count \
  --save_csv analysis_results.csv --plots_dir analysis_plots/
```

**Analysis Details:**

**a) Group Size Analysis**
- Varies group size k from 1 to 16
- Shows how accuracy/F1 changes with group aggregation
- Output: Accuracy vs. group size plots

**b) Rating Scale Analysis**
- Helpsteer2 variants: continuous, binary (1-2→0, 3-5→1), 3-point
- Helpsteer3 variants: continuous, binary (≤0→0, >0→1), ternary (-1, 0, 1)
- Output: Accuracy comparison across rating scales

**c) Dimension Count Analysis**
- Varies number of judgment dimensions from 1 to all available
- Helpsteer2: 1-5 dimensions, Neurips: 1-5 dimensions
- Output: Accuracy vs. dimension count curves

**Output:**
- CSV file with results (accuracy, F1, AUROC, AUPRC)
- Multi-panel figure (Figure 6 style) showing:
  - Accuracy/F1 vs. group size
  - Accuracy/F1 vs. rating scale
  - Accuracy/F1 vs. dimension count
- Individual plots for each analysis type

---

### Task 5: Bias Quantification (Bonus)

Analyze GPT-4o's judgment bias using model interpretability:

```bash
# Default: analyze helpsteer2 and neurips
python bias_analysis.py

# Specify datasets
python bias_analysis.py --datasets helpsteer2 neurips

# Use Logistic Regression (better for interpretability)
python bias_analysis.py --model logistic

# Customize top-k features
python bias_analysis.py --top_k 30

# Custom output directory
python bias_analysis.py --output_dir my_bias_results/
```

**Output Files:**
- `bias_top_features_{dataset}.csv` - Top k features ranked by importance
- `bias_intrinsic_stats_{dataset}.csv` - Bias statistics for judgment dimensions:
  - Mean values for human and LLM
  - Differences
  - Cohen's d (effect size)
  - Point-biserial correlation
- `bias_top_features_{dataset}.png` - Visualization of top features (color-coded by category)
- `bias_summary.csv` - Summary across datasets


---

## Complete Workflow Example

```bash
# Create output directory
mkdir results

# Task 1: Base detector
python base_detector.py --save_results results/task1_base.csv

# Task 2: Augmented detector
python base_detector.py --use_linguistic --use_llm_enhanced --save_results results/task2_augmented.csv

# Task 3: Group detector
python group_detector.py --group_sizes 2 4 8 16 \
  --save_results results/task3_groups.csv

# Task 4: Detectability analysis
python detectability_analysis.py --run_group_size --run_rating_scale --run_dimension_count \
  --save_csv results/task4_analysis.csv --plots_dir results/task4_plots/

# Task 5: Bias analysis
python bias_analysis.py --output_dir results/task5_bias/
```

## Key Features

### Classifiers Supported
- **Random Forest** (default, robust to non-linear patterns)
- **Logistic Regression** (faster, better for interpretability)

### Feature Types
1. **Judgment Dimensions** - Intrinsic judgment scales (e.g., helpfulness, correctness)
2. **Linguistic Features** - Linguistic properties extracted from responses (pre-computed)
3. **LLM-Enhanced Features** - Features from auxiliary LLM analysis (pre-computed)

### Aggregation Methods (for group-level)
- **Mean** - Average of instance probabilities (default)
- **Sum** - Sum of instance probabilities (with adaptive threshold)

### Output Metrics
- **Accuracy** - Overall classification accuracy
- **F1 Score** - Harmonic mean of precision and recall
- **AUROC** - Area under ROC curve
- **AUPRC** - Area under PR curve
- **Classification Report** - Per-class precision, recall, F1

## Data Format

### Dataset Files (JSON)
Each group contains:
```json
{
  "label": "llm" or "human",
  "examples": [
    {
      "judgment_field_1": value,
      "judgment_field_2": value,
      ...
    }
  ]
}
```

### Feature Files (CSV)
Linguistic features with numeric columns and metadata:
- Automatically aligned with judgment data
- Rows correspond to judgments/pairs
- Numeric columns selected automatically

### LLM-Enhanced Features (JSON)
LLM analysis results stored per record:
- Single-response format: `llm_enhanced_feature`
- Pairwise format (helpsteer3): `llm_enhanced_feature_r1`, `llm_enhanced_feature_r2`

## Troubleshooting

### Feature Alignment Issues
Use the `--debug` flag to see matching statistics:
```bash
python base_detector.py --use_linguistic --use_llm_enhanced --debug
```

### Helpsteer3 Augmentation
If augmentation fails for helpsteer3, try the full-join strategy:
```bash
python base_detector.py --datasets helpsteer3 --use_linguistic --use_llm_enhanced --hs3_full_join
```

### Missing Plots
Ensure matplotlib is installed:
```bash
pip install matplotlib
```

### Data File Issues
Verify data structure:
```bash
# Check if directories exist
ls data/dataset_detection/
ls data/features/linguistic_feature/
ls data/features/llm_enhanced_features/
```


## References

- Paper 1: "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge" (Sections 1-2)
- Paper 2: "Who's Your Judge? On the Detectability of LLM-Generated Judgments"




