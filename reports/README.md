# Model Performance Reports

This directory contains the evaluation results for different stages of the model training pipeline.

## Files
- **`baseline_report.csv`**: Performance of the model after **Linear Probing** (Stage 1). Only the custom head was trained; the ResNet34 backbone was frozen.
- **`finetuned_report.csv`**: Performance of the model after **Fine-Tuning** (Stage 2). The `layer4` of the backbone was unfrozen and trained with a lower learning rate.

## Key Findings

### Overall Performance Improvement
The fine-tuning stage yielded a significant jump in performance across all metrics.

| Metric | Baseline (Stage 1) | Fine-Tuned (Stage 2) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 51.9% | **77.2%** | +25.3% |
| **Weighted F1-Score** | 0.50 | **0.77** | +0.27 |

### Why Fine-Tuning Mattered
The baseline model struggled with the specific stylized features of anime characters (large eyes, specific shadings) because it was pre-trained on ImageNet (real-world photos). Unfreezing `layer4` allowed the model to adapt its high-level feature extractors to the domain of anime illustrations.

### Specific Improvements
- **Hard Cases Solved**: Characters like `kaname_madoka` and `sengoku_nadeko` went from **0.0 F1-Score** in the baseline to respectable scores (around **0.4 - 0.5**), proving the model learned to recognize previously indistinguishable features.
- **Perfect Matches**: Several unique characters achieved near **100% Precision/Recall** after fine-tuning (e.g., `sakurauchi_riko`, `tokoyami_towa`).

## Conclusion
The two-stage training strategy was highly effective. While the linear probe provided a stable starting point, fine-tuning was essential to reach production-ready accuracy levels (~77%) for this fine-grained classification task.
