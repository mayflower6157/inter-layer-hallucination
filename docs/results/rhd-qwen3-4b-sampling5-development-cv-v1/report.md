# RHD domain-stratified development cross-validation

Four folds over 294 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.576 | 0.593 | 0.531 | 0.611 ± 0.072 |
| rhd_only | 0.685 | 0.694 | 0.613 | 0.709 ± 0.124 |
| rhd_plus_length_step | 0.677 | 0.679 | 0.630 | 0.694 ± 0.105 |
