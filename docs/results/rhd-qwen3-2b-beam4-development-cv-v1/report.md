# RHD domain-stratified development cross-validation

Four folds over 48 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.477 | 0.549 | 0.516 | 0.498 ± 0.043 |
| rhd_only | 0.692 | 0.625 | 0.666 | 0.654 ± 0.182 |
| rhd_plus_length_step | 0.670 | 0.600 | 0.719 | 0.631 ± 0.171 |
