# RHD domain-stratified development cross-validation

Four folds over 59 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.476 | 0.522 | 0.491 | 0.485 ± 0.072 |
| rhd_only | 0.759 | 0.713 | 0.678 | 0.784 ± 0.117 |
| rhd_plus_length_step | 0.722 | 0.695 | 0.711 | 0.751 ± 0.136 |
