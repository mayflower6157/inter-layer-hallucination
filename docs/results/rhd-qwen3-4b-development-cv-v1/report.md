# RHD domain-stratified development cross-validation

Four folds over 57 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.620 | 0.606 | 0.591 | 0.633 ± 0.060 |
| rhd_only | 0.665 | 0.575 | 0.619 | 0.661 ± 0.098 |
| rhd_plus_length_step | 0.642 | 0.557 | 0.581 | 0.693 ± 0.123 |
