# RHD domain-stratified development cross-validation

Four folds over 47 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.641 | 0.617 | 0.598 | 0.679 ± 0.085 |
| rhd_only | 0.665 | 0.663 | 0.660 | 0.674 ± 0.024 |
| rhd_plus_length_step | 0.683 | 0.683 | 0.680 | 0.693 ± 0.017 |
