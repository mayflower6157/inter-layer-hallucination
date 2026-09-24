# RHD domain-stratified development cross-validation

Four folds over 217 scored development traces. Scaling, feature direction, weights, and threshold are fitted using each training fold only.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |
|---|---:|---:|---:|---:|
| length_step_baseline | 0.472 | 0.437 | 0.516 | 0.528 ± 0.104 |
| rhd_only | 0.458 | 0.392 | 0.565 | 0.512 ± 0.121 |
| rhd_plus_length_step | 0.438 | 0.388 | 0.514 | 0.474 ± 0.095 |
