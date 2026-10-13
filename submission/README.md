# Submission artifacts

Generated prediction rows and ZIP archives are intentionally excluded from the public repository. They can be rebuilt with `prediction.py` and the track-specific scripts after the dataset and checkpoints are available.

The expected CSV columns are `ID`, `Task1`, `Task2`, and `Task1&2`. Run `scripts/check_compliance.py` before packaging a submission.

- [Codabench competition](https://www.codabench.org/competitions/5372/)
- Retained displayed Task 1&2 score: **0.9817 AUC**
- Screenshot: [`../evidence/codabench_result.png`](../evidence/codabench_result.png)
