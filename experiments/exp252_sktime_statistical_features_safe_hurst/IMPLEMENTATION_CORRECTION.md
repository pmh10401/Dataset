# Exp252 implementation correction

Exp251 stopped during cache generation before prediction or evaluation because
the official sktime R/S Hurst transformer raised an expected numerical failure
for degenerate individual series. The failed Exp251 output and logs remain
unchanged.

Exp252 preserves the five preregistered feature variants, seed, KNN local-gap
scoring, TRAIN OOF calibration, alpha grid, and frozen Exp236 lanes. It changes
only failure handling before evaluation:

- Hurst remains the official sktime panel transformer, wrapped so each
  vectorized series can recover independently from the known invalid R/S case.
- The known invalid R/S result becomes three NaN feature values for that series.
- Non-finite values are replaced using TRAIN-only finite medians.
- A feature column with no finite TRAIN values is disabled for both TRAIN and
  TEST so a TEST-only finite value cannot create an artificial anomaly signal.
- Unexpected Hurst or transformer exceptions still fail the dataset.

This is an implementation correction discovered before TEST metrics existed,
not a TEST-performance-driven model adjustment. Exp252 remains a retrospective
score-source qualification experiment and does not change autonomous alerts.
