# AI-Driven Equipment Intelligence: Equipment Anomaly Detection Using Multi-Source Evidence and Stability-Gated Candidate Selection

## Abstract

Equipment anomaly detection is often performed in settings where normal
operating data are abundant but labeled failures are scarce. We present an
AI-driven equipment intelligence method that combines multiple one-class
evidence sources, low-frequency candidate expansion, and deterministic
training-omission stability selection. The method assigns a continuous anomaly
score while exactly preserving its frozen binary alert route, enabling both
threshold-based and ranking-based evaluation. We retrospectively evaluated the
method on 1,117 public univariate time-series datasets containing 195,900 test
instances and compared it with six published one-class baselines. At 2,276
alerts, the proposed method achieved a micro F1 of 0.6306, a macro F1 of 0.6229,
an average precision of 0.6741, and an AUROC of 0.9611. Under a matched-alert
budget, its micro and macro F1 exceeded those of the strongest baseline, RBF
One-Class SVM, by 0.1221 and 0.1581, respectively. On the common support of 922
datasets completed by all seven methods, the proposed method achieved an
average precision of 0.6697 and an AUROC of 0.9642. Paired dataset bootstrap
intervals for the macro-F1 difference between every baseline and the proposed
method were entirely below zero. These results support the method as a
retrospective research candidate; they do not establish calibrated failure
probabilities or prospective performance on semiconductor manufacturing
equipment.

**Keywords:** equipment anomaly detection; one-class learning; time series;
continuous anomaly score; stability selection; predictive maintenance
