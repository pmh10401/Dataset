# Exp438 third-party provenance

This document freezes the external research sources inspected before Exp438
product-code implementation. Exp438 remains a project-owned implementation and
does not vendor third-party repositories.

## ASPE reference semantics

- Repository: <https://github.com/Brioal/ASPE>
- Pinned commit: `f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89`
- Reference symbol: `entropy/aspe.m::aspe`
- Supporting symbols: `util/ordinalIndex.m::ordinalIndex` and
  `util/shannonEntropy.m::shannonEntropy`
- Detected license: Apache-2.0
- Exp438 policy: implement a source-compatible Python port with attribution and
  independent literal golden vectors; do not copy MATLAB source line-for-line.
- Explicit project extension: flat input or total finite weight `<=1e-12`
  returns `0.0` with `flat_or_zero_weight=true` instead of propagating `NaN`.

## Adaptive Fourier Decomposition reference

- Repository:
  <https://github.com/pikipity/Toolbox-for-Adaptive-Fourier-Decomposition>
- Pinned commit: `2b11bcc0eacd30f979ef4d6b24715c7fec327b02`
- Documented license: CC BY-NC-ND 4.0
- Exp438 policy: read-only research reference only. Do not copy, modify, import,
  vendor, or use it as a runtime dependency. Exp438's FFT cutoff split is a
  transparent project-owned approximation and is not called AFD or MAFD.

## Excluded namesake PhysDiff repository

- Repository: <https://github.com/VUT-HFUT/PhysDiff>
- Inspected commit: `c4a17679cc3c81d0eea7e5ec07cccada1272bad7`
- Repository description: AAAI 2025 Physiology-based Dynamicity Disentangled
  Diffusion Model for Remote Physiological Measurement
- Relevant file: `model/physdiff.py`
- Detected license: none
- Exp438 policy: excluded. It consumes camera-derived spatial-temporal maps and
  fixed-length blood-volume-pulse signals and hard-codes CUDA. It is not the
  NeurIPS 2025 multivariate time-series anomaly-detection implementation and no
  code is copied from it.

## PhysDiff research paper

- Paper: Long Li, Wencheng Zhang, Shi Yuan, Hongle Guo, and Wanghu Chen,
  "PhysDiff: A Physically-Guided Diffusion Model for Multivariate Time Series
  Anomaly Detection," NeurIPS 2025.
- Paper URL:
  <https://papers.nips.cc/paper_files/paper/2025/file/980ea04d23d1f6908964eba2a74afe45-Paper-Conference.pdf>
- Paper code link: `https://anonymous.4open.science/r/PhysDiff-4726`
- Inspection result: the anonymous repository was not publicly retrievable at
  design time and no verified replacement repository was found.
- Exp438 policy: use the published high/low decomposition and amplitude-sensitive
  complexity ideas only as research inspiration. Do not claim PhysDiff code or
  result compatibility.
