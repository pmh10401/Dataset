# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


service_root = Path(SPEC).resolve().parent.parent
analysis = Analysis(
    [str(service_root / "service_launcher.py")],
    pathex=[str(service_root.parents[1])],
    binaries=[],
    datas=[
        (str(service_root / "web"), "services/time_series_anomaly_diagnosis/web"),
        (
            str(service_root / "resources"),
            "services/time_series_anomaly_diagnosis/resources",
        ),
    ],
    hiddenimports=[
        "aeon.transformations.collection.convolution_based._minirocket",
        "aeon.transformations.collection.convolution_based._multirocket",
        "aeon.transformations.collection.convolution_based._hydra",
    ],
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="TimeSeriesAnomalyDiagnosis",
    console=True,
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    name="TimeSeriesAnomalyDiagnosis",
)
if sys.platform == "darwin":
    app = BUNDLE(
        distribution,
        name="TimeSeriesAnomalyDiagnosis.app",
        bundle_identifier="local.research.time-series-anomaly-diagnosis",
    )
