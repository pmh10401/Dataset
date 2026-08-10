"""Closed, reviewed authority literals for the Exp445 predecessor verifier."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


EXP444_REVIEWED_COMMIT_ANTIGRAVITY = "f4ae53c46422a46cc880d5d204f68a6a3971e92d"
EXP444_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp444_train_only_stability_precision_gate_antigravity_20260810"
)
EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY = (
    "attempts_antigravity",
    "attempt_0000_antigravity",
)
SOURCE_DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")

EXP444_TERMINAL_SHA256_ANTIGRAVITY = MappingProxyType({
    "checkpoint/context_antigravity.json": "dcbc3755bf94d771c3f506746785e544d88940cc36bb75c0da72e99506ec9926",
    "checkpoint/state_antigravity.json": "bcb524292f3ce2332a06b05fa6205bafce268f0d6d1ca6ff5622eea0e72be8a4",
    "progress_antigravity.json": "bba24f851ede8aa498a5d2f94d4696792c81754290a48b8138e9c9ecd704f6ae",
    "pid_antigravity.json": "e93fdbb36dd0e3339cf0900df22a14ee7627f74c25c6af31018adf8e0ca3397e",
    "events_antigravity.jsonl": "e58780ef8b127f7ad103278ac47494176b56c4653d2498405c8e125cd69042ff",
    "text_antigravity.log": "8876d3db294294ecbb3c60ab170ed44776b61bbe9bcf3735bf32f27c156feef5",
})

EXP444_RECORD_SHA256_PREFIX_ANTIGRAVITY = (
    "adb31fd3f1d92250e52b47a854f12d1bfd7268e80a9cbdab4db576029d1aad88",
    "e18aa085fb468b253fb56e1e8124cfb3b60590a97c7ce4f76041867fcfe22088",
    "48bb6a1cf5b3b420214278faea165322eaf256854de392fc70866b2b64c53437",
    "6b21145fb5af9cd2e70cdf267729d3ec48aa471c3b7f84ca6b9dca965457c6da",
    "f98fed0dee32bee7f18a39891acf2bcf4cab498b32071082acadcd644c12adfa",
    "5bec1c44406900caf2c624f5bb556855373fc329ed2959c9792b1ea55b03c57c",
    "befcca68ab198770dfcb4131d40486976538eedfa2cbdbde0e2fbaf341baeeba",
    "2e20f6cc3d0cfa45f0fc339f2c31039826af66a226947a64599da52e3e816c0e",
    "e28979eca5aea590165bef4a50362b8b03801949e23ebda107b056fe7ab848e4",
    "ffc4cdf27bdf8d44e0154ca74c6dcf38e767643b1d7a0f5ddce58b4fc60c6d39",
    "74afc6dc81e71a3a6ed75100c2865da18256eaa9ec666cd7c781f578c8a44329",
    "849c451a8fe17f879af3a827660ce01e4d06f05cf33da66cf3a1f310ed9381f1",
    "3bab641833280e2037bbe5f482ee1112e11b70a863fae0ef5f467538862bdb49",
    "b540585682105c84f0d45b6e0f5e3e41d806214db903622364a3495e7e3c603b",
    "c3a6cfb4d5693820571b80f6b3c45b303fc3aaaaa8fc2245d207efad9553fda5",
    "dc090fc61916a8b392e3d17bf0e4beb267fd1b4089ace1a5f59abfb35331ec55",
    "c5da3b239cdcf7af1321933e84aba03ead68abeaf9275ca9e07dcecd3118e3dc",
    "6884004741267c756b19dfe879a0308d46891dd9c4b11cf09c34c844342554f6",
    "d125e9d379c0f24b02081c1752a0fdca7c09e65781d5a6c259d1c57cfe82d9f0",
    "ef8d44a60ac0fa0c3c44cd57367f4145f3f5b17b8b2e50baceb2bddb6a9f88c2",
    "6fe44d9a89400734836a0b16a1c0b2f94ccd25bff939b2ed5e9e7f5e6a2302a9",
    "8661cfe2445af6d4a84182d0c29d13424a8eddfb2c1d1d3f6ed264571ece320c",
    "dbb81a49f6e330c7c8e0016d6f8282f61b89a90f28475acb8f69c69c19c3b611",
    "ff9fed1ed083eec2a1225f3a19423041bdb639420a17738933c6e1693ec98059",
    "f64da0f6380032bcadd68d2458ed0db49390c5c813c4ca1817a40c249e81ba00",
    "0406feb516e32e00e3ee1a3a1801ca4743579b4c42e7e1af0732e2fca9d0d447",
    "74737b9ce40714732107d9933927561dc0accaf2d89b04bb868f5c821b05ce2f",
    "e0f63e1b5f2c3200f27887ca55fe9609d2bcefb5223a6fd5251088f69a850ffc",
    "38c6815c6f22763d3cd0e003cba6e7d5b8d2534f236a8b67a5aa4409cb247827",
    "ca9dc14a1528cab387a7cd6ab067e7b47bfe4b6cee8ed0bd742b87c352d73420",
)
EXP444_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY = "af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034"

EXP444_CONTEXT_ANTIGRAVITY = MappingProxyType({
    "environment_lock_sha256": "8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94",
    "experiment_id": "Exp444",
    "inner_threads": 1,
    "outer_workers": 11,
    "reviewed_commit": EXP444_REVIEWED_COMMIT_ANTIGRAVITY,
    "run_identity": "exp444-train-only-stability-precision-gate-antigravity-20260810",
    "scientific_config_sha256": "8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a",
    "total_units": 30,
})
