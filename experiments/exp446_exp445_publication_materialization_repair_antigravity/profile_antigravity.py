"""Closed literals for authenticating Exp445's immutable failed publication."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


EXP445_REVIEWED_COMMIT_ANTIGRAVITY = "2fdd4fbd08bc5a7fd3d2ddad77942422fb3f081d"
EXP445_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
)
EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY = ("attempts_antigravity", "attempt_0000_antigravity")
SOURCE_DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")

EXP445_TERMINAL_SHA256_ANTIGRAVITY = MappingProxyType({
    "checkpoint/context_antigravity.json": "e7ea9465327ae4b71dfe06c8a83f7c3eb77e4d1445b0598da3de6afa4a5a53c3",
    "checkpoint/state_antigravity.json": "b8a4d6f1bb1cb30680f4244c7e025c43367a41ed90518b6e49dc4d09e890ede4",
    "progress_antigravity.json": "8d34590450e2fcd2e21dcc6f03454ac7d4cd9fc1d15e1ecb25fc858c20615eb5",
    "pid_antigravity.json": "e0f2d8ecc65008decdb8ba8292c7204c377bebeccb8ab564bb225bae307b3fbf",
    "publication_failure_antigravity.json": "9c58eddaa152bde221a5b7673be3d413204ff3dfe9fd2b4262e6d9065b2c6db0",
    "events_antigravity.jsonl": "b23b1bbf5ae9811e06319a46a67d07d3e4d040c87be798335f04c559047a3e60",
    "text_antigravity.log": "d631391057a3fcfecdc46fc9be8a15148fa214ecbfb7e2c1fd5cff21009141e9",
})

EXP445_RECORD_SHA256_PREFIX_ANTIGRAVITY = (
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
EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY = "af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034"

EXP445_CONTEXT_ANTIGRAVITY = MappingProxyType({
    "environment_lock_sha256": "8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94",
    "experiment_id": "Exp445",
    "inner_threads": 1,
    "outer_workers": 11,
    "preregistration_sha256": "20d73ed9cebfbd4cd40d4d3354197907483ac9092a8392e93e9bf1b44027a37a",
    "reviewed_commit": EXP445_REVIEWED_COMMIT_ANTIGRAVITY,
    "reviewed_design_sha256": "099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d",
    "run_identity": "exp445-exp444-publication-source-authority-repair-antigravity-20260810",
    "schema": "exp445",
    "scientific_config_sha256": "8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a",
    "total_units": 30,
})
EXP445_PUBLICATION_FAILURE_ANTIGRAVITY = MappingProxyType({
    "error": MappingProxyType({"message": "Exp444 original-control replay is invalid", "type": "ValueError"}),
    "schema": "exp445.publication-failure.v1",
    "stage": "publication",
})
