#!/usr/bin/env python3
"""Local-only frozen-subset runtime benchmark for Exp419."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
import re
import resource
import subprocess
import sys
from time import monotonic, process_time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    assert_label_free_sql_contract,
    _verified_database_connection,
    freeze_lane_inputs,
    freeze_run_context,
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources
from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    RuntimeBenchmark,
    RuntimeTrial,
    WorkerBootstrap,
    WorkerSpec,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import (
    _context_sha256,
    select_runtime,
)
from experiments.exp419_deterministic_cpu_parallel.planner import (
    build_shard_plan,
    estimate_dataset_cost,
    select_benchmark_datasets,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    TrainLengthGroupMeta,
)
from experiments.exp419_deterministic_cpu_parallel.workers import spawn_workers
from scratch.run_exp417_causal_multihorizon_shadow import (
    _backend_audit_path,
    _provenance,
)


BENCHMARK_FILE = "EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only Exp419 label-free benchmark for exactly "
            "1, 2, 4, and 6 CPU workers."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--exp291-prediction", type=Path, required=True)
    parser.add_argument("--exp291-manifest", type=Path, required=True)
    parser.add_argument("--exp292-prediction", type=Path, required=True)
    parser.add_argument("--exp292-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser


def require_clean_git(repository: Path = REPO_ROOT) -> str:
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("Exp419 benchmark requires a clean Git worktree")
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise ValueError("Exp419 benchmark could not verify Git HEAD")
    return head


def _static_string(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, constants)
        right = _static_string(node.right, constants)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            target = value.value if isinstance(value, ast.FormattedValue) else value
            part = _static_string(target, constants)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _contains_projected_label(query: str) -> bool:
    without_comments = re.sub(
        r"/\*.*?\*/|--[^\r\n]*", " ", query, flags=re.DOTALL
    )
    normalized = " ".join(without_comments.split())
    normalized = normalized.translate(
        str.maketrans({"\"": "", "`": "", "[": "", "]": ""})
    )
    for match in re.finditer(
        r"\bselect\b(.*?)\bfrom\b",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        if re.search(
            _LABEL_SQL_PROJECTION_DENYLIST,
            match.group(1),
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _looks_like_label_artifact(value: str) -> bool:
    lowered = value.casefold()
    artifact_shape = any(character in lowered for character in ("/", "\\", ".", "_", "-"))
    return artifact_shape and bool(
        re.search(
            _LABEL_LITERAL_DENYLIST[2],
            lowered,
        )
        or _LABEL_LITERAL_DENYLIST[3] in lowered
    )


_SENSITIVE_OPERATION_ALLOWLIST = tuple(
    (basename, scope, "scope-sensitive-multiset", nodes, scope_sha, count)
    for line in """
benchmark_exp419_deterministic_cpu_parallel.py|<module>|b6ab6f113ad3c6756570a3a1d2ad5174662a877246be11acb47c47ee2fd98e16|89fa91befd69016479b599609bba0c1c236c311ad7b0f0b0d3858683c07889f0|7
benchmark_exp419_deterministic_cpu_parallel.py|_load_train_metadata_costs|38ec405bfddc715ccf426332c6f92743df3b31fb9363ae0c41c88c5c05032116|de5b9ffc7f10fa12c27759f3bfa854c497299c41776cd3bb51d830abc79186d2|2
benchmark_exp419_deterministic_cpu_parallel.py|_run_trials|0df827eeefb81bf38aa2dacdbc548f299917e3af72cdd6089e7f083479449060|87c604ee5d7986681f5fcba1ac38cfd27032e230f561b20e79a011bcd4b0d733|1
benchmark_exp419_deterministic_cpu_parallel.py|assert_exp419_label_isolation|447d6541a0c0ad9d4eb84d0f9e5efb17636963167db1374bc66dd1a92fc45497|bcc287d6f7601f4ef481167f2f1d430e78e4928f27c190511700ab8e32389f1a|2
benchmark_exp419_deterministic_cpu_parallel.py|build_parser|d720b6687ca0d2e883c3ebd12c3a3f26076f257c472899ca08f0ba6498873eed|83522f1048ab39bb84850043836387c71bb649bf17ecb0a0f26421473458485c|6
benchmark_exp419_deterministic_cpu_parallel.py|prepare_trial_roots|ed1df775f83a5001f667ed67bd37366a43589da063ef7d7b28787a26516f6c12|f81616bf266ef4c27b3c434cfd7b82f6215a0ce2a6a8c49181f6585765b119c2|4
benchmark_exp419_deterministic_cpu_parallel.py|propose_sensitive_inventory|c67917dfb424c49e2aa07696c1a040d6a319193db6d257b371788e34a194c179|ee5dab760c6f2d252db80b3d42d4d1a698b22a887a0b31ed1b65b2d635610179|1
benchmark_exp419_deterministic_cpu_parallel.py|require_clean_git|5f3939b7f381473326ceeb4da4cce8951819b26af0d341d6345e1c0fb777bfc6|7d0ab73d742b2ebac8afaecbde92b2132a2506035016204864982eaf38b1544a|1
benchmark_exp419_deterministic_cpu_parallel.py|write|b3799d3b3eaf180652e3dd7911e201c7b8fd57f6015044feedb5bd6ade48bb94|2b4c492b84bc8362e88e67797f2c9281062d28b77e3fc742fc1ebe9521573bbb|1
benchmark_exp419_deterministic_cpu_parallel.py|write_benchmark_evidence|a1df4382cd8b93e32ed1e772b5acdbb394da4caa3bf1d56392a6727eb9a1f2bc|893945d988368f7b6315a2216230b4b3b3c03cc6631e3b188343cfcc7ef73339|6
config.py|validate|d1af20d490bec84d012ccf14bd8018a13572a72e20439ba62c701b6f6fc580ea|7996524cc33e71502082341a63cd975786570f3804c45528175d9f776e51998c|4
contracts.py|<module>|4fe68104df8e95eda06e560a2bedb8886a0e9a755f370eaa501a7d3abe632b3c|8d6ceab27f1ddcb9975600a6a765d1d4a6f485f770df8aa71c8fac4b4df28ec9|5
contracts.py|__post_init__|b78f9041960368efab0a255f0c78a04c7426cc3aaf2536d5196e823933f85f62|ab27ab41fde4daf50f9b219e797f75241c648dde3c51440e1c659ab118f0e9d2|1
contracts.py|_path|c84e111c42913bf7451d53027a5b1cfd2fb46473b96c37b3a8b91d36c70421dc|33b9152d9839668769e0eceb6b0e879075d73183e399930eeaed874b729908b4|2
coordinator.py|<module>|b31c56b868a86b16d10a7012c7b7576ec4cd3e470a754f9bb2de4158dc33dad4|71aa20f4cfe5624326ff6b358a8f503dad5f8bccdac8d25062b2a9b640cedede|7
coordinator.py|__init__|8471a99e1e9465d3e7e94b03e8232d359997ddb6b662f61facc1d65b24e84fc5|39a4fbb966fd0a6c34e954e60b35781fef8f0220ca86d5794d998b76c6d816d7|7
coordinator.py|_freeze_plan_binding|ba8713e43be27e336d886c89ba1ccc34fc379051845bfca9e067b63d43e22c53|da7640731e8f8bb6395465974d7c40b6decef99b293179b131fb339c36c30711|4
coordinator.py|_interrupted_attempt_record|debd60fcc44c493f6338fec54fd78cdd32807b424a972257c90d9ca24abbe6c6|0430516c21f6d2e63b7738763610ee237ef125af818a93056becbc66934a2b0b|1
coordinator.py|_nonempty_checkpoint_root|bb428c04f274dcb81bcc60bac0f909456cbd08a077ad135ccc49873ce99cac6f|d12e630b0e07e80237afc570404537654619ae873ac6f1f509fcdcf5891bccca|3
coordinator.py|_open_checkpoint_store|5f3939b7f381473326ceeb4da4cce8951819b26af0d341d6345e1c0fb777bfc6|a0ca2153c37239169a172153940c3670c346b85e1884c00c6b5f5e37104a636a|1
coordinator.py|_open_checkpoint_store_checked|e04f066b031ad5ce397d25277977dee8c05b7f4f50fd806cdac9e2c13f91dc8e|f389c63fee9385d155bcc78ee4ae161035a38d1da26e09e39277212e975a2906|1
coordinator.py|_provenance_payload|e74827eabb20022f005ecdf77679f2ccdeaec052fa9a4f7d6bb95bdf79f6753e|1144900c58e1199b0ce7183b56671390644d7fc0b8509b0c5261666fc3eceeef|1
coordinator.py|_publish_binding_no_replace|ff3acda069ce80ef536e5159925d5eef945d29be693e955e048c57a1f7ad2e75|7d7273bef9b8b58bd9ddc8a11a6399369906c4b51116e84bc8c3e51806bda9d7|3
coordinator.py|_read_optional_root_file|5561e4f1c4c4d830f4c78e9269f3bf7fd9a0607a50309a4227bd499370086d4a|fd1881ed96e9831dab36a2f2a5eb6fc33d80eb878ac501c93f768137b8043c3d|1
coordinator.py|_replace_runtime_evidence|eca9a96416f171e51d433cb100cc649b9b682401d7a07a0bc46109e85f5939b1|745497b5fb90f2bc301c1ca03bac10d5074b2c3fa5204ec914ac8985f1c462c6|3
coordinator.py|_running_attempt_record|b2a4b532a000a16271a990faee1b053762db321861684cec163e56f494ec7b73|4fa881793bfa39292d8bf197976e4442e4696e49b46d97173cab22b19eee237f|4
coordinator.py|_validate_and_recover_attempts|f7664a85b0a12e105e5af1795bd2ecf323c49cf5f6a80e915edb750f2f6ddb68|616b93ba668b87539598099771d1d8e97419dd9076dd366e354de934b2d97a55|1
coordinator.py|_validate_partial_worker_stores|48bf42d671bc7658bf90758b09fcea99b7bb213a89668778b4b083e18915f5a0|ea5cbfb6018a2c79f66348d908963e5ba8e6a9dde501b062287cc693528de6b9|2
coordinator.py|_validated_evidence_digests|07ca3049dfb8e9236539d1085212102094bc3a348a1ff2bcc30d733eda6739e1|fe96a16da8b9b7b3666e70032de0ffb714f5309b4aa6cd31698ed4dbb3d53c58|4
coordinator.py|_validated_partial_store_state|5bb739a8bd80a5f2d640f3ef2ec642ff85ce2c4d039ca2e8d80e8385efd09e14|ef414f0f5b773df4f98a858ebc939d6f33d0b913ef93bec2324a344335223bc3|1
coordinator.py|merge_worker_checkpoints|05cc6369ad88979ccb7bf8c423e0bd3d537c4ada82dfdef227e29ded807464f2|58ed4afdeb857d56b8bedeb13009276a9b253f3a78d258934cdccdf76d4d22cc|4
coordinator.py|verify|ef0ef8daad24dbee87dca68c88463546b05bb978bf7a40024a2a6d382ab33a05|7d02aaf6ae10b62fd9da81eb8aa0f8b36fef357cba9556eb554531b809cd7b8b|2
coordinator.py|verify_checkpoint_path|1c2dfaa151ae8d8d34e7d2fc9d22e7b014b12fb84432888e1180d81698efb899|e88b30a3a6075b9f53c160928b53ae633ebb7b1566579e16a8f2975e1522ad19|3
publication.py|<module>|529f072b5a7d22ae3fca14c213d2eb0cc42555b314ce62c2e9ab46b7f0493693|be934048aa3cab1c54db32e8ca21472e2b2c38f689008502c36cabc0b5659d56|1
publication.py|publish_exp419|248c24d9cb06641cb8afbe18be7d680f6cdca017621cae2499f418ed8e569a9a|def0bbd59f2aeebf80263a5082f49843eb72a39eddfd6c1f94d7075b2073b5ed|4
reporting.py|<module>|bc0f68647dbd8a357fd3430aacb8e57e969065ed63d361d9a38646222f42c596|504235cb9c0f93ac4e381923888e8c07004a2bec4c9312a315008a64625e1924|1
reporting.py|_exp419_source_digests|9c57d4934b6f6da39c5304f3a01dc454a6431530b9e04ff9ca2870a046b4a56b|3d99a7728a18d56c693e4ebacd041a141257b6b0538183b3399478f79cbabd6b|3
reporting.py|_source_digest|6186a2f62c9d0ebec0bea8b4e98817d3eb751c09f30eba8e00e5a298f90f6b28|9eebcd0f58d02b4db994faeb03c9b8d27e8a4c9746f25fc554a8d201c6f0c7b4|3
run_exp419_deterministic_cpu_parallel.py|<module>|effa9365cf3c1039e7022c2f9e1651aee84043caf8e7ed9c6cf5120a21062ac1|a5a4f7a752c9eb41b206b9be48b5bf9f9f19f38a4431fdaf9b8930c44b47f47b|4
run_exp419_deterministic_cpu_parallel.py|_identity|8107e81b9963cbc8b59e104dd30ba746fcf9a8e038c2bcae05fc5aec8ac6775e|c67cbb868fd9faa0992368faa6e3e0848ce556b716e124e6fe390bfd66ff4f37|2
run_exp419_deterministic_cpu_parallel.py|_read_json_regular|8440dfa0386fc0acb9a6e4c274666c41f7921d6f749a85572855377a0ca4cf2d|80cc45b3b148fdc79e5bc961e90c2afefe3fdfc846e50a8adc69929bb7f1688d|3
run_exp419_deterministic_cpu_parallel.py|build_parser|6199d448944077f77c169bb96d6afd0e9f669179d4e920fdd11bcf683953fc59|389721a33218244fc2fabc086c81dd30f9e22deb0b8ceab4daf980dfd5ddf08c|8
run_exp419_deterministic_cpu_parallel.py|require_disjoint_roots|b353cb70b8098d07ec088eda9998560c3a89f5dd9b596d8642a4d2e89d436974|a1f403fbc5b2b5262ddd1620e4b838bff66f9f9c92bdd80b0f070f7d85700806|6
run_exp419_deterministic_cpu_parallel.py|verify_benchmark_evidence|e25a8952d254a97f1e72cb2e68ed29dbffd4ac675eacfdd6f1bd1d17756d6f08|282319147ad01a4f6f0aedad07cd3b5aa25edb40ef2d68b9b6a1cc7febaec04f|4
workers.py|<module>|9d355b6b984b67f64bc881e3d5e6d92b7eb1bb82631184cfec579a6f4d5b993e|a84728fa409f1abff1e168b9c26fa085b51943e8d8870ef9c0c762b2a29b2d6a|2
workers.py|_bootstrap_from_payload|a5ebe40a9335a5895fdff73a4b171807897b4bb168cb1176ff94f60dfe66d0b9|0a0ecbebd07889efbf0da832c8b6067562bdae58a81d2f84341c4fa88b9063ef|6
workers.py|_casefolded_path_parts|2322396516cd7afc4a465889c004e8c97368cc196269cbc3e4ad77e9486a7c58|ce3262196e911d09067aa0d9b3aac65060a40f382645d93e220f8d944f605bc5|2
workers.py|_filesystem_identity|7407c6dd5c879cc88319735dd1fff46ca766ce5ea6ad971bf64550c8fabbb6ed|e8f4d0a3d7d323db7b03cf7e4bc3b5d0266adbe87d17cc2bbf7cbb15c2f67537|2
workers.py|_validate_context_provenance|bf31d244f74dfab3b694ce5f633a93e7f01ff61da6510a57cbc3ba61d63113df|1fc2f698cc104dbddce5fe57673050fabe3d5810036db976abd4ab23625abe05|1
workers.py|_validated_specs|57ab8347aa0bddad408f29b3579a806b91cf3161377bb29b6878e104e395d631|a56b90bd64b6fe6cd0c5377b90d551233621fba2be75c67767b93ab7cd60b34e|1
""".strip().splitlines()
    for basename, scope, nodes, scope_sha, count in (line.split("|"),)
)

_APPROVED_SQL_SHA256 = (
    "f94aadff07a9626329efaf06331740d80c62cf5799a928bc08d690b6ad31b146",
)

_LABEL_LITERAL_DENYLIST = (
    r"(?:load|read|fetch|get).*labels?",
    r"(?:^|\.)labels?(?:\.|$)",
    r"(?:^|[/\\._-])(?:frozen[/\\._-]*)?(?:test[/\\._-]*)?"
    r"labels?(?:$|[/\\._-])",
    "labels_blob",
)

_LABEL_SQL_PROJECTION_DENYLIST = (
    r"\b(?:[a-z_][a-z0-9_]*\.)?(?:label|labels_blob)\b"
)

_LABEL_API_ALLOWLIST = (
    ".assert_label_free_sql_contract",
    "assert_label_free_sql_contract",
)

_INVENTORY_DATA_ASSIGNMENT_ALLOWLIST = (
    "_APPROVED_SQL_SHA256",
    "_FORBIDDEN_KEYS",
    "_INVENTORY_DATA_ASSIGNMENT_ALLOWLIST",
    "_LABEL_API_ALLOWLIST",
    "_LABEL_LITERAL_DENYLIST",
    "_LABEL_SQL_PROJECTION_DENYLIST",
    "_SENSITIVE_OPERATION_ALLOWLIST",
)

_SENSITIVE_CALL_NAMES = (
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "importlib.import_module",
    "locals",
    "open",
    "operator.attrgetter",
    "operator.methodcaller",
    "os.lstat",
    "os.open",
    "os.stat",
    "path",
    "vars",
)

_SENSITIVE_METHOD_NAMES = (
    "execute",
    "executemany",
    "executescript",
    "glob",
    "lstat",
    "open",
    "read_bytes",
    "read_text",
    "resolve",
    "stat",
)

_SENSITIVE_REFERENCE_NAMES = (
    "Path",
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "vars",
)


def _inventory_data_nodes(tree: ast.AST) -> set[ast.AST]:
    nodes: set[ast.AST] = set()
    for assignment in ast.walk(tree):
        target = _assignment_name(assignment)
        value = _assignment_value(assignment)
        if (
            target in _INVENTORY_DATA_ASSIGNMENT_ALLOWLIST
            and value is not None
        ):
            nodes.update(ast.walk(value))
    return nodes


def _normalized_scope(scope: ast.AST) -> str:
    reparsed = ast.fix_missing_locations(ast.parse(ast.unparse(scope)))
    scope = reparsed if isinstance(scope, ast.Module) else reparsed.body[0]
    if isinstance(scope, ast.Module):
        assignments = [
            node
            for node in scope.body
            if _assignment_name(node) == "_SENSITIVE_OPERATION_ALLOWLIST"
        ]
        if len(assignments) > 1:
            raise ValueError("label isolation allowlist assignment is not unique")
        if assignments:
            assignments[0].value = ast.Tuple(elts=[], ctx=ast.Load())
    return ast.dump(scope, include_attributes=False)


def _enclosing_scope(
    node: ast.AST,
    tree: ast.Module,
    parents: Mapping[ast.AST, ast.AST],
) -> tuple[str, ast.AST]:
    scopes: list[str] = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(current.name)
            return ".".join(reversed(scopes)), current
        if isinstance(current, ast.Lambda):
            return "<lambda>", current
        if isinstance(current, ast.ClassDef):
            scopes.append(current.name)
    return "<module>", tree


def _path_from_scope(scope: ast.AST, target: ast.AST) -> str:
    def visit(node: ast.AST, path: tuple[str, ...]) -> tuple[str, ...] | None:
        if node is target:
            return path
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                found = visit(value, path + (field,))
                if found is not None:
                    return found
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        found = visit(item, path + (f"{field}[{index}]",))
                        if found is not None:
                            return found
        return None

    result = visit(scope, ())
    if result is None:
        raise ValueError("label isolation could not locate sensitive node")
    return "/".join(result) or "<scope>"


def _is_loader_like(value: str) -> bool:
    return bool(re.search(_LABEL_LITERAL_DENYLIST[0], value.casefold()))


def _sensitive_kind(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> str | None:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or "", *(alias.name for alias in node.names)]
        )
        if any(
            name in {"importlib", "os", "pathlib", "operator", "sqlite3"}
            or _is_loader_like(name)
            for name in names
        ):
            return "import"
        return None
    if isinstance(node, ast.Call):
        called = _call_name(node.func).casefold()
        last = called.rsplit(".", 1)[-1]
        if (
            called in _SENSITIVE_CALL_NAMES
            or last in _SENSITIVE_METHOD_NAMES
            or _is_loader_like(called)
        ):
            return f"call:{called or '<dynamic>'}"
        return None
    parent = parents.get(node)
    is_direct_callee = isinstance(parent, ast.Call) and parent.func is node
    if isinstance(node, ast.Name) and not is_direct_callee:
        if node.id in _SENSITIVE_REFERENCE_NAMES:
            return f"reference:{node.id}"
    if isinstance(node, ast.Attribute) and not is_direct_callee:
        called = _call_name(node).casefold()
        if (
            node.attr in _SENSITIVE_METHOD_NAMES
            or node.attr == "__dict__"
            or called in {"importlib.import_module", "operator.attrgetter",
                          "operator.methodcaller", "os.open", "os.stat", "os.lstat"}
            or _is_loader_like(called)
        ):
            return f"reference:{called}"
    return None


def _static_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            target = _assignment_name(assignment)
            value_node = _assignment_value(assignment)
            if target is None or value_node is None:
                continue
            value = _static_string(value_node, constants)
            if value is not None and constants.get(target) != value:
                constants[target] = value
                changed = True
        if not changed:
            break
    return constants


def _assert_direct_label_and_sql_denials(
    name: str,
    tree: ast.Module,
    constants: Mapping[str, str],
    data_nodes: set[ast.AST],
) -> None:
    for node in ast.walk(tree):
        if node in data_nodes:
            continue
        if isinstance(node, ast.ImportFrom):
            qualified = [
                f"{node.module or ''}.{alias.name}".casefold()
                for alias in node.names
            ]
            if any(
                value != _LABEL_API_ALLOWLIST[0]
                and (
                    _is_loader_like(value)
                    or re.search(_LABEL_LITERAL_DENYLIST[1], value)
                )
                for value in qualified
            ):
                raise ValueError(f"label loader import in {name}")
        elif isinstance(node, ast.Import):
            if any(
                re.search(
                    _LABEL_LITERAL_DENYLIST[1], alias.name.casefold()
                )
                for alias in node.names
            ):
                raise ValueError(f"label module import in {name}")
        elif isinstance(node, ast.Attribute) and _is_loader_like(
            _call_name(node)
        ):
            raise ValueError(f"label loader attribute in {name}")
        if isinstance(node, ast.Call):
            called = _call_name(node.func).casefold()
            if (
                _LABEL_API_ALLOWLIST[1] not in called
                and _is_loader_like(called)
            ):
                raise ValueError(f"label loader call in {name}")
            if called == "getattr" and len(node.args) >= 2:
                attribute = _static_string(node.args[1], constants)
                if attribute is not None and _is_loader_like(attribute):
                    raise ValueError(f"aliased label loader in {name}")
            if called in {"__import__", "importlib.import_module"}:
                target = (
                    _static_string(node.args[0], constants)
                    if node.args
                    else None
                )
                if target is None:
                    raise ValueError(f"dynamic label import target in {name}")
                if re.search(_LABEL_LITERAL_DENYLIST[1], target.casefold()):
                    raise ValueError(f"label import target in {name}")
            if called.rsplit(".", 1)[-1] in {
                "execute", "executemany", "executescript"
            }:
                query = (
                    _static_string(node.args[0], constants)
                    if node.args
                    else None
                )
                if query is None:
                    raise ValueError(f"dynamic label SQL in {name}")
                query_sha256 = sha256(
                    " ".join(query.split()).encode("utf-8")
                ).hexdigest()
                if query_sha256 not in _APPROVED_SQL_SHA256:
                    raise ValueError(f"unapproved label SQL in {name}")
        value = None if isinstance(node, ast.Name) else _static_string(
            node, constants
        )
        if value is None:
            continue
        if _contains_projected_label(value):
            raise ValueError(f"label SQL projection in {name}")
        if _looks_like_label_artifact(value):
            raise ValueError(f"frozen label artifact in {name}")


def propose_sensitive_inventory(
    sources: Mapping[str, str],
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    """Return deterministic proposed entries; never changes the checked-in list."""
    nodes_by_scope: dict[
        tuple[str, str, str], list[tuple[str, str, str]]
    ] = {}
    for name, source in sorted(sources.items()):
        tree = ast.parse(source, filename=name)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        data_nodes = _inventory_data_nodes(tree)
        constants = _static_constants(tree)
        _assert_direct_label_and_sql_denials(
            name, tree, constants, data_nodes
        )
        for node in ast.walk(tree):
            if node in data_nodes:
                continue
            kind = _sensitive_kind(node, parents)
            if kind is None:
                continue
            scope_name, scope = _enclosing_scope(node, tree, parents)
            scope_sha256 = sha256(
                _normalized_scope(scope).encode()
            ).hexdigest()
            nodes_by_scope.setdefault(
                (Path(name).name, scope_name, scope_sha256), []
            ).append(
                (
                    kind,
                    sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest(),
                    _path_from_scope(scope, node),
                )
            )
    entries = []
    for (basename, scope_name, scope_sha256), nodes in nodes_by_scope.items():
        normalized_nodes = tuple(sorted(nodes))
        entries.append(
            (
                basename,
                scope_name,
                "scope-sensitive-multiset",
                sha256(
                    "\n".join("|".join(node) for node in normalized_nodes).encode()
                ).hexdigest(),
                scope_sha256,
                str(len(normalized_nodes)),
            )
        )
    return tuple(sorted(entries))


def _verify_sensitive_inventory(
    sources: Mapping[str, str],
    expected: Sequence[tuple[str, str, str, str, str, str]],
) -> None:
    if len(expected) != len(set(expected)):
        raise ValueError("label isolation inventory has duplicate entries")
    observed = propose_sensitive_inventory(sources)
    if len(observed) != len(set(observed)):
        raise ValueError("label isolation inventory observed duplicate entries")
    if tuple(sorted(expected)) != observed:
        missing = len(set(expected) - set(observed))
        extra = len(set(observed) - set(expected))
        raise ValueError(
            "label isolation inventory mismatch "
            f"(missing={missing}, extra={extra})"
        )


def assert_source_label_isolation(sources: Mapping[str, str]) -> None:
    """Reject labels first, then require the exact checked-in sensitive inventory."""
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("label isolation requires source mappings")
    if any(
        not isinstance(name, str) or not isinstance(source, str)
        for name, source in sources.items()
    ):
        raise ValueError("label isolation sources must be strings")
    try:
        _verify_sensitive_inventory(
            sources, _SENSITIVE_OPERATION_ALLOWLIST
        )
    except SyntaxError as error:
        raise ValueError("label isolation cannot parse source") from error


def assert_exp419_label_isolation() -> None:
    assert_label_free_sql_contract()
    package = REPO_ROOT / "experiments/exp419_deterministic_cpu_parallel"
    paths = tuple(sorted(package.glob("*.py"))) + (
        REPO_ROOT / "scratch/benchmark_exp419_deterministic_cpu_parallel.py",
        REPO_ROOT / "scratch/run_exp419_deterministic_cpu_parallel.py",
    )
    assert_source_label_isolation(
        {str(path): path.read_text(encoding="utf-8") for path in paths}
    )


@dataclass(frozen=True)
class FrozenBenchmarkInputs:
    context: Any
    resources: RunnerResources
    costs: tuple[DatasetCost, ...]
    bootstrap: WorkerBootstrap
    paths: tuple[Path, ...]


def _sha256_text(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class WorkerUtilization:
    shard_index: int
    pid: int
    elapsed_seconds: float
    wall_occupancy_fraction: float

    def __post_init__(self) -> None:
        if type(self.shard_index) is not int or self.shard_index < 0:
            raise ValueError("worker utilization shard index is invalid")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("worker utilization PID is invalid")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, Real)
            or not isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("worker elapsed time must be finite and nonnegative")
        if (
            isinstance(self.wall_occupancy_fraction, bool)
            or not isinstance(self.wall_occupancy_fraction, Real)
            or not isfinite(self.wall_occupancy_fraction)
            or not 0.0 <= self.wall_occupancy_fraction <= 1.0
        ):
            raise ValueError("worker utilization must be finite and within [0, 1]")


def _worker_utilization_payload(worker: WorkerUtilization) -> dict[str, object]:
    return {
        "elapsed_seconds": worker.elapsed_seconds,
        "pid": worker.pid,
        "shard_index": worker.shard_index,
        "wall_occupancy_fraction": worker.wall_occupancy_fraction,
    }


@dataclass(frozen=True)
class TrialRuntimeMeasurement:
    worker_count: int
    wall_seconds: float
    process_cpu_seconds: float
    process_cpu_utilization: float
    worker_utilization: tuple[WorkerUtilization, ...]
    scientific_evidence_sha256: str
    telemetry_sha256: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.worker_count) is not int
            or self.worker_count not in FROZEN_CONFIG.worker_candidates
        ):
            raise ValueError("runtime measurement worker count is not frozen")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, Real)
            or not isfinite(self.wall_seconds)
            or self.wall_seconds <= 0.0
        ):
            raise ValueError("runtime wall time must be finite and positive")
        if (
            isinstance(self.process_cpu_seconds, bool)
            or not isinstance(self.process_cpu_seconds, Real)
            or not isfinite(self.process_cpu_seconds)
            or self.process_cpu_seconds < 0.0
        ):
            raise ValueError("process CPU time must be finite and nonnegative")
        expected_cpu_utilization = self.process_cpu_seconds / (
            self.wall_seconds * (self.worker_count + 1)
        )
        if (
            isinstance(self.process_cpu_utilization, bool)
            or not isinstance(self.process_cpu_utilization, Real)
            or not isfinite(self.process_cpu_utilization)
            or not 0.0 <= self.process_cpu_utilization <= 1.0
            or not isclose(
                self.process_cpu_utilization,
                expected_cpu_utilization,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("process CPU utilization is semantically inconsistent")
        workers = tuple(self.worker_utilization)
        if (
            len(workers) != self.worker_count
            or any(not isinstance(worker, WorkerUtilization) for worker in workers)
            or tuple(worker.shard_index for worker in workers)
            != tuple(range(self.worker_count))
        ):
            raise ValueError("worker utilization must cover exact frozen shards")
        for worker in workers:
            expected_occupancy = worker.elapsed_seconds / self.wall_seconds
            if (
                worker.elapsed_seconds > self.wall_seconds
                or not isclose(
                    worker.wall_occupancy_fraction,
                    expected_occupancy,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("worker utilization is semantically inconsistent")
        scientific = _sha256_text(
            self.scientific_evidence_sha256,
            "scientific evidence",
        )
        telemetry_payload = {
            "process_cpu_seconds": self.process_cpu_seconds,
            "process_cpu_utilization": self.process_cpu_utilization,
            "wall_seconds": self.wall_seconds,
            "worker_count": self.worker_count,
            "worker_utilization": [
                _worker_utilization_payload(worker) for worker in workers
            ],
        }
        telemetry = sha256(canonical_json_bytes(telemetry_payload)).hexdigest()
        evidence = sha256(
            canonical_json_bytes(
                {
                    "scientific_evidence_sha256": scientific,
                    "telemetry_sha256": telemetry,
                }
            )
        ).hexdigest()
        if self.telemetry_sha256 and self.telemetry_sha256 != telemetry:
            raise ValueError("runtime telemetry digest mismatch")
        if self.evidence_sha256 and self.evidence_sha256 != evidence:
            raise ValueError("runtime evidence digest mismatch")
        object.__setattr__(self, "worker_utilization", workers)
        object.__setattr__(self, "telemetry_sha256", telemetry)
        object.__setattr__(self, "evidence_sha256", evidence)


@dataclass(frozen=True)
class MeasuredBenchmark:
    benchmark: RuntimeBenchmark
    runtime_measurements: tuple[TrialRuntimeMeasurement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, RuntimeBenchmark):
            raise ValueError("measured benchmark requires RuntimeBenchmark")
        measurements = tuple(self.runtime_measurements)
        if (
            tuple(item.worker_count for item in measurements)
            != FROZEN_CONFIG.worker_candidates
        ):
            raise ValueError("measurements must cover exact frozen worker candidates")
        for trial, measurement in zip(
            self.benchmark.trials, measurements, strict=True
        ):
            if (
                trial.worker_count != measurement.worker_count
                or not isclose(
                    trial.elapsed_seconds,
                    measurement.wall_seconds,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or trial.evidence_sha256 != measurement.evidence_sha256
            ):
                raise ValueError("runtime measurement differs from benchmark trial")
        object.__setattr__(self, "runtime_measurements", measurements)


def costs_from_metadata_rows(
    rows: Sequence[tuple[object, object, object]],
    *,
    expected_datasets: Sequence[str],
) -> tuple[DatasetCost, ...]:
    grouped: dict[str, list[TrainLengthGroupMeta]] = {}
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
            or type(row[1]) is not int
            or row[1] <= 0
            or row[1] % 4
            or type(row[2]) is not int
            or row[2] <= 0
        ):
            raise ValueError("TRAIN metadata row is invalid")
        grouped.setdefault(row[0], []).append(
            TrainLengthGroupMeta(row[0], row[1] // 4, row[2])
        )
    expected = tuple(expected_datasets)
    if tuple(sorted(grouped)) != expected:
        raise ValueError("TRAIN metadata does not cover the exact frozen population")
    return tuple(
        estimate_dataset_cost(dataset, tuple(grouped[dataset]))
        for dataset in expected
    )


def _load_train_metadata_costs(database: Path, context: Any) -> tuple[DatasetCost, ...]:
    query = """
    SELECT d.name, length(i.values_blob), count(*)
    FROM instances AS i
    JOIN datasets AS d ON d.id = i.dataset_id
    WHERE i.split = 'TRAIN' AND i.label = '0'
    GROUP BY d.name, length(i.values_blob)
    ORDER BY d.name, length(i.values_blob)
    """
    with _verified_database_connection(
        database, context.config.database_sha256
    ) as connection:
        rows = connection.execute(query).fetchall()
    return costs_from_metadata_rows(rows, expected_datasets=context.datasets)


def freeze_benchmark_inputs(args: argparse.Namespace, verified_head: str) -> FrozenBenchmarkInputs:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    context = freeze_run_context(paths[0], lanes, verified_code_commit=verified_head)
    resources = RunnerResources(
        *paths, backend_audit_path=_backend_audit_path(paths[0])
    )
    provenance = _provenance(context)
    costs = _load_train_metadata_costs(paths[0], context)
    return FrozenBenchmarkInputs(
        context,
        resources,
        costs,
        WorkerBootstrap(resources, provenance, _context_sha256(context)),
        paths,
    )


def prepare_trial_roots(root: Path) -> Mapping[int, Path]:
    root = Path(root).resolve(strict=False)
    if root.exists():
        raise ValueError("benchmark root must be new")
    if not root.parent.is_dir():
        raise ValueError("benchmark parent must exist")
    root.mkdir()
    return {
        count: root / f"workers-{count}"
        for count in FROZEN_CONFIG.worker_candidates
    }


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _run_trials(
    frozen: FrozenBenchmarkInputs,
    trial_roots: Mapping[int, Path],
) -> MeasuredBenchmark:
    selected = select_benchmark_datasets(frozen.costs)
    if not selected:
        raise ValueError("frozen benchmark subset is empty")
    selected_set = set(selected)
    subset_costs = tuple(cost for cost in frozen.costs if cost.dataset in selected_set)
    if tuple(sorted(cost.dataset for cost in subset_costs)) != selected:
        raise ValueError("benchmark subset differs from frozen TRAIN-only selection")

    trials: list[RuntimeTrial] = []
    measurements: list[TrialRuntimeMeasurement] = []
    reference_digests: dict[str, str] | None = None
    for worker_count in FROZEN_CONFIG.worker_candidates:
        root = trial_roots[worker_count]
        if root.exists():
            raise ValueError("every benchmark candidate root must be fresh")
        root.mkdir()
        plan = build_shard_plan(subset_costs, worker_count)
        specs = tuple(
            WorkerSpec(
                assignment,
                root / f"worker-{assignment.shard_index}",
            )
            for assignment in plan.assignments
        )
        child_cpu_started = _children_cpu_seconds()
        parent_cpu_started = process_time()
        started = monotonic()
        completions = spawn_workers(specs, frozen.bootstrap)
        elapsed = monotonic() - started
        process_cpu_seconds = (
            process_time() - parent_cpu_started
            + _children_cpu_seconds() - child_cpu_started
        )
        record_digests = {
            dataset: digest
            for completion in completions
            for dataset, digest in completion.record_digests.items()
        }
        completed = (
            tuple(sorted(record_digests))
            if set(record_digests) == set(selected)
            else selected
        )
        finite = (
            elapsed > 0.0
            and all(completion.status == "completed" for completion in completions)
            and set(record_digests) == set(selected)
        )
        if worker_count == 1 and finite:
            reference_digests = dict(record_digests)
        parity = finite and reference_digests is not None and record_digests == reference_digests
        fallback = "" if finite else "; ".join(
            completion.error
            for completion in completions
            if completion.error
        ) or "incomplete benchmark trial"
        scientific_evidence = sha256(
            canonical_json_bytes(
                {
                    "completed_datasets": completed,
                    "record_digests": record_digests,
                }
            )
        ).hexdigest()
        worker_utilization = tuple(
            WorkerUtilization(
                completion.shard_index,
                completion.pid,
                completion.elapsed_seconds,
                completion.elapsed_seconds / elapsed,
            )
            for completion in sorted(
                completions, key=lambda item: item.shard_index
            )
        )
        measurement = TrialRuntimeMeasurement(
            worker_count,
            elapsed,
            process_cpu_seconds,
            process_cpu_seconds / (elapsed * (worker_count + 1)),
            worker_utilization,
            scientific_evidence,
        )
        measurements.append(measurement)
        parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            parent_peak_bytes = int(parent_peak)
        else:
            parent_peak_bytes = int(parent_peak) * 1024
        peak_rss = parent_peak_bytes + sum(
            completion.peak_rss_bytes for completion in completions
        )
        trials.append(
            RuntimeTrial(
                worker_count,
                elapsed,
                peak_rss,
                parity,
                finite,
                fallback,
                selected,
                measurement.evidence_sha256,
            )
        )
    return MeasuredBenchmark(
        select_runtime(tuple(trials)),
        tuple(measurements),
    )


def _trial_payload(trial: RuntimeTrial) -> dict[str, object]:
    return {
        "worker_count": trial.worker_count,
        "elapsed_seconds": trial.elapsed_seconds,
        "peak_rss_bytes": trial.peak_rss_bytes,
        "parity": trial.parity,
        "finite": trial.finite,
        "fallback": trial.fallback,
        "completed_datasets": list(trial.completed_datasets),
        "evidence_sha256": trial.evidence_sha256,
    }


def benchmark_payload(benchmark: RuntimeBenchmark) -> dict[str, object]:
    return {
        "trials": [_trial_payload(trial) for trial in benchmark.trials],
        "selected_worker_count": benchmark.selected_worker_count,
        "status": benchmark.status,
        "benchmark_datasets": list(benchmark.benchmark_datasets),
        "sha256": benchmark.sha256,
    }


def runtime_measurement_payload(
    measurement: TrialRuntimeMeasurement,
) -> dict[str, object]:
    return {
        "evidence_sha256": measurement.evidence_sha256,
        "process_cpu_seconds": measurement.process_cpu_seconds,
        "process_cpu_utilization": measurement.process_cpu_utilization,
        "scientific_evidence_sha256":
            measurement.scientific_evidence_sha256,
        "telemetry_sha256": measurement.telemetry_sha256,
        "wall_seconds": measurement.wall_seconds,
        "worker_count": measurement.worker_count,
        "worker_utilization": [
            _worker_utilization_payload(worker)
            for worker in measurement.worker_utilization
        ],
    }


def runtime_measurements_from_payload(
    payload: object,
) -> tuple[TrialRuntimeMeasurement, ...]:
    if not isinstance(payload, list):
        raise ValueError("runtime measurements must be a list")
    exact_fields = {
        "evidence_sha256",
        "process_cpu_seconds",
        "process_cpu_utilization",
        "scientific_evidence_sha256",
        "telemetry_sha256",
        "wall_seconds",
        "worker_count",
        "worker_utilization",
    }
    worker_fields = {
        "elapsed_seconds",
        "pid",
        "shard_index",
        "wall_occupancy_fraction",
    }
    measurements = []
    for raw in payload:
        if not isinstance(raw, dict) or set(raw) != exact_fields:
            raise ValueError("runtime measurement fields are invalid")
        raw_workers = raw["worker_utilization"]
        if not isinstance(raw_workers, list):
            raise ValueError("worker utilization must be a list")
        workers = []
        for raw_worker in raw_workers:
            if (
                not isinstance(raw_worker, dict)
                or set(raw_worker) != worker_fields
            ):
                raise ValueError("worker utilization fields are invalid")
            workers.append(
                WorkerUtilization(
                    raw_worker["shard_index"],
                    raw_worker["pid"],
                    raw_worker["elapsed_seconds"],
                    raw_worker["wall_occupancy_fraction"],
                )
            )
        measurements.append(
            TrialRuntimeMeasurement(
                raw["worker_count"],
                raw["wall_seconds"],
                raw["process_cpu_seconds"],
                raw["process_cpu_utilization"],
                tuple(workers),
                raw["scientific_evidence_sha256"],
                raw["telemetry_sha256"],
                raw["evidence_sha256"],
            )
        )
    return tuple(measurements)


def benchmark_from_payload(payload: object) -> RuntimeBenchmark:
    if not isinstance(payload, dict) or set(payload) != {
        "trials",
        "selected_worker_count",
        "status",
        "benchmark_datasets",
        "sha256",
    }:
        raise ValueError("benchmark evidence fields are invalid")
    raw_trials = payload["trials"]
    if not isinstance(raw_trials, list):
        raise ValueError("benchmark trials are invalid")
    trials = []
    exact_fields = {
        "worker_count",
        "elapsed_seconds",
        "peak_rss_bytes",
        "parity",
        "finite",
        "fallback",
        "completed_datasets",
        "evidence_sha256",
    }
    for raw in raw_trials:
        if not isinstance(raw, dict) or set(raw) != exact_fields:
            raise ValueError("benchmark trial fields are invalid")
        trials.append(
            RuntimeTrial(
                raw["worker_count"],
                raw["elapsed_seconds"],
                raw["peak_rss_bytes"],
                raw["parity"],
                raw["finite"],
                raw["fallback"],
                raw["completed_datasets"],
                raw["evidence_sha256"],
            )
        )
    benchmark = RuntimeBenchmark(
        tuple(trials),
        payload["selected_worker_count"],
        payload["status"],
        payload["benchmark_datasets"],
    )
    if benchmark.sha256 != payload["sha256"]:
        raise ValueError("benchmark digest mismatch")
    if select_runtime(benchmark.trials) != benchmark:
        raise ValueError("benchmark selection is not the frozen passing selection")
    return benchmark


def write_benchmark_evidence(
    path: Path,
    *,
    benchmark: RuntimeBenchmark,
    git_commit: str,
    context_sha256: str,
    trial_roots: Sequence[Path],
    runtime_measurements: Sequence[TrialRuntimeMeasurement],
) -> None:
    measured = MeasuredBenchmark(
        benchmark,
        tuple(runtime_measurements),
    )
    roots = tuple(Path(root).resolve(strict=False) for root in trial_roots)
    if len(roots) != len(FROZEN_CONFIG.worker_candidates) or len(set(roots)) != len(roots):
        raise ValueError("benchmark evidence requires exact disjoint trial roots")
    payload = {
        "artifact_kind": "exp419_non_official_runtime_benchmark",
        "benchmark": benchmark_payload(benchmark),
        "context_sha256": context_sha256,
        "git_commit": git_commit,
        "runtime_measurements": [
            runtime_measurement_payload(measurement)
            for measurement in measured.runtime_measurements
        ],
        "schema_version": 1,
        "trial_roots": [str(root) for root in roots],
        "worker_candidates": list(FROZEN_CONFIG.worker_candidates),
    }
    canonical = canonical_json_bytes(payload) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical)


@dataclass(frozen=True)
class BenchmarkStages:
    clean_git: Callable[[], str]
    label_guard: Callable[[], Any]
    freeze_inputs: Callable[[str], FrozenBenchmarkInputs]
    prepare_roots: Callable[[], Mapping[int, Path]]
    run_trials: Callable[
        [FrozenBenchmarkInputs, Mapping[int, Path]], MeasuredBenchmark
    ]
    revalidate: Callable[[], Any]
    write: Callable[
        [MeasuredBenchmark, str, FrozenBenchmarkInputs, Mapping[int, Path]],
        Any,
    ]


def execute_benchmark(stages: BenchmarkStages) -> int:
    head = stages.clean_git()
    stages.label_guard()
    frozen = stages.freeze_inputs(head)
    roots = stages.prepare_roots()
    measured = stages.run_trials(frozen, roots)
    stages.revalidate()
    stages.write(measured, head, frozen, roots)
    return 0


def make_test_stages(*, clean_git: Callable[[], str]) -> BenchmarkStages:
    unreachable = lambda *_args: (_ for _ in ()).throw(
        AssertionError("later stage reached")
    )
    return BenchmarkStages(
        clean_git,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
    )


def execute_for_test(stages: BenchmarkStages) -> int:
    return execute_benchmark(stages)


def _build_stages(args: argparse.Namespace) -> BenchmarkStages:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    state: dict[str, Any] = {}

    def freeze(head: str) -> FrozenBenchmarkInputs:
        frozen = freeze_benchmark_inputs(args, head)
        state["head"] = head
        return frozen

    def revalidate() -> None:
        if require_clean_git() != state["head"]:
            raise ValueError("Git HEAD changed during Exp419 benchmark")
        verify_input_snapshots(*paths)

    def write(
        measured: MeasuredBenchmark,
        head: str,
        frozen: FrozenBenchmarkInputs,
        roots: Mapping[int, Path],
    ) -> None:
        write_benchmark_evidence(
            args.benchmark_root / BENCHMARK_FILE,
            benchmark=measured.benchmark,
            git_commit=head,
            context_sha256=_context_sha256(frozen.context),
            trial_roots=tuple(roots[count] for count in FROZEN_CONFIG.worker_candidates),
            runtime_measurements=measured.runtime_measurements,
        )

    return BenchmarkStages(
        require_clean_git,
        assert_exp419_label_isolation,
        freeze,
        lambda: prepare_trial_roots(args.benchmark_root),
        _run_trials,
        revalidate,
        write,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_benchmark(_build_stages(args))


if __name__ == "__main__":
    raise SystemExit(main())
