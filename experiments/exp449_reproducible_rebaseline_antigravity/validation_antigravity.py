"""Pure Exp449 manifest verification primitives; intentionally paper-writer free."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path, PurePosixPath
import stat

import numpy as np

from .config_antigravity import FROZEN_EXP449_CONFIG, FROZEN_EXP449_POLICY_KEYS
from .contracts_antigravity import Exp449CheckpointDescriptor, Exp449Context, Exp449ProposalManifest, Exp449ShardDescriptor


_ARTIFACT_DIRECTORY = "stage_artifacts_antigravity"
_FREEZE_DIRECTORY = "proposal_freeze_antigravity"
_SHARDS_DIRECTORY = "shards_antigravity"
_MANIFEST_NAME = "proposal_manifest_antigravity.json"
_SHARD_SCHEMA = "exp449-proposal-shard-antigravity.v1"
_TYPED_SHARD_SCHEMA = "paper-task8-proposal-shard.v3"
_HARD_SOURCES = ("rocket_exp40", "exp55_best", "exp56_best", "exp84", "rocket_block_b", "canonical_normal_state")
_SOURCE_ALPHA = {"rocket_exp40": .03, "exp55_best": .03, "exp56_best": .02, "exp84": .02, "rocket_block_b": .015, "canonical_normal_state": .02}
_HARD_STAGE_IDS = ("exp74d", "exp89", "exp90", "exp93", "exp137", "exp234", "exp236", "exp253")
_TASK45_SCIENTIFIC_CONFIG_SHA256 = "6b276d90f5edeb8a49810c9caa062cc8c8d1750f202f56720b9a9df537a2f6e3"


def canonical_json_bytes_antigravity(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Exp449 canonical JSON differs") from error


def canonical_float64_bytes_antigravity(values: object) -> bytes:
    array = np.asarray(values, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Exp449 score vector differs")
    return np.ascontiguousarray(array).tobytes(order="C")


def _plain(value: object) -> object:
    if isinstance(value, Mapping): return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple): return [_plain(child) for child in value]
    if isinstance(value, Path): return str(value)
    return value


def _lstat(path: Path):
    try: return path.lstat()
    except FileNotFoundError: return None


def _safe_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts or "exp448" in root.as_posix().casefold():
        raise ValueError("Exp449 root authority differs")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part; details = _lstat(current)
        if details is not None and stat.S_ISLNK(details.st_mode): raise ValueError("Exp449 symlinked root component is forbidden")
    return root


def _context_sha256(context: Exp449Context) -> str:
    if not isinstance(context, Exp449Context) or context.experiment_id != "Exp449" or context.historical_parity_status != "not_used":
        raise ValueError("Exp449 context differs")
    value = {"experiment_id":context.experiment_id,"mode":context.mode,"run_root":str(context.run_root),"database_sha256":context.database_sha256,"catalog_sha256":context.catalog_sha256,"code_closure":_plain(context.code_closure),"lineage_sha256":context.lineage_sha256,"codemap_sha256":context.codemap_sha256,"interpreter":context.interpreter,"environment":_plain(context.environment),"backend_record":_plain(context.backend_record),"seeds_and_constants":_plain(context.seeds_and_constants),"historical_parity_status":context.historical_parity_status}
    return sha256(canonical_json_bytes_antigravity(value)).hexdigest()


def _task45_context_sha256(context: Exp449Context) -> str:
    """The exact paper Task4/5 bridge identity, reconstructed without imports."""
    value = {"experiment_id":context.experiment_id,"mode":context.mode,"run_root":str(context.run_root),"database_sha256":context.database_sha256,"catalog_sha256":context.catalog_sha256,"code_closure":_plain(context.code_closure),"lineage_sha256":context.lineage_sha256,"codemap_sha256":context.codemap_sha256,"interpreter":context.interpreter,"environment":_plain(context.environment),"backend_record":_plain(context.backend_record),"seeds_and_constants":_plain(context.seeds_and_constants),"historical_parity_status":context.historical_parity_status}
    return sha256(canonical_json_bytes_antigravity(value)).hexdigest()


def _task45_code_closure_sha256(context: Exp449Context) -> str:
    return sha256(canonical_json_bytes_antigravity(_plain(context.code_closure))).hexdigest()


def _catalog_names(catalog: object, mode: str, manifest: Exp449ProposalManifest | None = None) -> tuple[str, ...]:
    if mode not in {"smoke", "official"} or not isinstance(catalog, Mapping) or type(catalog.get("catalog_sha256")) is not str:
        raise ValueError("Exp449 catalog differs")
    bare={key:value for key,value in catalog.items() if key != "catalog_sha256"}
    if sha256(canonical_json_bytes_antigravity(bare)).hexdigest() != catalog["catalog_sha256"]:
        raise ValueError("Exp449 catalog self-hash differs")
    entries=catalog.get("datasets")
    if not isinstance(entries,list): raise ValueError("Exp449 catalog differs")
    names=tuple(item.get("dataset") if isinstance(item,Mapping) else None for item in entries)
    if any(type(name) is not str or not name or "/" in name or "\\" in name for name in names) or names != tuple(sorted(names)) or len(set(names)) != len(names): raise ValueError("Exp449 catalog differs")
    if mode == "smoke":
        if catalog.get("mode") != "smoke" or names != FROZEN_EXP449_CONFIG.smoke_datasets: raise ValueError("Exp449 smoke catalog differs")
    else:
        counts=[item.get("test",{}).get("count") if isinstance(item,Mapping) and isinstance(item.get("test"),Mapping) else None for item in entries]
        if catalog.get("mode") != "full" or len(names) != 1117 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 195900: raise ValueError("Exp449 full catalog differs")
    if manifest is not None and tuple(item.dataset for item in manifest.entries) != names: raise ValueError("Exp449 manifest catalog inventory differs")
    return names


def validate_exp449_catalog_antigravity(catalog: object, *, mode: str, manifest: Exp449ProposalManifest | None = None) -> tuple[str, ...]:
    return _catalog_names(catalog, mode, manifest)


def _entry(catalog: Mapping[str, object], dataset: str) -> dict[str, object]:
    rows=[item for item in catalog["datasets"] if isinstance(item,dict) and item.get("dataset")==dataset]
    if len(rows)!=1: raise ValueError("Exp449 catalog shard identity differs")
    return rows[0]


def _relative(dataset: str) -> str:
    return f"{_SHARDS_DIRECTORY}/{sha256(dataset.encode()).hexdigest()}_antigravity/{dataset}_proposal_shard_antigravity.json"


def _integer_list(value: object, n: int, name: str) -> list[int]:
    if not isinstance(value,list) or any(type(x) is not int or x < 0 or x >= n for x in value) or value != sorted(set(value)): raise ValueError(f"Exp449 {name} differs")
    return value


def _scores(value: object, n: int) -> list[float]:
    if not isinstance(value,list) or len(value)!=n or any(type(x) not in (int,float) or not isfinite(float(x)) for x in value): raise ValueError("Exp449 policy scores differ")
    return [float(x) for x in value]


def _bools(value: object, n: int, name: str) -> list[bool]:
    if not isinstance(value, list) or len(value) != n or any(type(item) is not bool for item in value):
        raise ValueError(f"Exp449 {name} differs")
    return value


def _lanes(value: object, n: int, name: str) -> dict[str, list[int]]:
    if not isinstance(value, dict) or set(value) != {"hard", "standard", "priority"}:
        raise ValueError(f"Exp449 {name} lanes differ")
    result = {key: _integer_list(value[key], n, f"{name} {key}") for key in value}
    lanes = [set(result[key]) for key in ("hard", "standard", "priority")]
    if lanes[0] & lanes[1] or lanes[0] & lanes[2] or lanes[1] & lanes[2]:
        raise ValueError(f"Exp449 {name} lanes overlap")
    return result


def _low_provisional(value: object, n: int) -> set[int]:
    """Check the full serialised Task6 row grammar and its local relations."""
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("Exp449 low-frequency row coverage differs")
    provisional: set[int] = set()
    for index, row in enumerate(value):
        required = {"index", "evidence_missing", "strict_pass", "provisional", "depths"}
        if not isinstance(row, dict) or set(row) != required or row["index"] != index:
            raise ValueError("Exp449 low-frequency row schema differs")
        if any(type(row[key]) is not bool for key in ("evidence_missing", "strict_pass", "provisional")) or not isinstance(row["depths"], list):
            raise ValueError("Exp449 low-frequency row flags differ")
        depths = row["depths"]
        if row["evidence_missing"]:
            if depths or row["strict_pass"] or row["provisional"]:
                raise ValueError("Exp449 missing low-frequency evidence differs")
            continue
        if not depths:
            raise ValueError("Exp449 available low-frequency evidence differs")
        seen: set[int] = set()
        passing = 0
        for depth in depths:
            needed = {"depth", "score", "threshold", "strict_pass", "excess_ratio", "empirical_rank"}
            if not isinstance(depth, dict) or set(depth) != needed or type(depth["depth"]) is not int or depth["depth"] < 1 or depth["depth"] in seen or type(depth["strict_pass"]) is not bool:
                raise ValueError("Exp449 low-frequency depth schema differs")
            seen.add(depth["depth"])
            if any(type(depth[key]) not in (int, float) or not isfinite(float(depth[key])) for key in ("score", "threshold", "excess_ratio", "empirical_rank")):
                raise ValueError("Exp449 low-frequency depth values differ")
            score, threshold = float(depth["score"]), float(depth["threshold"])
            ratio, rank = float(depth["excess_ratio"]), float(depth["empirical_rank"])
            if score < 0 or threshold < 0 or ratio < 0 or not 0 < rank <= 1 or depth["strict_pass"] != (score > threshold) or not isclose(ratio, score / max(threshold, 1e-12), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Exp449 low-frequency depth relation differs")
            passing += int(depth["strict_pass"])
        if [depth["depth"] for depth in depths] != sorted(seen):
            raise ValueError("Exp449 low-frequency depth order differs")
        strict = passing >= 2
        if row["strict_pass"] != strict or row["provisional"] != strict:
            raise ValueError("Exp449 low-frequency provisional relation differs")
        if strict:
            provisional.add(index)
    return provisional


def _frozen_policy_scores(source_rows: list[object], low_rows: object, n: int, alerts: list[int]) -> list[float]:
    """Rebuild the serialised route_preserving_scores inputs without paper imports."""
    hard_leaves: list[list[float]] = []
    eligible = False
    for row in source_rows:
        assert isinstance(row, dict)
        if row["source_id"] in {"rocket_exp40", "exp55_best", "exp56_best", "exp84", "canonical_normal_state"}:
            hard_leaves.append([1.0 - float(value) if row["eligible"] else 0.0 for value in row["pvalue"]])
            eligible = eligible or bool(row["eligible"])
    hard = np.sort(np.asarray(hard_leaves, dtype=np.float64), axis=0)[-2]
    source = np.zeros(n, dtype=np.float64)
    source_missing = np.ones(n, dtype=bool)
    assert isinstance(low_rows, list)
    for row in low_rows:
        assert isinstance(row, dict)
        if row["evidence_missing"]:
            continue
        ranks = [float(depth["empirical_rank"]) for depth in row["depths"]]
        if len(ranks) < 2:
            raise ValueError("Exp449 scoreable low-frequency row differs")
        source[row["index"]] = np.sort(1.0 - np.asarray(ranks, dtype=np.float64))[-2]
        source_missing[row["index"]] = False
    hard_values = np.where(not eligible, 0.0, hard)
    source_values = np.where(source_missing, 0.0, source)
    combined = np.maximum(hard_values, source_values)
    combined = np.where(np.full(n, not eligible, dtype=bool) & source_missing, .5, combined)
    mask = np.asarray([index in set(alerts) for index in range(n)], dtype=bool)
    return np.where(mask, .5 + .5 * combined, .5 * combined).tolist()


def _scientific(value: object, entry: Mapping[str, object], automatic_override: object = None) -> dict[str, object]:
    required={"schema","dataset","train_identity","test_identity","source_rows","hard_stages","low_frequency_rows","stability","policies","backend"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!=_TYPED_SHARD_SCHEMA or value["dataset"]!=entry.get("dataset") or value["train_identity"]!=entry.get("train") or value["test_identity"]!=entry.get("test"): raise ValueError("Exp449 scientific payload differs")
    test=entry.get("test"); n=test.get("count") if isinstance(test,Mapping) else None
    if type(n) is not int or n < 0: raise ValueError("Exp449 scientific test count differs")

    source_rows=value["source_rows"]
    if not isinstance(source_rows,list) or [row.get("source_id") if isinstance(row,dict) else None for row in source_rows] != list(_HARD_SOURCES): raise ValueError("Exp449 source order differs")
    for order,row in enumerate(source_rows):
        fields={"source_id","order","test_score","pvalue","native_pass","eligible","reason"}
        if not isinstance(row,dict) or set(row)!=fields or row["order"]!=order or type(row["eligible"]) is not bool or type(row["reason"]) is not str or not row["reason"]: raise ValueError("Exp449 source schema differs")
        pvalue=_scores(row["pvalue"],n)
        if any(item<=0 or item>1 for item in pvalue): raise ValueError("Exp449 source pvalue domain differs")
        native=_bools(row["native_pass"],n,"source native pass")
        if native != [item <= _SOURCE_ALPHA[row["source_id"]] for item in pvalue]: raise ValueError("Exp449 source native-pass relation differs")
        if row["eligible"]:
            _scores(row["test_score"],n)
            if row["reason"] != "eligible": raise ValueError("Exp449 eligible source reason differs")
        elif row["test_score"] != [] or pvalue != [1.0]*n or any(native): raise ValueError("Exp449 abstained source relation differs")

    stages=value["hard_stages"]
    if not isinstance(stages,list) or [row.get("stage_id") if isinstance(row,dict) else None for row in stages] != list(_HARD_STAGE_IDS): raise ValueError("Exp449 hard stage order differs")
    lanes: dict[str,dict[str,list[int]]] = {}; indices: dict[str,list[int]] = {}
    for order,row in enumerate(stages):
        if not isinstance(row,dict) or row.get("order") != order: raise ValueError("Exp449 hard stage schema differs")
        stage=_HARD_STAGE_IDS[order]
        if stage in {"exp74d","exp137","exp236","exp253"}:
            if set(row)!={"stage_id","order","lanes"}: raise ValueError("Exp449 hard lane stage schema differs")
            lanes[stage]=_lanes(row["lanes"],n,stage)
        elif stage == "exp234":
            if set(row)!={"stage_id","order","minimum_two","minimum_three"}: raise ValueError("Exp449 source expansion schema differs")
            indices["minimum_two"]=_integer_list(row["minimum_two"],n,"minimum two"); indices["minimum_three"]=_integer_list(row["minimum_three"],n,"minimum three")
            if not set(indices["minimum_three"]) <= set(indices["minimum_two"]): raise ValueError("Exp449 source expansion relation differs")
        else:
            if set(row)!={"stage_id","order","indices"}: raise ValueError("Exp449 hard index stage schema differs")
            indices[stage]=_integer_list(row["indices"],n,stage)
    if not set(lanes["exp74d"]["hard"]) <= set(indices["exp89"]) or not set(indices["exp89"]) <= set(indices["exp90"]): raise ValueError("Exp449 early hard-stage relation differs")
    exp137=lanes["exp137"]
    if not (set(exp137["hard"]) | set(exp137["standard"])) <= set(indices["exp93"]): raise ValueError("Exp449 Exp137 route relation differs")
    if set(indices["minimum_two"]) & set().union(*(set(item) for item in exp137.values())): raise ValueError("Exp449 source expansion route overlap differs")
    exp236,exp253=lanes["exp236"],lanes["exp253"]
    if set(exp236["hard"]) != set(exp137["hard"]) | set(indices["minimum_three"]) or set(exp236["standard"]) != set(exp137["standard"]) | (set(indices["minimum_two"])-set(indices["minimum_three"])) or exp236["priority"] != exp137["priority"]: raise ValueError("Exp449 Exp236 route relation differs")
    if not set(exp236["hard"]) <= set(exp253["hard"]) or not set(exp253["standard"]) <= set(exp236["standard"]) or exp253["priority"] != exp236["priority"]: raise ValueError("Exp449 Exp253 route relation differs")
    automatic=_integer_list(exp253["hard"] if automatic_override is None else automatic_override,n,"hard alerts")
    provisional=_low_provisional(value["low_frequency_rows"],n)
    stability=value["stability"]
    if not isinstance(stability,dict) or set(stability)!={"selected","stable","unstable"}: raise ValueError("Exp449 stability differs")
    def coordinates(name: str) -> list[int]:
        rows=stability[name]
        if not isinstance(rows,list) or any(not isinstance(row,list) or len(row)!=2 or row[0]!=entry["dataset"] for row in rows): raise ValueError("Exp449 stability differs")
        return _integer_list([row[1] for row in rows],n,"stability coordinates")
    selected,stable,unstable=coordinates("selected"),coordinates("stable"),coordinates("unstable")
    if set(stable)&set(unstable) or set(stable)|set(unstable)!=set(selected): raise ValueError("Exp449 stability partition differs")
    if len(selected)>1 or any(index not in provisional for index in selected) or (automatic and selected): raise ValueError("Exp449 selected low-frequency row relation differs")
    if selected:
        candidates=[]
        for row in value["low_frequency_rows"]:
            if row["provisional"]:
                strict=[item for item in row["depths"] if item["strict_pass"]]
                candidates.append((-sorted((float(item["excess_ratio"]) for item in strict),reverse=True)[1],sorted(float(item["empirical_rank"]) for item in strict)[1],row["index"]))
        if selected[0] != min(candidates)[2]: raise ValueError("Exp449 selected low-frequency ranking differs")
    policies=value["policies"]
    if not isinstance(policies,list) or [row.get("policy_id") if isinstance(row,dict) else None for row in policies] != list(FROZEN_EXP449_POLICY_KEYS): raise ValueError("Exp449 policy closure differs")
    expected=(set(automatic),set(automatic)|set(selected),set(automatic)|set(stable),set(automatic)|set(unstable))
    for order,(row,want) in enumerate(zip(policies,expected,strict=True)):
        if not isinstance(row,dict) or set(row)!={"policy_id","order","alerts","scores","score_mask"} or row["order"]!=order or _integer_list(row["alerts"],n,"policy alerts") != sorted(want): raise ValueError("Exp449 policy alerts differ")
        scores=_scores(row["scores"],n)
        if scores != _frozen_policy_scores(source_rows,value["low_frequency_rows"],n,row["alerts"]): raise ValueError("Exp449 policy continuous score relation differs")
        if not isinstance(row["score_mask"],list) or len(row["score_mask"])!=n or any(type(x) is not bool for x in row["score_mask"]) or row["score_mask"] != [score >= .5 for score in scores]: raise ValueError("Exp449 policy score mask differs")
        if set(row["alerts"]) != {index for index,score in enumerate(scores) if score >= .5}: raise ValueError("Exp449 policy alerts and scores differ")
    return value


def _digests(scientific: Mapping[str, object], dataset: str) -> tuple[dict[str,str],str]:
    result={}
    candidate=None
    for row in scientific["policies"]:
        result[row["policy_id"]]=sha256(canonical_json_bytes_antigravity([[dataset,index] for index in row["alerts"]])).hexdigest()
        if row["policy_id"]==FROZEN_EXP449_CONFIG.policy_id: candidate=sha256(canonical_float64_bytes_antigravity(row["scores"])).hexdigest()
    assert candidate is not None
    return result,candidate


def validate_exp449_proposal_shard_antigravity(value: object, *, entry: dict[str, object], context: Exp449Context, catalog: Mapping[str, object], automatic_hard: object = None) -> tuple[dict[str,str],str]:
    if not isinstance(value,dict) or set(value)!={"schema","catalog_sha256","backend_record","scientific_payload"} or value["schema"]!=_SHARD_SCHEMA or value["catalog_sha256"]!=context.catalog_sha256 or value["catalog_sha256"]!=catalog.get("catalog_sha256") or value["backend_record"]!=_plain(context.backend_record): raise ValueError("Exp449 proposal shard authority differs")
    scientific=_scientific(value["scientific_payload"],entry,automatic_hard)
    if scientific["backend"] != value["backend_record"]: raise ValueError("Exp449 proposal shard backend differs")
    return _digests(scientific,entry["dataset"])


def _strict(path: Path) -> object:
    details=_lstat(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode): raise ValueError("Exp449 artifact is missing or symlinked")
    raw=path.read_bytes()
    try: value=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("Exp449 artifact JSON differs") from error
    if raw != canonical_json_bytes_antigravity(value): raise ValueError("Exp449 artifact noncanonical")
    return value


def _manifest(value: object) -> Exp449ProposalManifest:
    if not isinstance(value,dict) or set(value)!={"schema","context_sha256","catalog_sha256","backend_record","entries","historical_parity_status","task45_checkpoints"} or not isinstance(value.get("entries"),list) or not isinstance(value.get("task45_checkpoints"),list): raise ValueError("Exp449 proposal manifest schema differs")
    return Exp449ProposalManifest(value["schema"],value["context_sha256"],value["catalog_sha256"],value["backend_record"],tuple(Exp449ShardDescriptor(**item) for item in value["entries"]),value["historical_parity_status"],tuple(Exp449CheckpointDescriptor(**item) for item in value["task45_checkpoints"]))


def _task45_payload(value: object, entry: Mapping[str, object]) -> None:
    """Pure structural reconstruction of the paper publisher's typed success payload."""
    if not isinstance(value, dict) or set(value) != {"sources", "route"}:
        raise ValueError("Exp449 Task4/5 success payload differs")
    train = entry.get("train"); test = entry.get("test")
    n_train = train.get("count") if isinstance(train, Mapping) else None
    n_test = test.get("count") if isinstance(test, Mapping) else None
    if type(n_train) is not int or type(n_test) is not int:
        raise ValueError("Exp449 Task4/5 raw identity differs")
    sources = value["sources"]
    fields = {"source_id","order","origin","train_oof_score","test_score","test_pvalue","threshold","test_native_pass","eligible","eligibility_reason"}
    if not isinstance(sources, list) or [row.get("source_id") if isinstance(row, dict) else None for row in sources] != list(_HARD_SOURCES):
        raise ValueError("Exp449 Task4/5 source order differs")
    for order, row in enumerate(sources):
        if not isinstance(row, dict) or set(row) != fields or row["order"] != order or type(row["eligible"]) is not bool or type(row["eligibility_reason"]) is not str or not row["eligibility_reason"] or not isinstance(row["origin"], dict):
            raise ValueError("Exp449 Task4/5 source schema differs")
        threshold = row["threshold"]
        if type(threshold) not in (int,float) or float(threshold) != _SOURCE_ALPHA[row["source_id"]]:
            raise ValueError("Exp449 Task4/5 source threshold differs")
        pvalues = _scores(row["test_pvalue"], n_test)
        if any(item <= 0 or item > 1 for item in pvalues) or _bools(row["test_native_pass"], n_test, "Task4/5 native pass") != [item <= float(threshold) for item in pvalues]:
            raise ValueError("Exp449 Task4/5 source pvalue relation differs")
        if row["eligible"]:
            _scores(row["train_oof_score"], n_train); _scores(row["test_score"], n_test)
            if row["eligibility_reason"] != "eligible": raise ValueError("Exp449 Task4/5 eligible source differs")
        elif row["train_oof_score"] != [] or row["test_score"] != [] or pvalues != [1.0] * n_test:
            raise ValueError("Exp449 Task4/5 abstained source differs")
    route=value["route"]
    required={"n_test","exp74d","exp89","exp90","exp93","exp137","exp234","exp236","promoted","exp253","exact_normal_removed","automatic_hard"}
    if not isinstance(route,dict) or set(route)!=required or route["n_test"] != n_test or not isinstance(route["exp234"],dict) or set(route["exp234"]) != {"minimum_two","minimum_three"}:
        raise ValueError("Exp449 Task4/5 route schema differs")
    lanes={name:_lanes(route[name],n_test,f"Task4/5 {name}") for name in ("exp74d","exp137","exp236","exp253")}
    values={name:_integer_list(route[name],n_test,f"Task4/5 {name}") for name in ("exp89","exp90","exp93","promoted","exact_normal_removed","automatic_hard")}
    minimum_two=_integer_list(route["exp234"]["minimum_two"],n_test,"Task4/5 minimum two"); minimum_three=_integer_list(route["exp234"]["minimum_three"],n_test,"Task4/5 minimum three")
    if not set(minimum_three)<=set(minimum_two) or not set(values["promoted"])<=set(lanes["exp236"]["standard"]) or set(lanes["exp253"]["hard"]) != set(lanes["exp236"]["hard"]) | set(values["promoted"]) or set(lanes["exp253"]["standard"]) != set(lanes["exp236"]["standard"]) - set(values["promoted"]) or lanes["exp253"]["priority"] != lanes["exp236"]["priority"] or not set(values["exact_normal_removed"])<=set(lanes["exp253"]["hard"]) or set(values["automatic_hard"]) != set(lanes["exp253"]["hard"]) - set(values["exact_normal_removed"]):
        raise ValueError("Exp449 Task4/5 route algebra differs")


def _checkpoint(value: object, descriptor: Exp449CheckpointDescriptor, context: Exp449Context, catalog: Mapping[str,object]) -> None:
    required={"schema","context_sha256","catalog_sha256","raw_identity","backend","scientific_config_sha256","code_closure_sha256","dataset"}
    if not isinstance(value,dict) or not required <= set(value) or value.get("schema") != descriptor.schema or value.get("dataset") != descriptor.dataset or value.get("catalog_sha256") != context.catalog_sha256 or value.get("catalog_sha256") != catalog.get("catalog_sha256") or value.get("context_sha256") != _task45_context_sha256(context) or value.get("scientific_config_sha256") != _TASK45_SCIENTIFIC_CONFIG_SHA256 or value.get("code_closure_sha256") != _task45_code_closure_sha256(context): raise ValueError("Exp449 Task4/5 checkpoint authority differs")
    entry=_entry(catalog,descriptor.dataset) if descriptor.dataset is not None else None
    if entry is None or value.get("raw_identity") != entry: raise ValueError("Exp449 Task4/5 checkpoint identity differs")
    if descriptor.schema == "paper-task4-task5-success.v1":
        if set(value) != required|{"payload","payload_sha256"} or sha256(canonical_json_bytes_antigravity(value["payload"])).hexdigest()!=value["payload_sha256"]: raise ValueError("Exp449 Task4/5 success checkpoint differs")
        _task45_payload(value["payload"], entry)
    else:
        if set(value) != required|{"wave","attempt","exception_type","exception_message","record_sha256"}: raise ValueError("Exp449 Task4/5 attempt checkpoint differs")
        bare={key:child for key,child in value.items() if key!="record_sha256"}
        if sha256(canonical_json_bytes_antigravity(bare)).hexdigest()!=value["record_sha256"]: raise ValueError("Exp449 Task4/5 attempt checkpoint differs")


def _global_low_candidates(scientific: Mapping[str, object]) -> tuple[tuple[float,float,str,int], ...]:
    """Reconstruct Task6's Hard-empty local winner from one typed shard."""
    stages = scientific["hard_stages"]
    assert isinstance(stages, list) and isinstance(scientific["low_frequency_rows"], list)
    if stages[-1]["lanes"]["hard"]:
        return ()
    candidates=[]
    for row in scientific["low_frequency_rows"]:
        assert isinstance(row, dict)
        if row["provisional"]:
            strict=[item for item in row["depths"] if item["strict_pass"]]
            candidates.append((-sorted(float(item["excess_ratio"]) for item in strict)[-2],sorted(float(item["empirical_rank"]) for item in strict)[1],str(scientific["dataset"]),int(row["index"])))
    return (min(candidates),) if candidates else ()


def verify_exp449_proposal_root_antigravity(root: Path, *, context: Exp449Context, catalog: dict[str,object]) -> Exp449ProposalManifest:
    root=_safe_root(root)
    if "antigravity" not in root.name or Path(context.run_root).resolve(strict=False)!=root.resolve(strict=False): raise ValueError("Exp449 root authority differs")
    freeze=root/_ARTIFACT_DIRECTORY/_FREEZE_DIRECTORY
    manifest=_manifest(_strict(freeze/_MANIFEST_NAME)); names=_catalog_names(catalog,context.mode,manifest)
    if manifest.context_sha256!=_context_sha256(context) or manifest.catalog_sha256!=context.catalog_sha256 or manifest.catalog_sha256!=catalog.get("catalog_sha256") or _plain(manifest.backend_record)!=_plain(context.backend_record): raise ValueError("Exp449 proposal manifest authority differs")
    declared={f"{_ARTIFACT_DIRECTORY}/{_FREEZE_DIRECTORY}/{_MANIFEST_NAME}"}
    checkpoint_automatic: dict[str,list[int]] = {}
    for descriptor in manifest.task45_checkpoints:
        if descriptor.schema == "paper-task4-task5-success.v1":
            record = _strict(root / descriptor.relative_path)
            if isinstance(record, dict) and isinstance(record.get("payload"), dict) and isinstance(record["payload"].get("route"), dict):
                automatic = record["payload"]["route"].get("automatic_hard")
                if isinstance(automatic, list) and descriptor.dataset is not None:
                    checkpoint_automatic[descriptor.dataset] = automatic
    global_candidates: list[tuple[float,float,str,int]] = []
    selected_coordinates: list[tuple[str,int]] = []
    for descriptor in manifest.entries:
        if descriptor.relative_path != _relative(descriptor.dataset): raise ValueError("Exp449 shard path differs")
        rel=f"{_ARTIFACT_DIRECTORY}/{_FREEZE_DIRECTORY}/{descriptor.relative_path}"; declared.add(rel); path=root/rel; details=_lstat(path)
        if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_size!=descriptor.size or sha256(path.read_bytes()).hexdigest()!=descriptor.sha256: raise ValueError("Exp449 shard hash differs")
        shard=_strict(path)
        coords,candidate=validate_exp449_proposal_shard_antigravity(shard,entry=_entry(catalog,descriptor.dataset),context=context,catalog=catalog,automatic_hard=checkpoint_automatic.get(descriptor.dataset))
        if dict(descriptor.policy_coordinate_sha256)!=coords or descriptor.candidate_c_score_sha256!=candidate: raise ValueError("Exp449 shard digest differs")
        assert isinstance(shard, dict) and isinstance(shard["scientific_payload"], dict)
        scientific=shard["scientific_payload"]
        global_candidates.extend(_global_low_candidates(scientific))
        stability=scientific["stability"]
        assert isinstance(stability, dict)
        selected_coordinates.extend((str(row[0]),int(row[1])) for row in stability["selected"])
    expected_selected=tuple((dataset,index) for _ratio,_rank,dataset,index in sorted(global_candidates)[:30])
    if tuple(sorted(selected_coordinates)) != tuple(sorted(expected_selected)):
        raise ValueError("Exp449 global Candidate C selection differs")
    checkpoints = tuple(manifest.task45_checkpoints)
    if checkpoints:
        success = tuple(item for item in checkpoints if item.schema == "paper-task4-task5-success.v1")
        attempts = tuple(item for item in checkpoints if item.schema == "paper-task4-task5-attempt.v1")
        expected_success = {
            f"stage_artifacts_paper/proposal_work/task4_task5_success/{sha256(dataset.encode()).hexdigest()}_antigravity/task4_task5_success_antigravity.json"
            for dataset in names
        }
        if {item.relative_path for item in success} != expected_success or len(success) != len(names):
            raise ValueError("Exp449 Task4/5 success checkpoint inventory differs")
        expected_attempts = tuple(
            f"stage_artifacts_paper/proposal_work/task4_task5_attempts/attempt_{index:06d}_antigravity.json"
            for index in range(1, len(attempts) + 1)
        )
        if tuple(item.relative_path for item in attempts) != expected_attempts:
            raise ValueError("Exp449 Task4/5 attempt checkpoint inventory differs")
    for descriptor in checkpoints:
        rel=descriptor.relative_path; declared.add(rel); path=root/rel; details=_lstat(path)
        if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_size!=descriptor.size or sha256(path.read_bytes()).hexdigest()!=descriptor.sha256: raise ValueError("Exp449 Task4/5 checkpoint hash differs")
        _checkpoint(_strict(path),descriptor,context,catalog)
    declared_nodes=set(declared)
    for item in tuple(declared):
        parent=PurePosixPath(item).parent
        while parent != PurePosixPath("."):
            declared_nodes.add(parent.as_posix()); parent=parent.parent
    found={item.relative_to(root).as_posix() for item in root.rglob("*")}
    def permitted_name(item: str) -> bool:
        return item in {"stage_artifacts_paper", "stage_artifacts_paper/proposal_work"} or item.startswith("stage_artifacts_paper/proposal_work/") or all("antigravity" in part for part in item.split("/"))
    if found!=declared_nodes or any(not permitted_name(item) for item in found): raise ValueError("Exp449 root inventory differs")
    return manifest


__all__=["canonical_float64_bytes_antigravity","canonical_json_bytes_antigravity","validate_exp449_catalog_antigravity","validate_exp449_proposal_shard_antigravity","verify_exp449_proposal_root_antigravity"]
