import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yggdrasil.flow.model import CONTINUE_INDEPENDENT_POLICY, Plan, StepSpec

from .nextflow import NextflowRunOptions
from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    SamplesheetBranch,
)

_PREFIX = "demux_realm.steps"

VALIDATE_RUNFOLDER = "validate_runfolder"
SETUP_COMMON_DEMUX_DIR = "setup_common_demux_dir"
UPSERT_X_FLOWCELL = "upsert_x_flowcell_pre_demux"
AGGREGATE_ILLUMINA_STATS = "aggregate_illumina_stats"
AGGREGATE_UNDETERMINED_STATS = "aggregate_undetermined_stats"
PRUNE_STALE_LANES = "prune_stale_lanes"


def lane_namespace(lane_id: str) -> str:
    """Return the step-ID prefix of one lane's own steps, e.g. lane_2."""
    return f"lane_{lane_id}"


def branch_namespace(lane_id: str, settings_index: str) -> str:
    """Return the step-ID prefix of one lane/settings branch, e.g. lane_2_settings_0."""
    return f"lane_{lane_id}_settings_{settings_index}"


def clean_lane_step_id(lane_id: str) -> str:
    """Return the ID of the step that removes a lane's earlier results."""
    return f"{lane_namespace(lane_id)}__clean_lane"


def _digest(data: Any) -> str:
    """Return a stable sha256 of JSON-serializable data."""
    text = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _group_by_lane(
    branches: Sequence[SamplesheetBranch],
) -> dict[str, list[SamplesheetBranch]]:
    """Group branches by lane, keeping the order lanes and branches come in."""
    lanes: dict[str, list[SamplesheetBranch]] = {}
    for branch in branches:
        lanes.setdefault(branch.lane_id, []).append(branch)
    return lanes


def build_demux_plan(
    *,
    plan_id: str,
    realm: str,
    canonical_fcid: str,
    runfolder_id: str,
    hpc_runfolder_path: str,
    samplesheets: list[Any],
    uploaded_lims_info: list[Any],
    metadata: Mapping[str, Any],
    branch_run_options: Mapping[tuple[str, str], NextflowRunOptions],
    branches: Sequence[SamplesheetBranch],
    branch_prerequisites: Sequence[str],
    artifact_base: Path,
) -> Plan:
    """Build the flowcell's demux plan from validated planning data.

    Shared steps come first: runfolder validation, the common demux
    directory and the x_flowcells metadata update. Each lane follows: a
    clean_lane step that depends on branch_prerequisites (include
    UPSERT_X_FLOWCELL there to make lanes wait for the metadata update), then
    every lane/settings branch of the lane as a chain of steps whose first
    step depends on the clean_lane step.

    A lane is the smallest unit that is re-demultiplexed. The lane's steps
    all carry the same lane_digest, a hash of everything its branches are
    planned from, so a change to any branch of the lane changes the
    fingerprint of every step in the lane: its earlier results are removed
    and all of its branches re-run, while unchanged lanes are reused.

    Args:
        plan_id: The combined plan ID.
        realm: The registered realm ID.
        canonical_fcid: Canonical flowcell ID; also the plan scope.
        runfolder_id: Runfolder name.
        hpc_runfolder_path: Absolute runfolder path.
        samplesheets: All demux_sample_info samplesheets, in source order.
        uploaded_lims_info: demux_sample_info uploaded LIMS entries.
        metadata: demux_sample_info metadata, used by upload_results.
        branch_run_options: The resolved per-branch Nextflow run options,
            keyed by (lane_id, settings_index); carried only by each
            branch's execute_demux step.
        branches: Validated branches, in plan order.
        branch_prerequisites: Shared step IDs each lane's clean_lane step
            depends on.

    Returns:
        The complete plan, run under continue_independent.
    """
    # Execution parameters only: provenance stays out so that revision or
    # trigger changes cannot invalidate reuse.
    common = {
        "canonical_flowcell_id": canonical_fcid,
        "runfolder_id": runfolder_id,
        "hpc_runfolder_path": hpc_runfolder_path,
    }
    runfolder = Path(hpc_runfolder_path)

    steps = [
        StepSpec(
            step_id=VALIDATE_RUNFOLDER,
            name="Validate Runfolder in HPC",
            fn_ref=f"{_PREFIX}.validate_runfolder",
            params={"scenario": dict(common)},
        ),
        StepSpec(
            step_id=SETUP_COMMON_DEMUX_DIR,
            name="Setup Common Demux Directory",
            fn_ref=f"{_PREFIX}.setup_common_demux_dir",
            params={"scenario": dict(common)},
        ),
        StepSpec(
            step_id=UPSERT_X_FLOWCELL,
            name="Upsert x_flowcells Pre-Demux Document",
            fn_ref=f"{_PREFIX}.upsert_x_flowcell_pre_demux",
            params={
                "scenario": {
                    **common,
                    "samplesheets": samplesheets,
                    "uploaded_lims_info": uploaded_lims_info,
                }
            },
            deps=[VALIDATE_RUNFOLDER],
            inputs={
                "run_info_xml": str(runfolder / RUN_INFO_XML),
                "run_parameters_xml": str(runfolder / RUN_PARAMETERS_XML),
            },
        ),
    ]

    lane_digests: dict[str, str] = {}
    for lane_id, lane_branches in _group_by_lane(branches).items():
        lane_digest = _digest(
            {
                "metadata": metadata,
                "branches": [
                    {
                        "settings_index": b.settings_index,
                        "payload": b.payload,
                        "nextflow": branch_run_options[
                            (b.lane_id, b.settings_index)
                        ].to_dict(),
                    }
                    for b in lane_branches
                ],
            }
        )
        lane_digests[lane_id] = lane_digest
        clean_lane = clean_lane_step_id(lane_id)
        steps.append(
            StepSpec(
                step_id=clean_lane,
                name=f"Clean Earlier Demux Results (lane {lane_id})",
                fn_ref=f"{_PREFIX}.clean_lane",
                params={
                    "scenario": {
                        **common,
                        "lane_id": lane_id,
                        "lane_digest": lane_digest,
                    }
                },
                deps=list(branch_prerequisites),
            )
        )
        for branch in lane_branches:
            ns = branch_namespace(branch.lane_id, branch.settings_index)
            label = f"lane {branch.lane_id}, settings {branch.settings_index}"
            scenario = {
                **common,
                "lane_id": branch.lane_id,
                "settings_index": branch.settings_index,
                "lane_digest": lane_digest,
                "samplesheet_payload": branch.payload,
                "demux_sample_info_doc": {"metadata": metadata},
            }

            samplesheet_path = Path(artifact_base / ns / "SampleSheet.csv")
            run_options = branch_run_options[(branch.lane_id, branch.settings_index)]
            # Relative outputs resolve inside each producer's own work directory.
            steps += [
                StepSpec(
                    step_id=f"{ns}__materialize_config",
                    name=f"Materialize Demux Config ({label})",
                    fn_ref=f"{_PREFIX}.materialize_extra_config",
                    params={"scenario": scenario},
                    deps=[clean_lane],
                    outputs={"demux_config": DEMUX_CONFIG_FILENAME},
                ),
                StepSpec(
                    step_id=f"{ns}__generate_samplesheet",
                    name=f"Generate SampleSheet.csv ({label})",
                    fn_ref=f"{_PREFIX}.generate_samplesheet",
                    params={"scenario": scenario, "samplesheet": samplesheet_path},
                    deps=[f"{ns}__materialize_config"],
                    outputs={"samplesheet": str(samplesheet_path)},
                ),
                StepSpec(
                    step_id=f"{ns}__execute_demux",
                    name=f"Execute Demux (nf-core/demultiplex) ({label})",
                    fn_ref=f"{_PREFIX}.execute_demux",
                    params={
                        "scenario": {
                            **scenario,
                            "samplesheet_path": samplesheet_path,
                            "nextflow": run_options.to_dict(),
                        },
                        "samplesheet": samplesheet_path,
                    },
                    deps=[f"{ns}__generate_samplesheet"],
                    inputs={"samplesheet": str(samplesheet_path)},
                ),
                StepSpec(
                    step_id=f"{ns}__collect_results",
                    name=f"Collect Results and Artifacts ({label})",
                    fn_ref=f"{_PREFIX}.collect_results",
                    params={"scenario": scenario},
                    deps=[f"{ns}__execute_demux"],
                ),
                StepSpec(
                    step_id=f"{ns}__symlink_fastqs",
                    name=f"Symlink Fastqs ({label})",
                    fn_ref=f"{_PREFIX}.symlink_fastqs",
                    params={"scenario": scenario},
                    deps=[f"{ns}__execute_demux"],
                ),
                StepSpec(
                    step_id=f"{ns}__upload_results",
                    name=f"Upload/Simulate Upload Results ({label})",
                    fn_ref=f"{_PREFIX}.upload_results",
                    params={"scenario": scenario},
                    deps=[f"{ns}__collect_results"],
                ),
            ]

    steps.append(
        StepSpec(
            step_id=PRUNE_STALE_LANES,
            name="Remove Demux Results of Unplanned Lanes",
            fn_ref=f"{_PREFIX}.prune_stale_lanes",
            params={"scenario": {**common, "lanes": list(lane_digests)}},
            deps=list(branch_prerequisites),
        )
    )

    # The aggregates read every lane's stored stats, so they re-run whenever
    # any lane, or the set of lanes, changes.
    aggregate_scenario = {**common, "plan_digest": _digest(lane_digests)}
    aggregate_deps = [
        PRUNE_STALE_LANES,
        *(
            f"{branch_namespace(b.lane_id, b.settings_index)}__upload_results"
            for b in branches
        ),
    ]
    steps.append(
        StepSpec(
            step_id=AGGREGATE_ILLUMINA_STATS,
            name="Aggregate Illumina Demux Stats",
            fn_ref=f"{_PREFIX}.aggregate_illumina_stats",
            params={"scenario": dict(aggregate_scenario)},
            # Common step: runs once after every branch's upload_results, and
            # recomputes the combined 'illumina' field from whatever is
            # currently stored for each lane (see aggregate_illumina_stats).
            deps=list(aggregate_deps),
        )
    )

    steps.append(
        StepSpec(
            step_id=AGGREGATE_UNDETERMINED_STATS,
            name="Aggregate Undetermined Barcode Stats",
            fn_ref=f"{_PREFIX}.aggregate_undetermined_stats",
            params={"scenario": dict(aggregate_scenario)},
            # Common step: runs once after every branch's upload_results, and
            # recomputes the combined 'Undetermined' field from whatever is
            # currently stored for each lane (see aggregate_undetermined_stats).
            deps=list(aggregate_deps),
        )
    )

    return Plan(
        plan_id=plan_id,
        realm=realm,
        scope={"kind": "flowcell", "id": canonical_fcid},
        steps=steps,
        failure_policy=CONTINUE_INDEPENDENT_POLICY,
    )
