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


#: Input key under which every step of a lane declares its cleanup token.
LANE_CLEANUP_INPUT = "lane_cleanup"


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
    step depends on the clean_lane step, and finally the lane's post-demux
    aggregate steps, which wait for every branch of the lane.

    A lane is the smallest unit that is re-demultiplexed. The clean_lane
    step carries a lane_digest, a hash of everything the lane's branches are
    planned from, so a change to any branch of the lane re-runs it. It then
    writes a new cleanup token, which every later step of the lane declares
    as an input, so all of them run again too; unchanged lanes are reused.

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

    lanes = _group_by_lane(branches)
    # Lane level: one clean_lane step, then the lane's branches, then the
    # lane's post-demux steps. Everything in this loop shares the lane's
    # cleanup token, so a cleanup re-runs the whole lane.
    for lane_id, lane_branches in lanes.items():
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
        clean_lane = clean_lane_step_id(lane_id)
        cleanup_token = str(artifact_base / lane_namespace(lane_id) / "cleanup.token")
        lane_inputs = {LANE_CLEANUP_INPUT: cleanup_token}
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
                        "cleanup_token": cleanup_token,
                    }
                },
                deps=list(branch_prerequisites),
                outputs={"cleanup_token": cleanup_token},
            )
        )
        # Settings level: one chain of steps per lane/settings branch.
        for branch in lane_branches:
            ns = branch_namespace(branch.lane_id, branch.settings_index)
            label = f"lane {branch.lane_id}, settings {branch.settings_index}"
            scenario = {
                **common,
                "lane_id": branch.lane_id,
                "settings_index": branch.settings_index,
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
                    inputs=dict(lane_inputs),
                    outputs={"demux_config": DEMUX_CONFIG_FILENAME},
                ),
                StepSpec(
                    step_id=f"{ns}__generate_samplesheet",
                    name=f"Generate SampleSheet.csv ({label})",
                    fn_ref=f"{_PREFIX}.generate_samplesheet",
                    params={"scenario": scenario, "samplesheet": samplesheet_path},
                    deps=[f"{ns}__materialize_config"],
                    inputs=dict(lane_inputs),
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
                    inputs={**lane_inputs, "samplesheet": str(samplesheet_path)},
                ),
                StepSpec(
                    step_id=f"{ns}__collect_results",
                    name=f"Collect Results and Artifacts ({label})",
                    fn_ref=f"{_PREFIX}.collect_results",
                    params={"scenario": scenario},
                    deps=[f"{ns}__execute_demux"],
                    inputs=dict(lane_inputs),
                ),
                StepSpec(
                    step_id=f"{ns}__symlink_fastqs",
                    name=f"Symlink Fastqs ({label})",
                    fn_ref=f"{_PREFIX}.symlink_fastqs",
                    params={
                        "scenario": {
                            **scenario,
                            "lane_settings_count": len(lane_branches),
                        }
                    },
                    deps=[f"{ns}__execute_demux"],
                    inputs=dict(lane_inputs),
                ),
                StepSpec(
                    step_id=f"{ns}__upload_results",
                    name=f"Upload/Simulate Upload Results ({label})",
                    fn_ref=f"{_PREFIX}.upload_results",
                    params={"scenario": scenario},
                    deps=[f"{ns}__collect_results"],
                    inputs=dict(lane_inputs),
                ),
            ]
        # End of the settings level: every branch of this lane is planned.

        # Post-demux lane steps, back at the lane level: run once every branch
        # of the lane uploaded, and recompute the combined fields from
        # whatever is stored for every lane (see aggregate_illumina_stats).
        lane_scenario = {
            **common,
            "lane_id": lane_id,
            "settings_indices": [b.settings_index for b in lane_branches],
        }
        uploads = [
            f"{branch_namespace(b.lane_id, b.settings_index)}__upload_results"
            for b in lane_branches
        ]
        steps += [
            StepSpec(
                step_id=f"{lane_namespace(lane_id)}__aggregate_illumina_stats",
                name=f"Aggregate Illumina Demux Stats (lane {lane_id})",
                fn_ref=f"{_PREFIX}.aggregate_illumina_stats",
                params={"scenario": dict(lane_scenario)},
                deps=list(uploads),
                inputs=dict(lane_inputs),
            ),
            StepSpec(
                step_id=f"{lane_namespace(lane_id)}__aggregate_undetermined_stats",
                name=f"Aggregate Undetermined Barcode Stats (lane {lane_id})",
                fn_ref=f"{_PREFIX}.aggregate_undetermined_stats",
                params={"scenario": dict(lane_scenario)},
                deps=list(uploads),
                inputs=dict(lane_inputs),
            ),
        ]
    # End of the lane level: every lane is planned. What follows is
    # flowcell-wide again and does not depend on any lane.

    steps.append(
        StepSpec(
            step_id=PRUNE_STALE_LANES,
            name="Remove Demux Results of Unplanned Lanes",
            fn_ref=f"{_PREFIX}.prune_stale_lanes",
            params={"scenario": {**common, "lanes": list(lanes)}},
            deps=list(branch_prerequisites),
        )
    )

    return Plan(
        plan_id=plan_id,
        realm=realm,
        scope={"kind": "flowcell", "id": canonical_fcid},
        steps=steps,
        failure_policy=CONTINUE_INDEPENDENT_POLICY,
    )
