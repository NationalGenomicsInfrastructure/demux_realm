from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yggdrasil.flow.model import CONTINUE_INDEPENDENT_POLICY, Plan, StepSpec

from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    SAMPLESHEET_FILENAME,
    SamplesheetBranch,
)

_PREFIX = "demux_realm.steps"

VALIDATE_RUNFOLDER = "validate_runfolder"
UPSERT_X_FLOWCELL = "upsert_x_flowcell_pre_demux"


def branch_namespace(lane_id: str, settings_index: str) -> str:
    """Return the step-ID prefix of one lane/settings branch, e.g. lane_2_settings_0."""
    return f"lane_{lane_id}_settings_{settings_index}"


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
    branches: Sequence[SamplesheetBranch],
    branch_prerequisites: Sequence[str],
) -> Plan:
    """Build the flowcell's demux plan from validated planning data.

    Two shared steps come first: runfolder validation, then the x_flowcells
    metadata update. Each lane/settings branch follows as a chain of five
    steps whose first step depends on branch_prerequisites; include
    UPSERT_X_FLOWCELL there to make branches wait for the metadata update.

    Args:
        plan_id: The combined plan ID.
        realm: The registered realm ID.
        canonical_fcid: Canonical flowcell ID; also the plan scope.
        runfolder_id: Runfolder name.
        hpc_runfolder_path: Absolute runfolder path.
        samplesheets: All demux_sample_info samplesheets, in source order.
        uploaded_lims_info: demux_sample_info uploaded LIMS entries.
        metadata: demux_sample_info metadata, used by upload_results.
        branches: Validated branches, in plan order.
        branch_prerequisites: Shared step IDs each branch's first step depends on.

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

    for branch in branches:
        ns = branch_namespace(branch.lane_id, branch.settings_index)
        label = f"lane {branch.lane_id}, settings {branch.settings_index}"
        scenario = {
            **common,
            "lane_id": branch.lane_id,
            "settings_index": branch.settings_index,
            "samplesheet_payload": branch.payload,
            "demux_sample_info_doc": {"metadata": metadata},
        }
        # Relative outputs resolve inside each producer's own work directory.
        steps += [
            StepSpec(
                step_id=f"{ns}__materialize_config",
                name=f"Materialize Demux Config ({label})",
                fn_ref=f"{_PREFIX}.materialize_extra_config",
                params={"scenario": scenario},
                deps=list(branch_prerequisites),
                outputs={"demux_config": DEMUX_CONFIG_FILENAME},
            ),
            StepSpec(
                step_id=f"{ns}__generate_samplesheet",
                name=f"Generate SampleSheet.csv ({label})",
                fn_ref=f"{_PREFIX}.generate_samplesheet",
                params={"scenario": scenario},
                deps=[f"{ns}__materialize_config"],
                outputs={"samplesheet": SAMPLESHEET_FILENAME},
            ),
            StepSpec(
                step_id=f"{ns}__execute_demux",
                name=f"Simulate Execute Demux (Nextflow) ({label})",
                fn_ref=f"{_PREFIX}.execute_demux",
                params={"scenario": scenario},
                deps=[f"{ns}__generate_samplesheet"],
            ),
            StepSpec(
                step_id=f"{ns}__collect_results",
                name=f"Collect Results and Artifacts ({label})",
                fn_ref=f"{_PREFIX}.collect_results",
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

    return Plan(
        plan_id=plan_id,
        realm=realm,
        scope={"kind": "flowcell", "id": canonical_fcid},
        steps=steps,
        failure_policy=CONTINUE_INDEPENDENT_POLICY,
    )
