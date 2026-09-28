from pathlib import Path

from yggdrasil.flow.model import StepSpec

from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    SAMPLESHEET_FILENAME,
)

_PREFIX = "demux_realm.steps"

VALIDATE_RUNFOLDER = "validate_runfolder"
UPSERT_X_FLOWCELL = "upsert_x_flowcell_pre_demux"

# (stage, step name, step function, required outputs) of every branch, in order.
# Relative outputs resolve inside each producer's own work directory.
_BRANCH_STAGES: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "materialize_config",
        "Materialize Demux Config",
        "materialize_extra_config",
        {"demux_config": DEMUX_CONFIG_FILENAME},
    ),
    (
        "generate_samplesheet",
        "Generate SampleSheet.csv",
        "generate_samplesheet",
        {"samplesheet": SAMPLESHEET_FILENAME},
    ),
    ("execute_demux", "Simulate Execute Demux (Nextflow)", "execute_demux", {}),
    ("collect_results", "Collect Results and Artifacts", "collect_results", {}),
    ("upload_results", "Upload/Simulate Upload Results", "upload_results", {}),
)


def branch_namespace(lane_id: str, settings_index: str) -> str:
    """Return the step-ID prefix of one lane/settings branch, e.g. lane_2_settings_0."""
    return f"lane_{lane_id}_settings_{settings_index}"


def initial_steps(validation_scenario: dict, metadata_scenario: dict) -> list[StepSpec]:
    """Shared flowcell steps: validate_runfolder → upsert_x_flowcell_pre_demux."""
    runfolder = Path(metadata_scenario["hpc_runfolder_path"])
    return [
        StepSpec(
            step_id=VALIDATE_RUNFOLDER,
            name="Validate Runfolder in HPC",
            fn_ref=f"{_PREFIX}.validate_runfolder",
            params={"scenario": validation_scenario},
        ),
        StepSpec(
            step_id=UPSERT_X_FLOWCELL,
            name="Upsert x_flowcells Pre-Demux Document",
            fn_ref=f"{_PREFIX}.upsert_x_flowcell_pre_demux",
            params={"scenario": metadata_scenario},
            deps=[VALIDATE_RUNFOLDER],
            inputs={
                "run_info_xml": str(runfolder / RUN_INFO_XML),
                "run_parameters_xml": str(runfolder / RUN_PARAMETERS_XML),
            },
        ),
    ]


def demux_pipeline(
    namespace: str, scenario: dict, first_deps: list[str], label: str
) -> list[StepSpec]:
    """One lane/settings branch: five steps, each depending on the one before.

    Args:
        namespace: Step-ID prefix unique to the branch (see branch_namespace).
        scenario: Parameters of the branch's steps.
        first_deps: Shared steps that must succeed before the branch starts.
        label: Branch description used in step names, e.g. "lane 2, settings 0".
    """
    steps: list[StepSpec] = []
    deps = list(first_deps)
    for stage, name, fn_name, outputs in _BRANCH_STAGES:
        step_id = f"{namespace}__{stage}"
        steps.append(
            StepSpec(
                step_id=step_id,
                name=f"{name} ({label})",
                fn_ref=f"{_PREFIX}.{fn_name}",
                params={"scenario": scenario},
                deps=deps,
                outputs=dict(outputs),
            )
        )
        deps = [step_id]
    return steps
