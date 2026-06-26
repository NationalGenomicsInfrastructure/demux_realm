from yggdrasil.flow.model import StepSpec

_PREFIX = "demux_realm.steps"


def initial_steps(scenario: dict) -> list[StepSpec]:
    """Per-flowcell common plan: validate_runfolder → upsert_x_flowcell_pre_demux."""
    return [
        StepSpec(
            step_id="validate_runfolder",
            name="Validate Runfolder in HPC",
            fn_ref=f"{_PREFIX}.validate_runfolder",
            params={"scenario": scenario},
        ),
        StepSpec(
            step_id="upsert_x_flowcell_pre_demux",
            name="Upsert x_flowcells Pre-Demux Document",
            fn_ref=f"{_PREFIX}.upsert_x_flowcell_pre_demux",
            params={"scenario": scenario},
            deps=["validate_runfolder"],
        ),
    ]


def demux_pipeline(scenario: dict) -> list[StepSpec]:
    """
    Per-lane/settings demux steps.
    validate_runfolder is included here as well as in the common plan — lane plans
    are independent and cannot rely on the common plan having already executed.
    """
    return [
        StepSpec(
            step_id="validate_runfolder",
            name="Validate Runfolder in HPC",
            fn_ref=f"{_PREFIX}.validate_runfolder",
            params={"scenario": scenario},
        ),
        StepSpec(
            step_id="materialize_config",
            name="Materialize Demux Config",
            fn_ref=f"{_PREFIX}.materialize_extra_config",
            params={"scenario": scenario},
            deps=["validate_runfolder"],
        ),
        StepSpec(
            step_id="generate_samplesheet",
            name="Generate SampleSheet.csv",
            fn_ref=f"{_PREFIX}.generate_samplesheet",
            params={"scenario": scenario},
            deps=["materialize_config"],
        ),
        StepSpec(
            step_id="execute_demux",
            name="Simulate Execute Demux (Nextflow)",
            fn_ref=f"{_PREFIX}.execute_demux",
            params={"scenario": scenario},
            deps=["generate_samplesheet"],
        ),
        StepSpec(
            step_id="collect_results",
            name="Collect Results and Artifacts",
            fn_ref=f"{_PREFIX}.collect_results",
            params={"scenario": scenario},
            deps=["execute_demux"],
        ),
        StepSpec(
            step_id="upload_results",
            name="Upload/Simulate Upload Results",
            fn_ref=f"{_PREFIX}.upload_results",
            params={"scenario": scenario},
            deps=["collect_results"],
        ),
    ]
