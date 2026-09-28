from demux_realm.recipes import (
    UPSERT_X_FLOWCELL,
    VALIDATE_RUNFOLDER,
    branch_namespace,
    demux_pipeline,
    initial_steps,
)

_RUNFOLDER = "/incoming/path/230314_A00000_0000_AXXXXX"


def test_branch_namespace_is_flat_and_readable():
    assert branch_namespace("2", "0") == "lane_2_settings_0"


def test_initial_steps_declare_the_xml_the_metadata_step_reads():
    validation = {"hpc_runfolder_path": _RUNFOLDER}
    metadata = {"hpc_runfolder_path": _RUNFOLDER, "samplesheets": []}

    validate, upsert = initial_steps(validation, metadata)

    assert (validate.step_id, validate.deps) == (VALIDATE_RUNFOLDER, [])
    assert validate.params == {"scenario": validation}
    assert (upsert.step_id, upsert.deps) == (UPSERT_X_FLOWCELL, [VALIDATE_RUNFOLDER])
    assert upsert.params == {"scenario": metadata}
    assert upsert.inputs == {
        "run_info_xml": f"{_RUNFOLDER}/RunInfo.xml",
        "run_parameters_xml": f"{_RUNFOLDER}/RunParameters.xml",
    }
    assert validate.outputs == upsert.outputs == {}


def test_demux_pipeline_chains_five_namespaced_steps():
    first_deps = [VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL]
    scenario = {"lane_id": "3", "settings_index": "1"}

    steps = demux_pipeline(
        "lane_3_settings_1", scenario, first_deps, "lane 3, settings 1"
    )

    assert [(s.step_id, s.fn_ref, s.deps) for s in steps] == [
        (
            "lane_3_settings_1__materialize_config",
            "demux_realm.steps.materialize_extra_config",
            first_deps,
        ),
        (
            "lane_3_settings_1__generate_samplesheet",
            "demux_realm.steps.generate_samplesheet",
            ["lane_3_settings_1__materialize_config"],
        ),
        (
            "lane_3_settings_1__execute_demux",
            "demux_realm.steps.execute_demux",
            ["lane_3_settings_1__generate_samplesheet"],
        ),
        (
            "lane_3_settings_1__collect_results",
            "demux_realm.steps.collect_results",
            ["lane_3_settings_1__execute_demux"],
        ),
        (
            "lane_3_settings_1__upload_results",
            "demux_realm.steps.upload_results",
            ["lane_3_settings_1__collect_results"],
        ),
    ]
    assert [s.outputs for s in steps] == [
        {"demux_config": "extra_config_demultiplex.config"},
        {"samplesheet": "SampleSheet.csv"},
        {},
        {},
        {},
    ]
    assert all(s.params == {"scenario": scenario} for s in steps)
    assert all(s.name.endswith("(lane 3, settings 1)") for s in steps)
    assert steps[0].deps is not first_deps


def test_branches_never_share_step_specs():
    first = demux_pipeline("lane_1_settings_0", {}, [VALIDATE_RUNFOLDER], "a")
    second = demux_pipeline("lane_2_settings_0", {}, [VALIDATE_RUNFOLDER], "b")

    assert not {id(s) for s in first} & {id(s) for s in second}
    assert not {id(s.outputs) for s in first} & {id(s.outputs) for s in second}
