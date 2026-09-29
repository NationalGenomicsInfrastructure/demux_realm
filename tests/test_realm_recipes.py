from yggdrasil.flow.model import CONTINUE_INDEPENDENT_POLICY

from demux_realm.recipes import (
    UPSERT_X_FLOWCELL,
    VALIDATE_RUNFOLDER,
    branch_namespace,
    build_demux_plan,
)
from demux_realm.utils import SamplesheetBranch

_RUNFOLDER = "/incoming/path/230314_A00000_0000_AXXXXX"
_BRANCHES = [
    SamplesheetBranch("1", "0", 1, {"lane": 1}),
    SamplesheetBranch("3", "1", 0, {"lane": 3}),
]


def _plan(branches=_BRANCHES, prerequisites=(VALIDATE_RUNFOLDER,)):
    return build_demux_plan(
        plan_id="dmx_realm:SC123:demux",
        realm="dmx_realm",
        canonical_fcid="SC123",
        runfolder_id="230314_A00000_0000_AXXXXX",
        hpc_runfolder_path=_RUNFOLDER,
        samplesheets=[{"lane": 3}, {"lane": 1}],
        uploaded_lims_info=[],
        metadata={"m": 1},
        branches=branches,
        branch_prerequisites=prerequisites,
    )


def test_branch_namespace_is_flat_and_readable():
    assert branch_namespace("2", "0") == "lane_2_settings_0"


def test_shared_steps_declare_the_xml_the_metadata_step_reads():
    plan = _plan()
    validate, upsert = plan.steps[:2]

    assert plan.plan_id == "dmx_realm:SC123:demux"
    assert plan.scope == {"kind": "flowcell", "id": "SC123"}
    assert plan.failure_policy == CONTINUE_INDEPENDENT_POLICY
    assert (validate.step_id, validate.deps) == (VALIDATE_RUNFOLDER, [])
    assert (upsert.step_id, upsert.deps) == (UPSERT_X_FLOWCELL, [VALIDATE_RUNFOLDER])
    assert upsert.params["scenario"]["samplesheets"] == [{"lane": 3}, {"lane": 1}]
    assert upsert.inputs == {
        "run_info_xml": f"{_RUNFOLDER}/RunInfo.xml",
        "run_parameters_xml": f"{_RUNFOLDER}/RunParameters.xml",
    }
    assert validate.outputs == upsert.outputs == {}


def test_each_branch_is_a_chain_of_five_namespaced_steps():
    steps = _plan().steps[7:]

    assert [(s.step_id, s.fn_ref, s.deps) for s in steps] == [
        (
            "lane_3_settings_1__materialize_config",
            "demux_realm.steps.materialize_extra_config",
            [VALIDATE_RUNFOLDER],
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
    scenario = steps[0].params["scenario"]
    assert (scenario["lane_id"], scenario["settings_index"]) == ("3", "1")
    assert scenario["samplesheet_payload"] == {"lane": 3}
    assert scenario["demux_sample_info_doc"] == {"metadata": {"m": 1}}
    assert all(s.params == {"scenario": scenario} for s in steps)
    assert all(s.name.endswith("(lane 3, settings 1)") for s in steps)


def test_metadata_prerequisite_is_the_authors_choice():
    plan = _plan(prerequisites=(VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL))
    first_steps = [s for s in plan.steps if s.step_id.endswith("__materialize_config")]

    assert [s.deps for s in first_steps] == [
        [VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL]
    ] * 2


def test_branches_never_share_mutable_step_state():
    steps = _plan().steps
    first, second = steps[2:7], steps[7:]

    for attribute in ("outputs", "deps"):
        assert not {id(getattr(s, attribute)) for s in first} & {
            id(getattr(s, attribute)) for s in second
        }
