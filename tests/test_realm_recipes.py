from pathlib import Path

from yggdrasil.flow.model import CONTINUE_INDEPENDENT_POLICY

from demux_realm.recipes import (
    AGGREGATE_ILLUMINA_STATS,
    AGGREGATE_UNDETERMINED_STATS,
    PRUNE_STALE_LANES,
    SETUP_COMMON_DEMUX_DIR,
    UPSERT_X_FLOWCELL,
    VALIDATE_RUNFOLDER,
    branch_namespace,
    build_demux_plan,
)
from demux_realm.nextflow import NextflowRunOptions
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
        branch_run_options={
            ("1", "0"): NextflowRunOptions(skip_tools=("samshee",)),
            ("3", "1"): NextflowRunOptions(skip_tools=("samshee",)),
        },
        branches=branches,
        branch_prerequisites=prerequisites,
        artifact_base=Path("/work/artifacts"),
    )


def test_branch_namespace_is_flat_and_readable():
    assert branch_namespace("2", "0") == "lane_2_settings_0"


def test_branch_run_options_are_carried_per_branch():
    plan = build_demux_plan(
        plan_id="dmx_realm:SC123:demux",
        realm="dmx_realm",
        canonical_fcid="SC123",
        runfolder_id="230314_A00000_0000_AXXXXX",
        hpc_runfolder_path=_RUNFOLDER,
        samplesheets=[{"lane": 3}, {"lane": 1}],
        uploaded_lims_info=[],
        metadata={"m": 1},
        branch_run_options={
            ("1", "0"): NextflowRunOptions(skip_tools=("samshee",)),
            ("3", "1"): NextflowRunOptions(),
        },
        branches=_BRANCHES,
        branch_prerequisites=(VALIDATE_RUNFOLDER,),
        artifact_base=Path("/work/artifacts"),
    )
    options = {
        s.step_id: s.params["scenario"]["nextflow"]
        for s in plan.steps
        if s.step_id.endswith("__execute_demux")
    }
    assert options == {
        "lane_1_settings_0__execute_demux": {"skip_tools": ["samshee"]},
        "lane_3_settings_1__execute_demux": {"skip_tools": []},
    }


def test_shared_steps_declare_the_xml_the_metadata_step_reads():
    plan = _plan()
    validate, setup, upsert = plan.steps[:3]

    assert plan.plan_id == "dmx_realm:SC123:demux"
    assert plan.scope == {"kind": "flowcell", "id": "SC123"}
    assert plan.failure_policy == CONTINUE_INDEPENDENT_POLICY
    assert (validate.step_id, validate.deps) == (VALIDATE_RUNFOLDER, [])
    assert (setup.step_id, setup.deps) == (SETUP_COMMON_DEMUX_DIR, [])
    assert (upsert.step_id, upsert.deps) == (UPSERT_X_FLOWCELL, [VALIDATE_RUNFOLDER])
    assert upsert.params["scenario"]["samplesheets"] == [{"lane": 3}, {"lane": 1}]
    assert upsert.inputs == {
        "run_info_xml": f"{_RUNFOLDER}/RunInfo.xml",
        "run_parameters_xml": f"{_RUNFOLDER}/RunParameters.xml",
    }
    assert validate.outputs == upsert.outputs == {}


def _step(plan, step_id):
    return next(s for s in plan.steps if s.step_id == step_id)


def _branch_steps(plan, namespace):
    return [s for s in plan.steps if s.step_id.startswith(f"{namespace}__")]


def test_each_branch_is_a_chain_of_namespaced_steps():
    steps = _branch_steps(_plan(), "lane_3_settings_1")

    assert [(s.step_id, s.fn_ref, s.deps) for s in steps] == [
        (
            "lane_3_settings_1__materialize_config",
            "demux_realm.steps.materialize_extra_config",
            ["lane_3__clean_lane"],
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
            "lane_3_settings_1__symlink_fastqs",
            "demux_realm.steps.symlink_fastqs",
            ["lane_3_settings_1__execute_demux"],
        ),
        (
            "lane_3_settings_1__upload_results",
            "demux_realm.steps.upload_results",
            ["lane_3_settings_1__collect_results"],
        ),
    ]
    sheet = "/work/artifacts/lane_3_settings_1/SampleSheet.csv"
    assert [s.outputs for s in steps] == [
        {"demux_config": "extra_config_demultiplex.config"},
        {"samplesheet": sheet},
        {},
        {},
        {},
        {},
    ]
    scenario = steps[0].params["scenario"]
    assert (scenario["lane_id"], scenario["settings_index"]) == ("3", "1")
    assert scenario["lane_digest"] == _step(_plan(), "lane_3__clean_lane").params[
        "scenario"
    ]["lane_digest"]
    assert scenario["samplesheet_payload"] == {"lane": 3}
    assert scenario["demux_sample_info_doc"] == {"metadata": {"m": 1}}
    # Only execute_demux gets the sheet location and the per-run options.
    execute = steps[2].params["scenario"]
    assert execute == {
        **scenario,
        "samplesheet_path": Path(sheet),
        "nextflow": {"skip_tools": ["samshee"]},
    }
    assert steps[2].inputs == {"samplesheet": sheet}
    assert steps[1].params == {"scenario": scenario, "samplesheet": Path(sheet)}
    assert all(
        s.params == {"scenario": scenario} for i, s in enumerate(steps) if i not in (1, 2)
    )
    assert all(s.name.endswith("(lane 3, settings 1)") for s in steps)


def test_metadata_prerequisite_is_the_authors_choice():
    plan = _plan(prerequisites=(VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL))
    clean_steps = [s for s in plan.steps if s.step_id.endswith("__clean_lane")]

    assert [s.deps for s in clean_steps] == [
        [VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL]
    ] * 2


def test_each_lane_starts_with_a_clean_lane_step():
    plan = _plan()
    clean = _step(plan, "lane_1__clean_lane")

    assert clean.fn_ref == "demux_realm.steps.clean_lane"
    assert clean.deps == [VALIDATE_RUNFOLDER]
    assert clean.params["scenario"]["lane_id"] == "1"
    step_ids = [s.step_id for s in plan.steps]
    assert step_ids.index("lane_1__clean_lane") < step_ids.index(
        "lane_1_settings_0__materialize_config"
    )


def test_lane_digest_follows_the_lanes_samplesheets_only():
    def digests(branches):
        plan = _plan(branches=branches)
        return {
            lane: _step(plan, f"lane_{lane}__clean_lane").params["scenario"][
                "lane_digest"
            ]
            for lane in ("1", "3")
        }

    before = digests(_BRANCHES)
    after = digests([_BRANCHES[0], SamplesheetBranch("3", "1", 0, {"lane": 3, "x": 1})])

    assert before["1"] != before["3"]
    assert after["1"] == before["1"]
    assert after["3"] != before["3"]


def test_branches_never_share_mutable_step_state():
    plan = _plan()
    first = _branch_steps(plan, "lane_1_settings_0")
    second = _branch_steps(plan, "lane_3_settings_1")

    for attribute in ("outputs", "deps"):
        assert not {id(getattr(s, attribute)) for s in first} & {
            id(getattr(s, attribute)) for s in second
        }


def test_unplanned_lanes_are_pruned_before_aggregation():
    plan = _plan()
    prune = _step(plan, PRUNE_STALE_LANES)

    assert prune.fn_ref == "demux_realm.steps.prune_stale_lanes"
    assert prune.params["scenario"]["lanes"] == ["1", "3"]
    assert prune.deps == [VALIDATE_RUNFOLDER]
    for step_id in (AGGREGATE_ILLUMINA_STATS, AGGREGATE_UNDETERMINED_STATS):
        assert _step(plan, step_id).deps == [
            PRUNE_STALE_LANES,
            "lane_1_settings_0__upload_results",
            "lane_3_settings_1__upload_results",
        ]


def test_aggregates_change_with_any_lane():
    def aggregate_params(branches):
        plan = _plan(branches=branches)
        return [
            _step(plan, step_id).params
            for step_id in (AGGREGATE_ILLUMINA_STATS, AGGREGATE_UNDETERMINED_STATS)
        ]

    before = aggregate_params(_BRANCHES)
    after = aggregate_params(
        [_BRANCHES[0], SamplesheetBranch("3", "1", 0, {"lane": 3, "x": 1})]
    )

    assert before == aggregate_params(_BRANCHES)
    assert all(b != a for b, a in zip(before, after))
