import copy
import json
from types import SimpleNamespace

import pytest
from realm_support import (
    REALM_ID,
    demux_sample_info_doc,
    flowcell_status_doc,
    lane_entry,
    planning_ctx,
)
from yggdrasil.flow.model import CONTINUE_INDEPENDENT_POLICY, Plan

from demux_realm.descriptor import (
    _build_demux_sample_info_payload,
    _build_demux_sample_info_scope,
    _build_flowcell_status_payload,
    _build_flowcell_status_scope,
)
from demux_realm.handler import DemuxHandler
from demux_realm.recipes import UPSERT_X_FLOWCELL, VALIDATE_RUNFOLDER

PLAN_ID = "dmx_realm:SC123:demux"
FLOWCELL_SCOPE = {"kind": "flowcell", "id": "SC123"}
BRANCH_STAGES = (
    "materialize_config",
    "generate_samplesheet",
    "execute_demux",
    "collect_results",
    "upload_results",
)


def branch_ids(namespace: str) -> list[str]:
    return [f"{namespace}__{stage}" for stage in BRANCH_STAGES]


@pytest.fixture(autouse=True)
def _no_hpc_base_path(monkeypatch):
    monkeypatch.delenv("DMX_HPC_BASE_PATH", raising=False)


@pytest.fixture
def handler():
    handler = DemuxHandler()
    handler.realm_id = REALM_ID
    return handler


async def plan_from_demux_change(handler, demux_doc, fc_doc):
    """Plan as the demux_sample_info watcher does for a change without its document."""
    event = SimpleNamespace(doc=None, id=demux_doc["_id"])
    ctx = planning_ctx(
        _build_demux_sample_info_scope(event)["id"], demux_doc=demux_doc, fc_doc=fc_doc
    )
    payload = {**_build_demux_sample_info_payload(event), "planning_ctx": ctx}
    return await handler.generate_plan_drafts(payload), ctx


async def plan_from_flowcell_change(handler, demux_doc, fc_doc):
    """Plan as the flowcell_status watcher does for a changed document."""
    event = SimpleNamespace(doc=fc_doc, id=fc_doc["_id"])
    ctx = planning_ctx(
        _build_flowcell_status_scope(event)["id"], demux_doc=demux_doc, fc_doc=fc_doc
    )
    payload = {**_build_flowcell_status_payload(event), "planning_ctx": ctx}
    return await handler.generate_plan_drafts(payload), ctx


async def plan_for(handler, samplesheets):
    """Return the only draft planned for samplesheets on a ready flowcell."""
    (draft,), _ = await plan_from_flowcell_change(
        handler, demux_sample_info_doc(samplesheets), flowcell_status_doc()
    )
    return draft


def step_dicts(plan: Plan) -> dict[str, dict]:
    return {step["step_id"]: step for step in plan.to_dict()["steps"]}


@pytest.mark.asyncio
async def test_canonical_matching_scope():
    event_sc = SimpleNamespace(doc={"flowcell_id": "SC123"}, id="uuid-1")
    event_asc = SimpleNamespace(doc={"flowcell_id": "ASC123"}, id="uuid-1")

    assert _build_demux_sample_info_scope(event_sc) == {
        "kind": "flowcell",
        "id": "SC123",
    }
    assert _build_demux_sample_info_scope(event_asc) == {
        "kind": "flowcell",
        "id": "SC123",
    }

    assert _build_flowcell_status_scope(event_sc) == {"kind": "flowcell", "id": "SC123"}
    assert _build_flowcell_status_scope(event_asc) == {
        "kind": "flowcell",
        "id": "SC123",
    }


@pytest.mark.asyncio
async def test_single_lane_yields_one_plan_with_seven_steps(handler):
    draft = await plan_for(handler, [lane_entry(1)])
    plan = draft.plan

    assert draft.auto_run is True
    assert draft.notes.startswith("Ready for demultiplexing SC123")
    assert plan.plan_id == PLAN_ID
    assert plan.realm == REALM_ID
    assert plan.scope == FLOWCELL_SCOPE
    assert plan.failure_policy == CONTINUE_INDEPENDENT_POLICY
    lane_1 = branch_ids("lane_1_settings_0")
    assert [s.step_id for s in plan.steps] == [
        VALIDATE_RUNFOLDER,
        UPSERT_X_FLOWCELL,
        *lane_1,
    ]
    assert {s.step_id: s.deps for s in plan.steps} == {
        VALIDATE_RUNFOLDER: [],
        UPSERT_X_FLOWCELL: [VALIDATE_RUNFOLDER],
        lane_1[0]: [VALIDATE_RUNFOLDER],
        lane_1[1]: [lane_1[0]],
        lane_1[2]: [lane_1[1]],
        lane_1[3]: [lane_1[2]],
        lane_1[4]: [lane_1[3]],
    }
    scenario = plan.steps[2].params["scenario"]
    assert scenario["hpc_runfolder_path"] == "/incoming/path/230314_A00000_0000_AXXXXX"
    assert scenario["lane_id"] == "1"
    assert scenario["settings_index"] == "0"


@pytest.mark.asyncio
async def test_lanes_and_settings_become_independent_branches(handler):
    entries = [
        lane_entry(2, settings_index=1),
        lane_entry(1),
        lane_entry(2, settings_index=0),
    ]
    draft = await plan_for(handler, entries)
    plan = draft.plan
    namespaces = ["lane_1_settings_0", "lane_2_settings_0", "lane_2_settings_1"]

    assert len(plan.steps) == 2 + 5 * len(namespaces)
    assert [s.step_id for s in plan.steps] == [
        VALIDATE_RUNFOLDER,
        UPSERT_X_FLOWCELL,
        *(step_id for ns in namespaces for step_id in branch_ids(ns)),
    ]
    assert [s.fn_ref for s in plan.steps].count(
        "demux_realm.steps.validate_runfolder"
    ) == 1
    deps = {s.step_id: s.deps for s in plan.steps}
    assert deps[UPSERT_X_FLOWCELL] == [VALIDATE_RUNFOLDER]
    for ns in namespaces:
        ids = branch_ids(ns)
        assert deps[ids[0]] == [VALIDATE_RUNFOLDER]
        for previous, current in zip(ids, ids[1:]):
            assert deps[current] == [previous]

    assert draft.preview["metadata_required"] is False
    assert [
        (b["lane_id"], b["settings_index"], b["source_index"])
        for b in draft.preview["branches"]
    ] == [("1", "0", 1), ("2", "0", 2), ("2", "1", 0)]
    assert draft.preview["step_count"] == len(plan.steps)


@pytest.mark.asyncio
async def test_both_triggers_produce_the_same_executable_plan(handler):
    demux_doc = demux_sample_info_doc([lane_entry(1), lane_entry(2)], "SC123")
    fc_doc = flowcell_status_doc("ASC123")
    originals = copy.deepcopy((demux_doc, fc_doc))

    (from_demux,), demux_ctx = await plan_from_demux_change(handler, demux_doc, fc_doc)
    (from_fc,), fc_ctx = await plan_from_flowcell_change(handler, demux_doc, fc_doc)

    # The demux trigger's scope held the document UUID, not the flowcell.
    assert demux_ctx.scope == {"kind": "flowcell", "id": "uuid-dsi-1"}
    assert from_demux.plan.scope == FLOWCELL_SCOPE
    assert from_demux.plan.to_dict() == from_fc.plan.to_dict()
    assert from_demux.auto_run is from_fc.auto_run is True
    assert from_demux.preview["triggering_source"] == "demux_sample_info"
    assert from_fc.preview["triggering_source"] == "flowcell_status"
    assert (demux_doc, fc_doc) == originals

    selector = {"flowcell_id": {"$in": ["SC123", "ASC123"]}}
    demux_ctx.db_mocks["demux_sample_info_db"].get.assert_awaited_once_with(
        "uuid-dsi-1"
    )
    demux_ctx.db_mocks["flowcell_status_db"].find_one.assert_awaited_once_with(selector)
    fc_ctx.db_mocks["demux_sample_info_db"].find_one.assert_awaited_once_with(selector)


@pytest.mark.asyncio
async def test_revisions_and_trigger_source_do_not_change_execution_parameters(
    handler,
):
    entries = [lane_entry(1), lane_entry(2)]
    (first,), _ = await plan_from_demux_change(
        handler, demux_sample_info_doc(entries, rev="1-a"), flowcell_status_doc()
    )
    (second,), _ = await plan_from_flowcell_change(
        handler,
        demux_sample_info_doc(entries, rev="2-b"),
        flowcell_status_doc(rev="9-z"),
    )

    assert first.plan.to_dict() == second.plan.to_dict()
    assert first.preview["sources"] != second.preview["sources"]


@pytest.mark.asyncio
async def test_source_order_does_not_change_branches(handler):
    entries = [
        lane_entry(1, samples=["S_C", "S_A", "S_B"]),
        lane_entry(2, settings_index=0),
        lane_entry(2, settings_index=1),
    ]
    forward = step_dicts((await plan_for(handler, entries)).plan)
    backward = step_dicts((await plan_for(handler, list(reversed(entries)))).plan)

    branch_steps = [step_id for step_id in forward if step_id.startswith("lane_")]
    assert branch_steps == [
        step_id for step_id in backward if step_id.startswith("lane_")
    ]
    for step_id in branch_steps:
        assert forward[step_id] == backward[step_id]

    payload = forward["lane_1_settings_0__generate_samplesheet"]["params"]["scenario"][
        "samplesheet_payload"
    ]
    assert [row["Sample_ID"] for row in payload["BCLConvert_Data"]] == [
        "S_C",
        "S_A",
        "S_B",
    ]
    # The metadata update keeps the source order it flattens rows in.
    assert forward[UPSERT_X_FLOWCELL]["params"]["scenario"]["samplesheets"] == entries
    assert backward[UPSERT_X_FLOWCELL]["params"]["scenario"]["samplesheets"] == list(
        reversed(entries)
    )


@pytest.mark.asyncio
async def test_branch_payload_change_leaves_other_branches_unchanged(handler):
    before = step_dicts((await plan_for(handler, [lane_entry(1), lane_entry(2)])).plan)
    after = step_dicts(
        (await plan_for(handler, [lane_entry(1), lane_entry(2, samples=["NEW"])])).plan
    )

    for step_id in [VALIDATE_RUNFOLDER, *branch_ids("lane_1_settings_0")]:
        assert before[step_id] == after[step_id]
    for step_id in branch_ids("lane_2_settings_0"):
        assert before[step_id]["params"] != after[step_id]["params"]


@pytest.mark.asyncio
async def test_explicit_settings_index_names_the_branch(handler):
    draft = await plan_for(handler, [lane_entry(1, settings_index="02")])

    assert [s.step_id for s in draft.plan.steps[2:]] == branch_ids("lane_1_settings_2")
    assert draft.plan.steps[2].params["scenario"]["settings_index"] == "2"


def _without_lane(entry):
    del entry["lane"]
    return entry


def _without_row_field(entry, field):
    del entry["BCLConvert_Data"][0][field]
    return entry


# Each proposal holds at least one valid entry; expected offending entry indices.
REJECTED_PROPOSALS = {
    "missing lane": ([lane_entry(1), _without_lane(lane_entry(2))], {1}),
    "boolean lane": ([lane_entry(1), {**lane_entry(2), "lane": True}], {1}),
    "named settings": ([lane_entry(1), lane_entry(2, settings_index="fast")], {1}),
    "ambiguous settings": (
        [lane_entry(1), lane_entry(2), lane_entry(2, samples=["X"])],
        {1, 2},
    ),
    "duplicate identity": (
        [
            lane_entry(1, settings_index=0),
            lane_entry("01", settings_index="0", row_lane="1"),
        ],
        {0, 1},
    ),
    "header not a mapping": (
        [lane_entry(1), {**lane_entry(2), "Header": ["FileFormatVersion", "2"]}],
        {1},
    ),
    "settings not a mapping": (
        [lane_entry(1), {**lane_entry(2), "raw_samplesheet_settings": "x,1"}],
        {1},
    ),
    "row lane differs": ([lane_entry(1), lane_entry(2, row_lane="1")], {1}),
    "missing row field": (
        [lane_entry(1), _without_row_field(lane_entry(2), "index")],
        {1},
    ),
    "entry not a mapping": ([lane_entry(1), "lane 2"], {1}),
    "samplesheets not a list": ({"lane_1": lane_entry(1)}, {None}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("samplesheets", "offending"),
    REJECTED_PROPOSALS.values(),
    ids=REJECTED_PROPOSALS.keys(),
)
async def test_invalid_entry_rejects_the_whole_proposal(
    handler, samplesheets, offending
):
    draft = await plan_for(handler, samplesheets)

    assert draft.auto_run is False
    assert draft.plan.steps == []
    assert draft.plan.plan_id == PLAN_ID
    assert draft.plan.scope == FLOWCELL_SCOPE
    assert draft.notes.startswith("Rejected:")
    assert draft.preview["status"] == "rejected"
    assert {issue["entry_index"] for issue in draft.preview["issues"]} == offending


@pytest.mark.asyncio
async def test_corrected_input_becomes_runnable(handler):
    rejected = await plan_for(handler, [lane_entry(1), lane_entry(2, row_lane="1")])
    corrected = await plan_for(handler, [lane_entry(1), lane_entry(2)])

    assert rejected.auto_run is False
    assert corrected.auto_run is True
    assert corrected.plan.plan_id == rejected.plan.plan_id == PLAN_ID
    assert len(corrected.plan.steps) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("samplesheets", [[], None])
async def test_missing_samplesheets_defers(handler, samplesheets):
    draft = await plan_for(handler, samplesheets)

    assert draft.auto_run is False
    assert draft.plan.steps == []
    assert draft.plan.plan_id == PLAN_ID
    assert draft.notes == "Deferred: demux_sample_info missing samplesheets."


@pytest.mark.asyncio
async def test_missing_counterpart_defers(handler):
    (draft,), _ = await plan_from_flowcell_change(handler, None, flowcell_status_doc())

    assert draft.auto_run is False
    assert draft.plan.steps == []
    assert draft.plan.plan_id == PLAN_ID
    assert draft.plan.scope == FLOWCELL_SCOPE
    assert "No demux_sample_info document found" in draft.notes


@pytest.mark.asyncio
async def test_transferred_to_hpc_absent_defers(handler):
    fc_doc = flowcell_status_doc()
    fc_doc["events"] = fc_doc["events"][1:]

    (draft,), _ = await plan_from_demux_change(
        handler, demux_sample_info_doc([lane_entry(1)]), fc_doc
    )

    assert draft.auto_run is False
    assert draft.plan.plan_id == PLAN_ID
    assert (
        draft.notes == "Deferred: flowcell_status missing 'transferred_to_hpc' event."
    )


@pytest.mark.asyncio
async def test_unknown_flowcell_defers_under_trigger_identity(handler):
    event = SimpleNamespace(doc=None, id="uuid-gone")
    ctx = planning_ctx("uuid-gone", demux_doc=None, fc_doc=None)
    payload = {**_build_demux_sample_info_payload(event), "planning_ctx": ctx}

    (draft,) = await handler.generate_plan_drafts(payload)

    assert draft.auto_run is False
    assert draft.plan.plan_id == "dmx_realm:uuid-gone"
    assert draft.plan.scope == {"kind": "flowcell", "id": "uuid-gone"}
    assert "not found or deleted" in draft.notes


@pytest.mark.asyncio
async def test_metadata_can_be_made_a_branch_prerequisite(handler, monkeypatch):
    monkeypatch.setattr(
        DemuxHandler, "branch_prerequisites", (VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL)
    )
    draft = await plan_for(handler, [lane_entry(1), lane_entry(2)])
    deps = {s.step_id: s.deps for s in draft.plan.steps}

    for ns in ("lane_1_settings_0", "lane_2_settings_0"):
        assert deps[branch_ids(ns)[0]] == [VALIDATE_RUNFOLDER, UPSERT_X_FLOWCELL]
    assert draft.preview["metadata_required"] is True


@pytest.mark.asyncio
async def test_hpc_base_path_prefixes_the_runfolder(handler, monkeypatch):
    monkeypatch.setenv("DMX_HPC_BASE_PATH", "/proj/hpc")
    draft = await plan_for(handler, [lane_entry(1)])

    assert (
        draft.plan.steps[0].params["scenario"]["hpc_runfolder_path"]
        == "/proj/hpc/incoming/path/230314_A00000_0000_AXXXXX"
    )


@pytest.mark.asyncio
async def test_plan_survives_persistence_round_trip(handler):
    plan = (await plan_for(handler, [lane_entry(1), lane_entry(2)])).plan

    restored = Plan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored.to_dict() == plan.to_dict()
    assert restored.failure_policy == CONTINUE_INDEPENDENT_POLICY
    specs = {s.step_id: s for s in restored.steps}
    assert specs["lane_2_settings_0__materialize_config"].outputs == {
        "demux_config": "extra_config_demultiplex.config"
    }
    assert specs["lane_2_settings_0__generate_samplesheet"].outputs == {
        "samplesheet": "SampleSheet.csv"
    }
    assert specs[UPSERT_X_FLOWCELL].inputs == {
        "run_info_xml": "/incoming/path/230314_A00000_0000_AXXXXX/RunInfo.xml",
        "run_parameters_xml": (
            "/incoming/path/230314_A00000_0000_AXXXXX/RunParameters.xml"
        ),
    }
    assert specs["lane_2_settings_0__upload_results"].deps == [
        "lane_2_settings_0__collect_results"
    ]
