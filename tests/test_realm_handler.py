from unittest.mock import AsyncMock, MagicMock

import pytest

from dataflow_demux.yggdrasil_realm.descriptor import (
    _build_demux_sample_info_scope,
    _build_flowcell_status_scope,
)
from dataflow_demux.yggdrasil_realm.handler import DemuxHandler

# Minimal valid samplesheet entry reused across tests.
_VALID_LANE_ENTRY = {
    "lane": 1,
    "Header": {"FileFormatVersion": "2"},
    "raw_samplesheet_settings": {"CreateFastqForIndexReads": "1"},
    "BCLConvert_Data": [
        {
            "Lane": "1",
            "Sample_ID": "S1",
            "Sample_Name": "Sample1",
            "index": "AAAA",
            "Sample_Project": "P001",
        }
    ],
}

# Minimal yfc_doc with both required events, reused across tests.
_YFC_DOC = {
    "_id": "uuid-yfc-sc123",
    "flowcell_id": "SC123",
    "runfolder_id": "230314_A00000_0000_AXXXXX",
    "events": [
        {"event_type": "transferred_to_hpc"},
        {
            "event_type": "final_transfer_started",
            "data": {"destination_path": "/incoming/path"},
        },
    ],
}


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.scope = {"kind": "flowcell", "id": "SC123"}

    ctx.db_mocks = {
        "demux_sample_info_db": AsyncMock(),
        "flowcell_status_db": AsyncMock(),
    }

    def couchdb_mock(db_name):
        return ctx.db_mocks[db_name]

    ctx.data.couchdb.side_effect = couchdb_mock
    return ctx


@pytest.mark.asyncio
async def test_canonical_matching_scope():
    event_sc = MagicMock(doc={"flowcell_id": "SC123"})
    event_asc = MagicMock(doc={"flowcell_id": "ASC123"})

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
async def test_identical_plan_reconstruction(mock_ctx):
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    demux_doc = {
        "_id": "uuid123",
        "flowcell_id": "SC123",
        "samplesheets": [
            dict(_VALID_LANE_ENTRY)
        ],  # one entry, lane=1, no settings_index
    }

    # demux_sample_info trigger: fetches demux_doc by _id (get), then yfc_doc by find_one
    mock_ctx.db_mocks["demux_sample_info_db"].get.return_value = demux_doc
    mock_ctx.db_mocks["flowcell_status_db"].find_one.return_value = _YFC_DOC

    # flowcell_status trigger: yfc_doc comes from payload["doc"], fetches demux_doc by find_one
    mock_ctx.db_mocks["demux_sample_info_db"].find_one.return_value = demux_doc

    payload1 = {"source": "demux_sample_info", "planning_ctx": mock_ctx}
    plan1 = await handler.generate_plan_drafts(payload1)

    payload2 = {"source": "flowcell_status", "doc": _YFC_DOC, "planning_ctx": mock_ctx}
    plan2 = await handler.generate_plan_drafts(payload2)

    # Two plans returned: [common, lane]
    assert len(plan1) == 2
    assert len(plan2) == 2

    # plan[0] = common plan: 2 steps (validate_runfolder + upsert_x_flowcell_pre_demux)
    assert plan1[0].auto_run is True
    assert len(plan1[0].plan.steps) == 2
    assert plan1[0].plan.plan_id == "dmx_realm:SC123:init"
    assert "lane_id" not in plan1[0].preview["scenario"]

    # plan[1] = lane plan: 5 steps, correct ID
    assert plan1[1].auto_run is True
    assert len(plan1[1].plan.steps) == 6
    assert "lane_1:settings_0" in plan1[1].plan.plan_id

    # scenario checks (lane plan preview)
    p1_scenario = plan1[1].preview["scenario"]
    p2_scenario = plan2[1].preview["scenario"]
    assert p1_scenario["canonical_flowcell_id"] == "SC123"
    assert (
        p1_scenario["hpc_runfolder_path"] == "/incoming/path/230314_A00000_0000_AXXXXX"
    )
    assert p1_scenario["lane_id"] == "1"
    assert p1_scenario["settings_index"] == "0"

    # Only the triggering source should differ
    assert p1_scenario["triggering_source"] == "demux_sample_info"
    assert p2_scenario["triggering_source"] == "flowcell_status"


@pytest.mark.asyncio
async def test_missing_counterpart_defer(mock_ctx):
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    # Only YFC is present, Demux missing.
    # flowcell_status trigger: yfc_doc comes from payload["doc"], demux_doc is None.
    yfc_doc = {"_id": "uuid-yfc-sc123", "flowcell_id": "SC123"}
    mock_ctx.db_mocks["demux_sample_info_db"].find_one.return_value = None

    payload = {"source": "flowcell_status", "doc": yfc_doc, "planning_ctx": mock_ctx}
    plan = await handler.generate_plan_drafts(payload)

    assert plan[0].auto_run is False
    assert "No demux_sample_info document found" in plan[0].notes


@pytest.mark.asyncio
async def test_transferred_to_hpc_absent_defer(mock_ctx):
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    # Both present, but YFC missing transferred_to_hpc event.
    # demux_sample_info trigger: fetches demux_doc by _id (get), then yfc_doc by find_one.
    yfc_doc = {
        "_id": "uuid-yfc-sc123",
        "flowcell_id": "SC123",
        "runfolder_id": "230314_A00000_0000_AXXXXX",
        "events": [
            {
                "event_type": "final_transfer_started",
                "data": {"destination_path": "/incoming/path"},
            }
        ],  # missing "transferred_to_hpc"
    }
    demux_doc = {
        "_id": "uuid123",
        "flowcell_id": "SC123",
        "samplesheets": [{"Sample_ID": "S1"}],
    }

    mock_ctx.db_mocks["demux_sample_info_db"].get.return_value = demux_doc
    mock_ctx.db_mocks["flowcell_status_db"].find_one.return_value = yfc_doc

    payload = {"source": "demux_sample_info", "planning_ctx": mock_ctx}
    plan = await handler.generate_plan_drafts(payload)

    assert plan[0].auto_run is False
    assert (
        "Deferred: flowcell_status missing 'transferred_to_hpc' event" in plan[0].notes
    )
    assert plan[0].auto_run is False
    assert (
        "Deferred: flowcell_status missing 'transferred_to_hpc' event" in plan[0].notes
    )


@pytest.mark.asyncio
async def test_settings_index_explicit(mock_ctx):
    """Single entry with explicit settings_index=2 → plan_id ends in lane_1:settings_2."""
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    entry = {**_VALID_LANE_ENTRY, "settings_index": 2}
    demux_doc = {"_id": "uuid123", "flowcell_id": "SC123", "samplesheets": [entry]}

    mock_ctx.db_mocks["demux_sample_info_db"].get.return_value = demux_doc
    mock_ctx.db_mocks["flowcell_status_db"].find_one.return_value = _YFC_DOC

    drafts = await handler.generate_plan_drafts(
        {"source": "demux_sample_info", "planning_ctx": mock_ctx}
    )

    assert len(drafts) == 2
    lane_draft = drafts[1]
    assert "lane_1:settings_2" in lane_draft.plan.plan_id
    assert lane_draft.preview["scenario"]["settings_index"] == "2"


@pytest.mark.asyncio
async def test_settings_index_missing_single_entry(mock_ctx):
    """Single entry without settings_index → defaults to settings_0."""
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    demux_doc = {
        "_id": "uuid123",
        "flowcell_id": "SC123",
        "samplesheets": [dict(_VALID_LANE_ENTRY)],  # no settings_index
    }

    mock_ctx.db_mocks["demux_sample_info_db"].get.return_value = demux_doc
    mock_ctx.db_mocks["flowcell_status_db"].find_one.return_value = _YFC_DOC

    drafts = await handler.generate_plan_drafts(
        {"source": "demux_sample_info", "planning_ctx": mock_ctx}
    )

    assert len(drafts) == 2
    assert "lane_1:settings_0" in drafts[1].plan.plan_id


@pytest.mark.asyncio
async def test_settings_index_ambiguous(mock_ctx):
    """Two entries for lane 1, both missing settings_index → common plan + deferred."""
    handler = DemuxHandler()
    handler.realm_id = "dmx_realm"

    # Two entries for the same lane, neither has settings_index — ambiguous.
    demux_doc = {
        "_id": "uuid123",
        "flowcell_id": "SC123",
        "samplesheets": [dict(_VALID_LANE_ENTRY), dict(_VALID_LANE_ENTRY)],
    }

    mock_ctx.db_mocks["demux_sample_info_db"].get.return_value = demux_doc
    mock_ctx.db_mocks["flowcell_status_db"].find_one.return_value = _YFC_DOC

    drafts = await handler.generate_plan_drafts(
        {"source": "demux_sample_info", "planning_ctx": mock_ctx}
    )

    # Common plan is always returned; lane slot becomes a deferred.
    assert len(drafts) == 2
    assert drafts[0].auto_run is True  # common plan
    assert drafts[0].plan.plan_id == "dmx_realm:SC123:init"
    assert drafts[1].auto_run is False  # lane deferred
    assert "No valid samplesheet entries" in drafts[1].notes
