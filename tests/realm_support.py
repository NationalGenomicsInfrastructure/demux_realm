"""Source-document builders and a planning context shared by the realm tests."""

import textwrap
from typing import Any
from unittest.mock import AsyncMock, MagicMock

REALM_ID = "dmx_realm"

RUN_INFO_XML_TEXT = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunInfo Version="7">
      <Run Id="20260312_SH01140_0005_ASC2177698-SC3" Number="5">
        <Flowcell>BXA66715-2010</Flowcell>
        <Instrument>SH01140</Instrument>
        <Date>2026-03-12T14:25:57Z</Date>
        <Reads>
          <Read Number="1" NumCycles="301" IsIndexedRead="N" IsReverseComplement="N"/>
        </Reads>
        <FlowcellLayout LaneCount="2" SurfaceCount="1" SwathCount="9" TileCount="2"/>
      </Run>
    </RunInfo>
""")

RUN_PARAMETERS_XML_TEXT = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunParameters>
      <InstrumentType>MiSeqi100Plus</InstrumentType>
      <InstrumentSerialNumber>SH01140</InstrumentSerialNumber>
      <RunId>20260312_SH01140_0005_ASC2177698-SC3</RunId>
      <RunCounter>5</RunCounter>
    </RunParameters>
""")


def lane_entry(
    lane: Any,
    *,
    settings_index: Any = None,
    samples: list[str] | None = None,
    row_lane: Any = None,
) -> dict[str, Any]:
    """Return a valid demux_sample_info samplesheet entry for one lane."""
    samples = samples or [f"L{lane}S{settings_index or 0}_1"]
    entry: dict[str, Any] = {
        "lane": lane,
        "Header": {"FileFormatVersion": "2"},
        "raw_samplesheet_settings": {"CreateFastqForIndexReads": "1"},
        "BCLConvert_Data": [
            {
                "Lane": str(lane) if row_lane is None else row_lane,
                "Sample_ID": sample,
                "Sample_Name": sample,
                "index": "ACGTACGT",
                "Sample_Project": "P001",
            }
            for sample in samples
        ],
    }
    if settings_index is not None:
        entry["settings_index"] = settings_index
    return entry


def flowcell_status_doc(
    flowcell_id: str = "SC123",
    *,
    destination_path: str = "/incoming/path",
    runfolder_id: str = "230314_A00000_0000_AXXXXX",
    rev: str = "1-fc",
) -> dict[str, Any]:
    """Return a flowcell_status document that is ready for demultiplexing."""
    return {
        "_id": "uuid-fc-1",
        "_rev": rev,
        "flowcell_id": flowcell_id,
        "runfolder_id": runfolder_id,
        "events": [
            {"event_type": "transferred_to_hpc"},
            {
                "event_type": "final_transfer_started",
                "data": {"destination_path": destination_path},
            },
        ],
    }


def demux_sample_info_doc(
    samplesheets: Any,
    flowcell_id: str = "SC123",
    *,
    rev: str = "1-dsi",
    metadata: dict[str, Any] | None = None,
    uploaded_lims_info: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a demux_sample_info document holding samplesheets."""
    return {
        "_id": "uuid-dsi-1",
        "_rev": rev,
        "flowcell_id": flowcell_id,
        "samplesheets": samplesheets,
        "metadata": metadata or {"run_mode": "standard"},
        "uploaded_lims_info": uploaded_lims_info or [],
    }


def planning_ctx(
    scope_id: str,
    *,
    demux_doc: dict[str, Any] | None,
    fc_doc: dict[str, Any] | None,
) -> MagicMock:
    """Return a PlanningContext stand-in whose planning reads return the documents."""
    ctx = MagicMock()
    ctx.scope = {"kind": "flowcell", "id": scope_id}
    demux_db = AsyncMock()
    demux_db.get.return_value = demux_doc
    demux_db.find_one.return_value = demux_doc
    fc_db = AsyncMock()
    fc_db.find_one.return_value = fc_doc
    ctx.db_mocks = {"demux_sample_info_db": demux_db, "flowcell_status_db": fc_db}
    ctx.data.couchdb.side_effect = ctx.db_mocks.__getitem__
    return ctx
