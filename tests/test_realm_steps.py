import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from dataflow_demux.yggdrasil_realm.steps import upsert_x_flowcell_pre_demux

# The @step decorator changes the calling convention; the original function is
# accessible via __wrapped__ for direct unit testing.
_step = upsert_x_flowcell_pre_demux.__wrapped__  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Shared XML content and samplesheet fixtures
# ---------------------------------------------------------------------------

_RUN_INFO_XML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunInfo Version="7">
      <Run Id="20260312_SH01140_0005_ASC2177698-SC3" Number="5">
        <Flowcell>BXA66715-2010</Flowcell>
        <Instrument>SH01140</Instrument>
        <Date>2026-03-12T14:25:57Z</Date>
        <Reads>
          <Read Number="1" NumCycles="301" IsIndexedRead="N" IsReverseComplement="N"/>
        </Reads>
        <FlowcellLayout LaneCount="1" SurfaceCount="1" SwathCount="9" TileCount="2"/>
      </Run>
    </RunInfo>
""")

_RUN_PARAMS_XML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunParameters>
      <InstrumentType>MiSeqi100Plus</InstrumentType>
      <InstrumentSerialNumber>SH01140</InstrumentSerialNumber>
      <RunId>20260312_SH01140_0005_ASC2177698-SC3</RunId>
      <RunCounter>5</RunCounter>
    </RunParameters>
""")

_SAMPLESHEETS = [
    {
        "lane": 1,
        "Header": {"FileFormatVersion": "2"},
        "raw_samplesheet_settings": {"CreateFastqForIndexReads": "1"},
        "BCLConvert_Data": [
            {
                "Lane": "1",
                "Sample_ID": "P12345_1001",
                "Sample_Name": "P12345_1001",
                "index": "ACGTACGTAA",
                "index2": "TGCATGCATT",
                "Sample_Project": "G__Example_26_03",
            }
        ],
    }
]

_UPLOADED_LIMS_INFO = [
    {
        "lane": "1",
        "sample_name": "P12345_1001",
        "sample_ref": "Human (GRCh38)",
        "description": "G__Example_26_03",
        "control": "N",
        "recipe": "151-151",
        "operator": "Test_Operator",
    }
]


@pytest.fixture
def runfolder(tmp_path):
    rf = tmp_path / "20260312_SH01140_0005_ASC2177698-SC3"
    rf.mkdir()
    (rf / "RunInfo.xml").write_text(_RUN_INFO_XML)
    (rf / "RunParameters.xml").write_text(_RUN_PARAMS_XML)
    return rf


@pytest.fixture
def scenario(runfolder):
    return {
        "hpc_runfolder_path": str(runfolder),
        "runfolder_id": "20260312_SH01140_0005_ASC2177698-SC3",
        "canonical_flowcell_id": "SC2177698-SC3",
        "samplesheets": list(_SAMPLESHEETS),
        "uploaded_lims_info": list(_UPLOADED_LIMS_INFO),
    }


def _make_ctx(tmp_path, *, save_status="created", save_doc_id="couch-id-abc123"):
    """Return (ctx, x_flowcells_client) with a pre-configured save() mock."""
    ctx = MagicMock()
    ctx.workdir = tmp_path / "workdir"
    ctx.workdir.mkdir(exist_ok=True)

    write_result = MagicMock()
    write_result.status = save_status
    write_result.doc_id = save_doc_id

    x_flowcells_client = MagicMock()
    x_flowcells_client.save.return_value = write_result
    ctx.data.connection.return_value = x_flowcells_client
    return ctx, x_flowcells_client


def _saved_payload(x_client):
    """Return the doc dict that was passed as the first arg to client.save()."""
    return x_client.save.call_args.args[0]


# ---------------------------------------------------------------------------
# CouchDB write — call shape
# ---------------------------------------------------------------------------


def test_step_calls_connection_with_correct_db(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)
    ctx.data.connection.assert_called_once_with("x_flowcells_db")


def test_step_save_uses_selector_and_upsert_mode(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    x_client.save.assert_called_once()
    kwargs = x_client.save.call_args.kwargs
    assert kwargs["selector"] == {"name": {"$eq": "20260312_ASC2177698-SC3"}}
    assert kwargs["mode"] == "upsert"


def test_step_payload_has_no_id_or_rev(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    payload = _saved_payload(x_client)
    assert "_id" not in payload
    assert "_rev" not in payload


# ---------------------------------------------------------------------------
# Payload content
# ---------------------------------------------------------------------------


def test_step_payload_has_required_fields(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    payload = _saved_payload(x_client)
    assert payload["name"] == "20260312_ASC2177698-SC3"
    assert "RunInfo" in payload
    assert "RunParameters" in payload
    assert isinstance(payload["samplesheet_csv"], list)
    assert len(payload["samplesheet_csv"]) == 1


def test_step_run_info_flowcell_is_canonical_fcid(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    # RunInfo.xml <Flowcell> is chip lot number (BXA66715-2010); step must override
    # it with canonical_flowcell_id so the document holds the NGI flowcell ID.
    assert _saved_payload(x_client)["RunInfo"]["Flowcell"] == "SC2177698-SC3"


def test_step_samplesheet_csv_row_fields(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    row = _saved_payload(x_client)["samplesheet_csv"][0]
    assert row["Sample_ID"] == "P12345_1001"
    assert row["index"] == "ACGTACGTAA"
    assert row["index2"] == "TGCATGCATT"
    assert row["Sample_Project"] == "G__Example_26_03"


def test_step_samplesheet_csv_has_fcid(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    assert _saved_payload(x_client)["samplesheet_csv"][0]["FCID"] == "SC2177698-SC3"


def test_step_samplesheet_csv_has_lims_fields(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    row = _saved_payload(x_client)["samplesheet_csv"][0]
    assert row["Sample_Ref"] == "Human (GRCh38)"
    assert row["Description"] == "G__Example_26_03"
    assert row["Control"] == "N"
    assert row["Recipe"] == "151-151"
    assert row["Operator"] == "Test_Operator"


def test_step_samplesheet_csv_no_lims_when_not_provided(tmp_path, scenario):
    scenario["uploaded_lims_info"] = []
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    row = _saved_payload(x_client)["samplesheet_csv"][0]
    assert "Description" not in row
    assert "Operator" not in row


# ---------------------------------------------------------------------------
# StepResult metrics
# ---------------------------------------------------------------------------


def test_step_metrics_on_create(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path, save_status="created", save_doc_id="new-id-001")
    result = _step(ctx, scenario)
    assert result.metrics["x_flowcell_name"] == "20260312_ASC2177698-SC3"
    assert result.metrics["samplesheet_row_count"] == 1
    assert result.metrics["write_status"] == "created"
    assert result.metrics["doc_id"] == "new-id-001"


def test_step_metrics_on_update(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path, save_status="updated", save_doc_id="existing-id-999")
    result = _step(ctx, scenario)
    assert result.metrics["write_status"] == "updated"
    assert result.metrics["doc_id"] == "existing-id-999"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_step_missing_run_info_raises(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    (Path(scenario["hpc_runfolder_path"]) / "RunInfo.xml").unlink()
    with pytest.raises(FileNotFoundError):
        _step(ctx, scenario)


def test_step_missing_run_params_raises(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    (Path(scenario["hpc_runfolder_path"]) / "RunParameters.xml").unlink()
    with pytest.raises(FileNotFoundError):
        _step(ctx, scenario)


def test_step_empty_samplesheets_raises(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    scenario["samplesheets"] = []
    with pytest.raises(ValueError, match="non-empty"):
        _step(ctx, scenario)


def test_step_samplesheet_missing_bcl_data_raises(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    scenario["samplesheets"] = [{"lane": 1}]  # no BCLConvert_Data
    with pytest.raises(ValueError, match="BCLConvert_Data"):
        _step(ctx, scenario)


def test_step_raises_when_data_not_injected(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    ctx.data = None
    with pytest.raises(RuntimeError, match="DataAccess.*not injected"):
        _step(ctx, scenario)
