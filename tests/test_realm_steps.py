import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dataflow_demux.yggdrasil_realm.steps import upsert_x_flowcell_pre_demux

# The @step decorator changes the calling convention; the original function is
# accessible via __wrapped__ for direct unit testing.
_step = upsert_x_flowcell_pre_demux.__wrapped__  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Shared XML content and samplesheet fixture
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


def _make_ctx(tmp_path, existing_doc=None):
    ctx = MagicMock()
    ctx.workdir = tmp_path / "workdir"
    ctx.workdir.mkdir(exist_ok=True)

    x_flowcells_client = MagicMock()
    x_flowcells_client.find_one_blocking.return_value = existing_doc
    ctx.data.couchdb.return_value = x_flowcells_client
    return ctx, x_flowcells_client


# ---------------------------------------------------------------------------
# Create path (no existing doc)
# ---------------------------------------------------------------------------


def test_step_create_produces_artifact(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)

    result = _step(ctx, scenario)

    assert result.metrics["action"] == "create"
    assert result.metrics["x_flowcell_name"] == "20260312_ASC2177698-SC3"
    assert result.metrics["samplesheet_rows"] == 1

    out_file = ctx.workdir / "x_flowcell_pre_demux.json"
    assert out_file.exists()


def test_step_create_doc_has_no_id_or_rev(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert "_id" not in doc
    assert "_rev" not in doc


def test_step_create_doc_has_required_fields(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert doc["name"] == "20260312_ASC2177698-SC3"
    assert "RunInfo" in doc
    assert "RunParameters" in doc
    assert isinstance(doc["samplesheet_csv"], list)
    assert len(doc["samplesheet_csv"]) == 1


def test_step_create_samplesheet_csv_row_fields(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    row = doc["samplesheet_csv"][0]
    assert row["Sample_ID"] == "P12345_1001"
    assert row["index"] == "ACGTACGTAA"
    assert row["index2"] == "TGCATGCATT"
    assert row["Sample_Project"] == "G__Example_26_03"


def test_step_run_info_flowcell_is_canonical_fcid(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    # RunInfo.xml <Flowcell> is chip lot number (BXA66715-2010); step must override
    # it with canonical_flowcell_id so the document holds the NGI flowcell ID.
    assert doc["RunInfo"]["Flowcell"] == "SC2177698-SC3"


def test_step_samplesheet_csv_has_fcid(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert doc["samplesheet_csv"][0]["FCID"] == "SC2177698-SC3"


def test_step_samplesheet_csv_has_lims_fields(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    row = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())[
        "samplesheet_csv"
    ][0]
    assert row["Sample_Ref"] == "Human (GRCh38)"
    assert row["Description"] == "G__Example_26_03"
    assert row["Control"] == "N"
    assert row["Recipe"] == "151-151"
    assert row["Operator"] == "Test_Operator"


def test_step_samplesheet_csv_no_lims_when_not_provided(tmp_path, scenario):
    scenario["uploaded_lims_info"] = []
    ctx, _ = _make_ctx(tmp_path)
    _step(ctx, scenario)

    row = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())[
        "samplesheet_csv"
    ][0]
    assert "Description" not in row
    assert "Operator" not in row


# ---------------------------------------------------------------------------
# Update path (existing doc present)
# ---------------------------------------------------------------------------


def test_step_update_action_reported(tmp_path, scenario):
    existing = {
        "_id": "abc123",
        "_rev": "3-xyz",
        "name": "20260312_ASC2177698-SC3",
        "RunInfo": {"Id": "old"},
    }
    ctx, _ = _make_ctx(tmp_path, existing_doc=existing)

    result = _step(ctx, scenario)
    assert result.metrics["action"] == "update"


def test_step_update_preserves_id_and_rev(tmp_path, scenario):
    existing = {
        "_id": "abc123",
        "_rev": "3-xyz",
        "name": "20260312_ASC2177698-SC3",
        "RunInfo": {"Id": "old"},
    }
    ctx, _ = _make_ctx(tmp_path, existing_doc=existing)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert doc["_id"] == "abc123"
    assert doc["_rev"] == "3-xyz"


def test_step_update_overrides_owned_fields(tmp_path, scenario):
    existing = {
        "_id": "abc123",
        "_rev": "3-xyz",
        "name": "20260312_ASC2177698-SC3",
        "RunInfo": {"Id": "old-id", "stale": True},
    }
    ctx, _ = _make_ctx(tmp_path, existing_doc=existing)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert doc["RunInfo"]["Id"] == "20260312_SH01140_0005_ASC2177698-SC3"
    assert "stale" not in doc["RunInfo"]


def test_step_update_preserves_unrelated_fields(tmp_path, scenario):
    existing = {
        "_id": "abc123",
        "_rev": "3-xyz",
        "name": "20260312_ASC2177698-SC3",
        "RunInfo": {"Id": "old"},
        "Json_Stats": {"post": "demux data"},
        "Undetermined": 1234,
    }
    ctx, _ = _make_ctx(tmp_path, existing_doc=existing)
    _step(ctx, scenario)

    doc = json.loads((ctx.workdir / "x_flowcell_pre_demux.json").read_text())
    assert doc["Json_Stats"] == {"post": "demux data"}
    assert doc["Undetermined"] == 1234


# ---------------------------------------------------------------------------
# CouchDB lookup key
# ---------------------------------------------------------------------------


def test_step_queries_by_name(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    _step(ctx, scenario)
    x_client.find_one_blocking.assert_called_once_with(
        {"name": "20260312_ASC2177698-SC3"}
    )


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
