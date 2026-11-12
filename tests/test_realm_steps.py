import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from yggdrasil.flow.errors import PermanentStepError

from demux_realm.config import RealmConfig
from demux_realm.nextflow import settings_from_mapping
from demux_realm.steps import (
    aggregate_illumina_stats,
    clean_lane,
    collect_results,
    prune_stale_lanes,
    symlink_fastqs,
    upload_results,
    upsert_x_flowcell_pre_demux,
    validate_runfolder,
)

# The @step decorator changes the calling convention; the original function is
# accessible via __wrapped__ for direct unit testing.
_step = upsert_x_flowcell_pre_demux.__wrapped__  # type: ignore[attr-defined]
_collect_results = collect_results.__wrapped__  # type: ignore[attr-defined]
_upload_results = upload_results.__wrapped__  # type: ignore[attr-defined]
_validate_runfolder = validate_runfolder.__wrapped__  # type: ignore[attr-defined]
_aggregate_illumina_stats = aggregate_illumina_stats.__wrapped__  # type: ignore[attr-defined]
_clean_lane = clean_lane.__wrapped__  # type: ignore[attr-defined]
_prune_stale_lanes = prune_stale_lanes.__wrapped__  # type: ignore[attr-defined]
_symlink_fastqs = symlink_fastqs.__wrapped__  # type: ignore[attr-defined]

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
        "BCLConvert_Settings": {"CreateFastqForIndexReads": "1"},
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
    x_flowcells_client.find_one.return_value = None
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


def test_step_keeps_demux_results_of_the_existing_document(tmp_path, scenario):
    ctx, x_client = _make_ctx(tmp_path)
    x_client.find_one.return_value = {
        "_id": "doc-1",
        "_rev": "3-x",
        "name": "20260312_ASC2177698-SC3",
        "samplesheet_csv": ["stale"],
        "Json_Stats": {"x": 1},
        "Demultiplex_lane_1_settings_0": {"Demultiplex_Stats": [{"Lane": "1"}]},
        "illumina": {"Demultiplex_Stats": {}},
        "Undetermined": {"1": {}},
    }

    result = _step(ctx, scenario)

    payload = _saved_payload(x_client)
    assert payload["Demultiplex_lane_1_settings_0"] == {
        "Demultiplex_Stats": [{"Lane": "1"}]
    }
    assert payload["illumina"] == {"Demultiplex_Stats": {}}
    assert payload["Undetermined"] == {"1": {}}
    assert "Json_Stats" not in payload
    assert payload["samplesheet_csv"] != ["stale"]
    assert "_id" not in payload and "_rev" not in payload
    assert result.metrics["kept_fields"] == [
        "Demultiplex_lane_1_settings_0",
        "Undetermined",
        "illumina",
    ]


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


def test_validate_runfolder_exists(runfolder, scenario):
    ctx = MagicMock()
    result = _validate_runfolder(ctx, scenario)
    assert result.metrics["path_validated"] == str(runfolder)


def test_validate_runfolder_raises_on_missing_path(runfolder, scenario):
    ctx = MagicMock()
    scenario["hpc_runfolder_path"] = str(runfolder / "nonexistent")
    with pytest.raises(FileNotFoundError, match="HPC runfolder not found"):
        _validate_runfolder(ctx, scenario)


def test_validate_runfolder_raises_on_not_a_directory(runfolder, scenario):
    ctx = MagicMock()
    scenario["hpc_runfolder_path"] = str(runfolder / "not_a_directory")
    (runfolder / "not_a_directory").touch()
    with pytest.raises(
        NotADirectoryError, match="HPC runfolder path is not a directory"
    ):
        _validate_runfolder(ctx, scenario)


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


def test_collect_results_and_upload_results_update_x_flowcells(tmp_path, monkeypatch):
    realm_config = RealmConfig(
        hpc_base_path="",
        nextflow=settings_from_mapping({"outdir_base": str(tmp_path / "results")}),
    )
    monkeypatch.setattr("demux_realm.steps.load_realm_config", lambda: realm_config)
    scenario: dict[str, Any] = {
        "canonical_flowcell_id": "SC2177698-SC3",
        "lane_id": "1",
        "settings_index": "0",
        "runfolder_id": "20260312_SH01140_0005_ASC2177698-SC3",
    }
    outdir = tmp_path / "results" / "SC2177698-SC3" / "lane_1_settings_0"
    stats_dir = outdir / "SC2177698-SC3" / "Reports"
    stats_dir.mkdir(parents=True)

    (stats_dir / "Demultiplex_Stats.csv").write_text(
        "Sample,Reads,Percent_Aligned\nP12345_1001,12345,98.2\n"
    )
    (stats_dir / "Quality_Metrics.csv").write_text(
        "Sample,Mean_Q30\nP12345_1001,92.1\n"
    )
    (stats_dir / "Top_Unknown_Barcodes.csv").write_text("Barcode,Count\nACGT,14\n")

    ctx = MagicMock()
    result = _collect_results(ctx, scenario)
    metadata = cast(
        dict[str, list[dict[str, str]]],
        scenario["demultiplexing_metadata"],
    )
    assert metadata["Demultiplex_Stats"][0]["Sample"] == "P12345_1001"
    assert set(result.metrics["demultiplexing_metadata_files"]) == {
        "Demultiplex_Stats",
        "Quality_Metrics",
        "Top_Unknown_Barcodes",
    }

    client = MagicMock()
    client.find_one.return_value = {
        "_id": "doc-123",
        "_rev": "2-abc",
        "name": "20260312_ASC2177698-SC3",
        "RunInfo": {"Id": "20260312_SH01140_0005_ASC2177698-SC3"},
    }
    client.clean_doc.side_effect = lambda doc: {
        k: v for k, v in doc.items() if k not in ("_id", "_rev")
    }
    client.save.return_value = MagicMock(status="updated", doc_id="doc-123")
    ctx.data.connection.return_value = client

    upload_result = _upload_results(ctx, scenario)
    saved_doc = client.save.call_args.args[0]
    assert "_id" not in saved_doc
    assert "_rev" not in saved_doc
    branch_field = "Demultiplex_lane_1_settings_0"
    assert saved_doc[branch_field]["Demultiplex_Stats"][0]["Sample"] == "P12345_1001"
    assert saved_doc[branch_field]["Quality_Metrics"][0]["Mean_Q30"] == "92.1"
    assert saved_doc[branch_field]["Top_Unknown_Barcodes"][0]["Barcode"] == "ACGT"
    assert upload_result.metrics["upload_status"] == "simulated_success"
    assert upload_result.metrics["demultiplex_field"] == branch_field


def _qm_row(lane, sample, read_number, yield_, yield_q30, quality_sum):
    return {
        "Lane": lane,
        "SampleID": sample,
        "ReadNumber": read_number,
        "Yield": str(yield_),
        "YieldQ30": str(yield_q30),
        "QualityScoreSum": str(quality_sum),
    }


def test_aggregate_illumina_stats_combines_all_lanes():
    existing = {
        "_id": "doc-123",
        "_rev": "2-abc",
        "name": "20260312_ASC2177698-SC3",
        "samplesheet_csv": [
            {"Sample_ID": "P1_101", "Sample_Project": "A__Proj"},
            {"Sample_ID": "P2_201", "Sample_Project": "B__Proj"},
        ],
        "Demultiplex_lane_1_settings_0": {
            "Demultiplex_Stats": [
                {
                    "Lane": "1",
                    "SampleID": "P1_101",
                    "Index": "AAAA+TTTT",
                    "# Reads": "1000",
                    "# Perfect Index Reads": "900",
                    "# One Mismatch Index Reads": "100",
                    "% Reads": "50.00",
                    "% Perfect Index Reads": "90.00",
                    "% One Mismatch Index Reads": "10.00",
                },
                {
                    "Lane": "1",
                    "SampleID": "Undetermined",
                    "Index": "",
                    "# Reads": "1000",
                    "# Perfect Index Reads": "1000",
                    "# One Mismatch Index Reads": "0",
                    "% Reads": "50.00",
                    "% Perfect Index Reads": "100.00",
                    "% One Mismatch Index Reads": "0.00",
                },
            ],
            "Quality_Metrics": [
                _qm_row("1", "P1_101", "1", 100_000_000, 90_000_000, 3_500_000_000),
                _qm_row("1", "P1_101", "2", 100_000_000, 85_000_000, 3_400_000_000),
                _qm_row("1", "P1_101", "I1", 999, 999, 999),
                _qm_row(
                    "1", "Undetermined", "1", 50_000_000, 20_000_000, 1_000_000_000
                ),
                _qm_row("1", "Undetermined", "2", 50_000_000, 19_000_000, 900_000_000),
            ],
        },
        "Demultiplex_lane_2_settings_0": {
            "Demultiplex_Stats": [
                {
                    "Lane": "2",
                    "SampleID": "P2_201",
                    "Index": "CCCC+GGGG",
                    "# Reads": "500",
                    "# Perfect Index Reads": "400",
                    "# One Mismatch Index Reads": "50",
                    "% Reads": "100.00",
                    "% Perfect Index Reads": "80.00",
                    "% One Mismatch Index Reads": "10.00",
                }
            ],
            "Quality_Metrics": [
                _qm_row("2", "P2_201", "1", 50_000_000, 40_000_000, 1_500_000_000),
                _qm_row("2", "P2_201", "2", 50_000_000, 42_000_000, 1_600_000_000),
            ],
        },
    }

    client = MagicMock()
    client.find_one.return_value = existing
    client.clean_doc.side_effect = lambda doc: {
        k: v for k, v in doc.items() if k not in ("_id", "_rev")
    }
    client.save.return_value = MagicMock(status="updated", doc_id="doc-123")
    ctx = MagicMock()
    ctx.data.connection.return_value = client

    scenario = {"runfolder_id": "20260312_SH01140_0005_ASC2177698-SC3"}
    result = _aggregate_illumina_stats(ctx, scenario)

    saved_doc = client.save.call_args.args[0]
    assert "_id" not in saved_doc
    assert "_rev" not in saved_doc
    stats = saved_doc["illumina"]["Demultiplex_Stats"]
    by_sample = {row["Sample"]: row for row in stats["Barcode_lane_statistics"]}

    p1 = by_sample["P1_101"]
    assert p1["Project"] == "A__Proj"
    assert p1["PF Clusters"] == "1,000"
    assert p1["% of thelane"] == "50.00"
    assert p1["% PFClusters"] == "50.00"
    assert p1["% Perfectbarcode"] == "90.00"
    assert p1["% One mismatchbarcode"] == "10.00"
    assert p1["Yield (Mbases)"] == "200"
    assert p1["Mean QualityScore"] == "34.50"
    assert p1["% >= Q30bases"] == "87.50"

    undetermined = by_sample["Undetermined"]
    assert undetermined["Project"] == "default"
    assert undetermined["Barcode sequence"] == "unknown"
    assert undetermined["Yield (Mbases)"] == "100"

    lanes_by_id = {row["Lane"]: row for row in stats["Lanes_stats"]}
    lane1 = lanes_by_id["1"]
    assert lane1["PF Clusters"] == "2,000"
    assert lane1["% Perfectbarcode"] == "95.00"
    assert lane1["% One mismatchbarcode"] == "5.00"
    assert lane1["Yield (Mbases)"] == "300"
    assert lane1["% >= Q30bases"] == "71.33"
    assert lane1["Mean QualityScore"] == "29.33"

    lane2 = lanes_by_id["2"]
    assert lane2["PF Clusters"] == "500"
    assert lane2["% Perfectbarcode"] == "80.00"
    assert lane2["Yield (Mbases)"] == "100"

    flowcell = stats["Flowcell_stats"]
    assert flowcell["Clusters (Raw)"] == "0"  # Placeholder
    assert flowcell["Clusters(PF)"] == "2,500"
    assert flowcell["Yield (MBases)"] == "400"

    assert result.metrics["lane_count"] == 2
    assert result.metrics["sample_row_count"] == 3


def test_step_raises_when_data_not_injected(tmp_path, scenario):
    ctx, _ = _make_ctx(tmp_path)
    ctx.data = None
    with pytest.raises(RuntimeError, match="DataAccess.*not injected"):
        _step(ctx, scenario)


# ---------------------------------------------------------------------------
# Lane cleanup
# ---------------------------------------------------------------------------

_FCID = "SC2177698-SC3"
_BRANCHES = ("lane_1_settings_0", "lane_1_settings_1", "lane_2_settings_0")


@pytest.fixture
def earlier_run(tmp_path, monkeypatch):
    """Lay out the results, work dirs, links and fields of an earlier run."""
    results, work = tmp_path / "results", tmp_path / "work"
    realm_config = RealmConfig(
        hpc_base_path="",
        nextflow=settings_from_mapping(
            {"outdir_base": str(results), "work_base": str(work)}
        ),
    )
    monkeypatch.setattr("demux_realm.steps.load_realm_config", lambda: realm_config)
    runfolder = tmp_path / "runfolder"
    demux_dir = runfolder / "Demultiplexing"
    demux_dir.mkdir(parents=True)
    for branch in _BRANCHES:
        fastq = results / _FCID / branch / _FCID / f"{branch}_R1.fastq.gz"
        fastq.parent.mkdir(parents=True)
        fastq.touch()
        (work / _FCID / branch / "ab").mkdir(parents=True)
        (demux_dir / fastq.name).symlink_to(fastq)
    # A dangling link left by a branch whose results are already gone.
    (demux_dir / "gone_R1.fastq.gz").symlink_to(
        results / _FCID / "lane_1_settings_7" / "gone_R1.fastq.gz"
    )
    (results / _FCID / "lane_1_settings_7").mkdir()
    (demux_dir / "unrelated.txt").touch()

    doc = {
        "_id": "doc-123",
        "_rev": "2-abc",
        "name": "20260312_ASC2177698-SC3",
        "illumina": {"x": 1},
        **{f"Demultiplex_{branch}": {"Demultiplex_Stats": []} for branch in _BRANCHES},
    }
    ctx = MagicMock()
    client = ctx.data.connection.return_value
    client.find_one.return_value = doc
    client.clean_doc.side_effect = lambda d: {
        k: v for k, v in d.items() if k not in ("_id", "_rev")
    }
    scenario = {
        "canonical_flowcell_id": _FCID,
        "runfolder_id": "20260312_SH01140_0005_ASC2177698-SC3",
        "hpc_runfolder_path": str(runfolder),
    }
    return SimpleNamespace(
        results=results,
        work=work,
        demux_dir=demux_dir,
        ctx=ctx,
        client=client,
        scenario=scenario,
    )


def test_clean_lane_removes_only_that_lanes_results(earlier_run):
    result = _clean_lane(earlier_run.ctx, {**earlier_run.scenario, "lane_id": "1"})

    flowcell_results = earlier_run.results / _FCID
    assert sorted(p.name for p in flowcell_results.iterdir()) == ["lane_2_settings_0"]
    assert sorted(p.name for p in (earlier_run.work / _FCID).iterdir()) == [
        "lane_2_settings_0"
    ]
    assert sorted(p.name for p in earlier_run.demux_dir.iterdir()) == [
        "lane_2_settings_0_R1.fastq.gz",
        "unrelated.txt",
    ]
    saved = earlier_run.client.save.call_args.args[0]
    assert sorted(saved) == ["Demultiplex_lane_2_settings_0", "illumina", "name"]
    assert result.metrics["removed_fields"] == [
        "Demultiplex_lane_1_settings_0",
        "Demultiplex_lane_1_settings_1",
    ]
    assert len(result.metrics["removed_symlinks"]) == 3
    assert len(result.metrics["removed_dirs"]) == 5


def test_clean_lane_without_earlier_results_does_nothing(earlier_run):
    earlier_run.client.find_one.return_value = None

    result = _clean_lane(earlier_run.ctx, {**earlier_run.scenario, "lane_id": "3"})

    assert result.metrics["removed_dirs"] == []
    assert result.metrics["removed_symlinks"] == []
    assert result.metrics["removed_fields"] == []
    earlier_run.client.save.assert_not_called()


@pytest.mark.parametrize("fcid", ["", "..", "a/b"])
def test_clean_lane_refuses_a_flowcell_id_that_is_not_one_component(
    earlier_run, fcid
):
    scenario = {**earlier_run.scenario, "lane_id": "1", "canonical_flowcell_id": fcid}

    with pytest.raises(PermanentStepError, match="Refusing"):
        _clean_lane(earlier_run.ctx, scenario)


def test_prune_stale_lanes_removes_lanes_outside_the_plan(earlier_run):
    result = _prune_stale_lanes(
        earlier_run.ctx, {**earlier_run.scenario, "lanes": ["1"]}
    )

    assert sorted(p.name for p in (earlier_run.results / _FCID).iterdir()) == [
        "lane_1_settings_0",
        "lane_1_settings_1",
        "lane_1_settings_7",
    ]
    assert not (earlier_run.work / _FCID / "lane_2_settings_0").exists()
    assert "lane_2_settings_0_R1.fastq.gz" not in {
        p.name for p in earlier_run.demux_dir.iterdir()
    }
    assert result.metrics["removed_fields"] == ["Demultiplex_lane_2_settings_0"]


def test_prune_stale_lanes_keeps_every_planned_lane(earlier_run):
    result = _prune_stale_lanes(
        earlier_run.ctx, {**earlier_run.scenario, "lanes": ["1", "2"]}
    )

    assert result.metrics == {
        "removed_dirs": [],
        "removed_symlinks": [],
        "removed_fields": [],
    }
    earlier_run.client.save.assert_not_called()


def test_symlink_fastqs_replaces_a_dangling_link(earlier_run):
    scenario = {**earlier_run.scenario, "lane_id": "1", "settings_index": "0"}
    fastq = (
        earlier_run.results / _FCID / "lane_1_settings_0" / _FCID
        / "lane_1_settings_0_R1.fastq.gz"
    )
    link = earlier_run.demux_dir / fastq.name
    link.unlink()
    link.symlink_to(earlier_run.results / "gone" / fastq.name)

    _symlink_fastqs(earlier_run.ctx, scenario)

    assert link.resolve() == fastq.resolve()


def test_symlink_fastqs_keeps_a_current_link(earlier_run):
    scenario = {**earlier_run.scenario, "lane_id": "1", "settings_index": "0"}
    link = earlier_run.demux_dir / "lane_1_settings_0_R1.fastq.gz"
    before = os.readlink(link)

    _symlink_fastqs(earlier_run.ctx, scenario)

    assert os.readlink(link) == before
