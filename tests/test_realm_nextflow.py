"""Tests for the per-branch Nextflow launch module and the real execute_demux step."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from demux_realm.config import RealmConfig, realm_config_from_mapping
from demux_realm.nextflow import (
    NextflowRunOptions,
    NextflowSettings,
    branch_outdir,
    branch_run_tag,
    branch_work_dir,
    build_nextflow_command,
    parse_run_options,
    resolve_run_options,
    run_nextflow,
    settings_from_mapping,
)
from demux_realm.utils import SAMPLESHEET_FILENAME

from demux_realm import steps
from yggdrasil.flow.errors import PermanentStepError

SAMSHEE = NextflowRunOptions(skip_tools=("samshee",))
NO_OPTIONS = NextflowRunOptions()


def make_settings(**overrides) -> NextflowSettings:
    base = dict(
        pipeline="nf-core/demultiplex",
        profile="docker",
        common_config=None,
        work_base=Path("/work/nf-realm-work"),
        outdir_base=Path("/work/nf-realm-results"),
        extra_args=(),
    )
    base.update(overrides)
    return NextflowSettings(**base)


# ---------------------------------------------------------------------------
# settings_from_mapping / realm config
# ---------------------------------------------------------------------------


def test_settings_defaults_target_local_dind_stack():
    s = settings_from_mapping(None)
    assert s == make_settings()
    assert s == settings_from_mapping({})
    assert s.skip_tools == ()


def test_settings_overrides_for_hpc():
    s = settings_from_mapping(
        {
            "pipeline": "/opt/pipelines/nf-core/demultiplex",
            "profile": "slurm,singularity",
            "common_config": "/opt/demux/resources.config",
            "work_base": "/scratch/dmx",
            "outdir_base": "/results/dmx",
            "extra_args": ["-c", "/opt/extra.config", "--publish_dir_mode", "copy"],
            "skip_tools": ["samshee"],
        }
    )
    assert s.pipeline == "/opt/pipelines/nf-core/demultiplex"
    assert s.profile == "slurm,singularity"
    assert s.common_config == Path("/opt/demux/resources.config")
    assert s.work_base == Path("/scratch/dmx")
    assert s.outdir_base == Path("/results/dmx")
    assert s.extra_args == ("-c", "/opt/extra.config", "--publish_dir_mode", "copy")
    assert s.skip_tools == ("samshee",)


@pytest.mark.parametrize(
    "raw",
    [
        {"skip_tools": "samshee"},  # must be a list of strings
        {"profile": ""},
        {"work_base": 3},
        {"extra_args": "-resume"},
        {"common_config": ""},
        ["not", "an", "object"],
    ],
)
def test_settings_reject_invalid_config(raw):
    with pytest.raises(ValueError):
        settings_from_mapping(raw)


def test_realm_config_defaults_and_hpc_base_path():
    assert realm_config_from_mapping(None) == RealmConfig(
        hpc_base_path="", nextflow=make_settings()
    )
    cfg = realm_config_from_mapping(
        {"hpc_base_path": "/work", "nextflow": {"profile": "slurm"}}
    )
    assert cfg.hpc_base_path == "/work"
    assert cfg.nextflow.profile == "slurm"


@pytest.mark.parametrize(
    "raw", [{"hpc_base": "/work"}, {"hpc_base_path": 5}, {"nextflow": "docker"}]
)
def test_realm_config_rejects_invalid_config(raw):
    with pytest.raises(ValueError):
        realm_config_from_mapping(raw)


# ---------------------------------------------------------------------------
# Per-run options (demux_sample_info)
# ---------------------------------------------------------------------------


def test_run_options_default_to_nothing_skipped():
    assert parse_run_options(None) == NO_OPTIONS
    assert parse_run_options({}) == NO_OPTIONS


def test_run_options_skip_tools_round_trip():
    options = parse_run_options({"skip_tools": ["samshee", "fastp"]})
    assert options.skip_tools == ("samshee", "fastp")
    assert parse_run_options(options.to_dict()) == options


@pytest.mark.parametrize(
    "raw",
    [
        {"skip_tools": "samshee,fastp"},
        {"skip_tools": ["sam shee"]},
        {"skip_tools": ["a,b"]},
        {"skip_tools": [3]},
        {"skip_toools": ["samshee"]},
        "samshee",
    ],
)
def test_run_options_reject_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_run_options(raw)


# ---------------------------------------------------------------------------
# Resolving per-branch options (commandline_options > config default)
# ---------------------------------------------------------------------------


def _entry_with(cmdline: dict) -> dict:
    return {"commandline_options": cmdline}


def test_resolve_run_options_prefers_the_samplesheet_entry():
    settings = make_settings(skip_tools=("falco",))
    entry = _entry_with({"nextflow": {"skip_tools": ["samshee"]}})
    options = resolve_run_options(entry, settings)
    assert options.skip_tools == ("samshee",)


def test_resolve_run_options_falls_back_to_the_config_default():
    settings = make_settings(skip_tools=("falco", "multiqc"))
    default = NextflowRunOptions(skip_tools=("falco", "multiqc"))
    assert resolve_run_options({}, settings) == default
    assert resolve_run_options(None, settings) == default
    # A present-but-empty section still falls back.
    assert resolve_run_options(_entry_with({"nextflow": {}}), settings) == default


def test_resolve_run_options_explicit_empty_list_skips_nothing():
    settings = make_settings(skip_tools=("samshee",))
    entry = _entry_with({"nextflow": {"skip_tools": []}})
    assert resolve_run_options(entry, settings) == NO_OPTIONS


@pytest.mark.parametrize(
    "cmdline",
    [
        "nextflow",  # not a mapping
        {"fastp": {}},  # unknown tool section
        {"nextflow": "skip"},  # section not a mapping
        {"nextflow": {"skip_toools": []}},  # unknown key
        {"nextflow": {"skip_tools": "samshee"}},  # not a list
        {"nextflow": {"skip_tools": 3}},
        {"nextflow": {"skip_tools": ["bad name"]}},
    ],
)
def test_resolve_run_options_rejects_invalid_entries(cmdline):
    with pytest.raises(ValueError):
        resolve_run_options(_entry_with(cmdline), make_settings())


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def test_command_argv_and_branch_layout():
    s = make_settings(
        common_config=Path("/opt/demux/resources.config"),
        extra_args=("-c", "/opt/extra.config"),
    )
    cmd = build_nextflow_command(
        s,
        canonical_flowcell_id="SC2177698-SC3",
        lane_id="2",
        settings_index="0",
        samplesheet=Path("/wd/branch/SampleSheet.csv"),
        flowcell_path=Path("/incoming/SC2177698-SC3"),
        run_options=SAMSHEE,
    )
    # Pipeline parameters use the double dash; nextflow CLI options the single.
    assert cmd == [
        "nextflow",
        "run",
        "nf-core/demultiplex",
        "--flowcell_id",
        "SC2177698-SC3",
        "--flowcell_samplesheet",
        "/wd/branch/SampleSheet.csv",
        "--flowcell_path",
        "/incoming/SC2177698-SC3",
        "--skip_tools",
        "samshee",
        "-c",
        "/opt/demux/resources.config",
        "-profile",
        "docker",
        "-work-dir",
        "/work/nf-realm-work/SC2177698-SC3/lane_2_settings_0",
        "--outdir",
        "/work/nf-realm-results/SC2177698-SC3/lane_2_settings_0",
        "-resume",
        "-c",
        "/opt/extra.config",
    ]


def test_command_omits_absent_optionals():
    s = make_settings()
    cmd = build_nextflow_command(
        s,
        canonical_flowcell_id="FC",
        lane_id="1",
        settings_index="0",
        samplesheet=Path("/ss.csv"),
        flowcell_path=Path("/fc"),
        run_options=NO_OPTIONS,
    )
    assert "--skip_tools" not in cmd
    assert "-c" not in cmd
    assert cmd[-1] == "-resume"


def test_branch_directories_never_shared():
    s = make_settings()
    tags = (
        branch_run_tag("FC", "1", "0"),
        branch_run_tag("FC", "2", "0"),
        branch_run_tag("FC", "1", "1"),
    )
    dirs = {branch_work_dir(s, t) for t in tags} | {branch_outdir(s, t) for t in tags}
    assert len(dirs) == 6


def test_run_nextflow_streams_output_to_log():
    # The log is written while the process runs (live-tailable), and the
    # combined output is returned for error reporting.
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "nextflow_run.log"
        proc = run_nextflow(
            [sys.executable, "-c", "print('line one'); print('line two', file=__import__('sys').stderr)"],
            log_path,
        )
        log_text = log_path.read_text()
    assert proc.returncode == 0
    assert proc.stdout == "line one\nline two\n"
    assert log_text == "line one\nline two\n"


def test_command_has_no_stable_run_name():
    # A stable -name would make every re-run of a branch fail with
    # "Run name has been already used" (Nextflow indexes run names per
    # user home, not per work dir). Resume semantics come from -resume +
    # the unique per-branch work dir instead.
    s = make_settings()
    cmd = build_nextflow_command(
        s,
        canonical_flowcell_id="FC",
        lane_id="1",
        settings_index="0",
        samplesheet=Path("/step/SampleSheet.csv"),
        flowcell_path=Path("/incoming/FC"),
        run_options=NO_OPTIONS,
    )
    assert "-name" not in cmd
    assert "-resume" in cmd


# ---------------------------------------------------------------------------
# execute_demux / collect_results step bodies
# ---------------------------------------------------------------------------

SCENARIO = {
    "canonical_flowcell_id": "FC",
    "runfolder_id": "20260312_SH01140_0005_ASC2177698-SC3",
    "hpc_runfolder_path": "/incoming/FC",
    "lane_id": "1",
    "settings_index": "0",
    "samplesheet_payload": {
        "lane": 1,
        "Header": {"FileFormatVersion": "2"},
        "raw_samplesheet_settings": {"CreateFastqForIndexReads": "1"},
        "BCLConvert_Data": [
            {
                "Lane": "1",
                "Sample_ID": "P001_1",
                "Sample_Name": "P001_1",
                "index": "ACGTACGT",
                "Sample_Project": "P001",
            }
        ],
    },
}


PLAN_ID = "dmx_realm:FC:demux"
SHEET_REL = f"lane_1_settings_0__generate_samplesheet/{SAMPLESHEET_FILENAME}"


def make_ctx(root: Path):
    """A step context whose plan directory is root/PLAN_ID."""
    recorded: list[tuple] = []
    ctx = SimpleNamespace(
        workdir=root / PLAN_ID / "lane_1_settings_0__execute_demux",
        scope_dir=root,
        plan_id=PLAN_ID,
    )
    ctx.workdir.mkdir(parents=True)
    ctx.record_artifact = lambda ref, path=None: recorded.append((ref, path))
    return ctx, recorded


def write_sheet(root: Path) -> Path:
    """Write the sheet where the branch's generate_samplesheet step would."""
    sheet = root / PLAN_ID / SHEET_REL
    sheet.parent.mkdir(parents=True)
    sheet.write_text("[Header]\n")
    return sheet


def use_config(monkeypatch, tmp_path: Path, **nextflow) -> NextflowSettings:
    """Point the steps at a config with work/results under tmp_path."""
    config = realm_config_from_mapping(
        {
            "nextflow": {
                "work_base": str(tmp_path / "nf-work"),
                "outdir_base": str(tmp_path / "nf-results"),
                **nextflow,
            }
        }
    )
    monkeypatch.setattr(steps, "load_realm_config", lambda: config)
    return config.nextflow


def step_scenario(runfolder: Path, **overrides) -> dict:
    return {
        **SCENARIO,
        "hpc_runfolder_path": str(runfolder),
        "samplesheet_path": SHEET_REL,
        "nextflow": {"skip_tools": ["samshee"]},
        **overrides,
    }


def test_execute_demux_launches_nextflow_and_records(tmp_path, monkeypatch):
    settings = use_config(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, recorded = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)

    captured: dict = {}

    def fake_run(cmd, log_path):
        captured["cmd"] = cmd
        log_path.write_text("ok\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)

    result = steps.execute_demux.__wrapped__(ctx, scenario=step_scenario(runfolder))

    assert result.metrics["exit_status"] == 0
    # The step must delegate to the command builder with the branch values:
    # the producer step's sheet and the per-run options from the scenario.
    assert captured["cmd"] == build_nextflow_command(
        settings,
        canonical_flowcell_id="FC",
        lane_id="1",
        settings_index="0",
        samplesheet=sheet.resolve(),
        flowcell_path=runfolder,
        run_options=SAMSHEE,
    )
    assert "--skip_tools" in captured["cmd"]
    assert ("samplesheets", sheet.resolve()) in [
        (ref.folder, path) for ref, path in recorded
    ]
    assert "logs" in [ref.folder for ref, _ in recorded]


def test_execute_demux_without_options_skips_nothing(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    write_sheet(tmp_path)
    captured: dict = {}

    def fake_run(cmd, log_path):
        captured["cmd"] = cmd
        log_path.write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)
    scenario = step_scenario(runfolder)
    del scenario["nextflow"]

    steps.execute_demux.__wrapped__(ctx, scenario=scenario)

    assert "--skip_tools" not in captured["cmd"]


def test_execute_demux_nonzero_exit_fails_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    write_sheet(tmp_path)

    def fake_run(cmd, log_path):
        log_path.write_text("ERROR: something\n")
        return subprocess.CompletedProcess(cmd, 1, stdout="ERROR: something\n")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)

    with pytest.raises(PermanentStepError):
        steps.execute_demux.__wrapped__(ctx, scenario=step_scenario(runfolder))


def test_execute_demux_missing_samplesheet_fails_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    with pytest.raises(PermanentStepError, match="samplesheet missing"):
        steps.execute_demux.__wrapped__(
            ctx, scenario=step_scenario(tmp_path / "fc")
        )


def test_execute_demux_samplesheet_cannot_escape_plan_dir(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    (tmp_path / "outside.csv").write_text("x")
    with pytest.raises(PermanentStepError, match="escapes"):
        steps.execute_demux.__wrapped__(
            ctx,
            scenario=step_scenario(
                tmp_path / "fc", samplesheet_path="../outside.csv"
            ),
        )


def test_execute_demux_invalid_options_fail_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    write_sheet(tmp_path)
    with pytest.raises(PermanentStepError, match="nextflow options"):
        steps.execute_demux.__wrapped__(
            ctx,
            scenario=step_scenario(
                tmp_path / "fc", nextflow={"skip_tools": "samshee"}
            ),
        )


def test_collect_results_counts_branch_outputs(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    outdir = tmp_path / "nf-results" / "FC" / "lane_1_settings_0"
    (outdir / "FC").mkdir(parents=True)
    (outdir / "FC" / "sample1_L001_R1_001.fastq.gz").write_bytes(b"z")
    (outdir / "multiqc").mkdir()
    (outdir / "multiqc" / "multiqc_report.html").write_text("<html></html>")

    ctx, recorded = make_ctx(tmp_path)
    result = steps.collect_results.__wrapped__(ctx, scenario=SCENARIO)

    assert result.metrics["fastq_count"] == 1
    assert result.metrics["multiqc_report_count"] == 1
    assert any(ref.folder == "reports" for ref, _ in recorded)


def test_collect_results_missing_outdir_fails_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    with pytest.raises(PermanentStepError):
        steps.collect_results.__wrapped__(ctx, scenario=SCENARIO)
