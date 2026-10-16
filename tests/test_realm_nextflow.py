"""Tests for the per-branch Nextflow launch module and the real execute_demux step."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from demux_realm.nextflow import (
    ENV_CONFIG,
    ENV_EXTRA_ARGS,
    ENV_OUTDIR_BASE,
    ENV_PIPELINE,
    ENV_PROFILE,
    ENV_SKIP_TOOLS,
    ENV_WORK_BASE,
    NextflowSettings,
    branch_outdir,
    branch_run_tag,
    branch_work_dir,
    build_nextflow_command,
    sanitize_run_name,
    settings_from_env,
)

from demux_realm import steps
from yggdrasil.flow.errors import PermanentStepError


def make_settings(**overrides) -> NextflowSettings:
    base = dict(
        pipeline="nf-core/demultiplex",
        profile="docker",
        common_config=None,
        work_base=Path("/work/nf-realm-work"),
        outdir_base=Path("/work/nf-realm-results"),
        skip_tools=("samshee",),
        extra_args=(),
    )
    base.update(overrides)
    return NextflowSettings(**base)


def clean_env(monkeypatch) -> None:
    for name in (
        ENV_PIPELINE,
        ENV_PROFILE,
        ENV_CONFIG,
        ENV_WORK_BASE,
        ENV_OUTDIR_BASE,
        ENV_SKIP_TOOLS,
        ENV_EXTRA_ARGS,
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# settings_from_env
# ---------------------------------------------------------------------------


def test_settings_defaults_target_local_dind_stack(monkeypatch):
    clean_env(monkeypatch)
    s = settings_from_env()
    assert s.pipeline == "nf-core/demultiplex"
    assert s.profile == "docker"
    assert s.common_config is None
    assert s.work_base == Path("/work/nf-realm-work")
    assert s.outdir_base == Path("/work/nf-realm-results")
    assert s.skip_tools == ("samshee",)
    assert s.extra_args == ()


def test_settings_env_overrides_for_hpc(monkeypatch):
    env = {
        ENV_PIPELINE: "/opt/pipelines/nf-core/demultiplex",
        ENV_PROFILE: "slurm,singularity",
        ENV_CONFIG: "/opt/demux/resources.config",
        ENV_WORK_BASE: "/scratch/dmx",
        ENV_OUTDIR_BASE: "/results/dmx",
        ENV_SKIP_TOOLS: "samshee, fastp",
        ENV_EXTRA_ARGS: "-c /opt/extra.config --publish_dir_mode copy",
    }
    s = settings_from_env(env)
    assert s.pipeline == "/opt/pipelines/nf-core/demultiplex"
    assert s.profile == "slurm,singularity"
    assert s.common_config == Path("/opt/demux/resources.config")
    assert s.work_base == Path("/scratch/dmx")
    assert s.outdir_base == Path("/results/dmx")
    assert s.skip_tools == ("samshee", "fastp")
    assert s.extra_args == ("-c", "/opt/extra.config", "--publish_dir_mode", "copy")


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
        "-name",
        "sc2177698_sc3_lane_2_settings_0",
        "-resume",
        "-c",
        "/opt/extra.config",
    ]


def test_command_omits_absent_optionals():
    s = make_settings(skip_tools=())
    cmd = build_nextflow_command(
        s,
        canonical_flowcell_id="FC",
        lane_id="1",
        settings_index="0",
        samplesheet=Path("/ss.csv"),
        flowcell_path=Path("/fc"),
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


def test_sanitize_run_name_meets_nextflow_pattern():
    import re

    pattern = re.compile(r"^[a-z](?:[a-z\d]|[-_](?=[a-z\d])){0,79}$")
    cases = [
        ("SC2177698-SC3_lane_2_settings_0", "sc2177698_sc3_lane_2_settings_0"),
        ("20260310_LM43896_0386_B98QWERTY6/lane_1_settings_0", None),
        ("A--B__C", "a_b_c"),
        ("X" * 200, None),
    ]
    for raw, expected in cases:
        name = sanitize_run_name(raw.replace("/", "_"))
        assert pattern.match(name), name
        assert len(name) <= 80
        if expected is not None:
            assert name == expected, (raw, name)
    # Digit-leading names get a prefix so they still start with a letter.
    assert sanitize_run_name("2026_fc").startswith("run_")


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


def make_ctx(workdir: Path):
    recorded: list[tuple] = []
    ctx = SimpleNamespace(workdir=workdir)
    ctx.record_artifact = lambda ref, path=None: recorded.append((ref, path))
    return ctx, recorded


def step_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_WORK_BASE, str(tmp_path / "nf-work"))
    monkeypatch.setenv(ENV_OUTDIR_BASE, str(tmp_path / "nf-results"))
    monkeypatch.setenv(ENV_SKIP_TOOLS, "")


def test_execute_demux_launches_nextflow_and_records(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    step_env(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    scenario = {**SCENARIO, "hpc_runfolder_path": str(runfolder)}

    captured: dict = {}

    def fake_run(cmd, log_path):
        captured["cmd"] = cmd
        log_path.write_text("ok\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)

    ctx, recorded = make_ctx(tmp_path)
    result = steps.execute_demux.__wrapped__(ctx, scenario=scenario)

    assert result.metrics["exit_status"] == 0
    # The step must delegate to the command builder with the branch values.
    assert captured["cmd"] == build_nextflow_command(
        settings_from_env(),
        canonical_flowcell_id="FC",
        lane_id="1",
        settings_index="0",
        samplesheet=tmp_path / "SampleSheet.csv",
        flowcell_path=runfolder,
    )
    # The branch samplesheet is re-rendered into the step's own workdir.
    sheet = tmp_path / "SampleSheet.csv"
    assert sheet.read_text().startswith("[Header]")
    assert "[BCLConvert_Data]" in sheet.read_text()
    kinds = [ref.folder for ref, _ in recorded]
    assert "samplesheets" in kinds
    assert "logs" in kinds


def test_execute_demux_nonzero_exit_fails_permanently(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    step_env(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    scenario = {**SCENARIO, "hpc_runfolder_path": str(runfolder)}

    def fake_run(cmd, log_path):
        log_path.write_text("ERROR: something\n")
        return subprocess.CompletedProcess(cmd, 1, stdout="ERROR: something\n")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)

    ctx, _ = make_ctx(tmp_path)
    with pytest.raises(PermanentStepError):
        steps.execute_demux.__wrapped__(ctx, scenario=scenario)


def test_execute_demux_missing_runfolder_fails_permanently(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    step_env(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    with pytest.raises(PermanentStepError):
        steps.execute_demux.__wrapped__(
            ctx, scenario={**SCENARIO, "hpc_runfolder_path": str(tmp_path / "nope")}
        )


def test_collect_results_counts_branch_outputs(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    step_env(monkeypatch, tmp_path)
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
    clean_env(monkeypatch)
    step_env(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    with pytest.raises(PermanentStepError):
        steps.collect_results.__wrapped__(ctx, scenario=SCENARIO)
