"""Tests for the per-branch Nextflow launch module and the real execute_demux step."""

import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from yggdrasil.flow.errors import PermanentStepError
from yggdrasil.toolkit.sjob_manager import SlurmManagerFactory

from demux_realm import steps
from demux_realm.config import RealmConfig, realm_config_from_mapping
from demux_realm.nextflow import (
    NextflowRunOptions,
    NextflowSettings,
    branch_outdir,
    branch_run_tag,
    branch_work_dir,
    build_nextflow_command,
    build_slurm_batch_script,
    parse_run_options,
    resolve_run_options,
    run_nextflow,
    settings_from_mapping,
    slurm_job_name,
)
from demux_realm.steps import SAMPLESHEET

from realm_support import REQUIRED_NEXTFLOW

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


def test_settings_defaults_for_optional_keys():
    s = settings_from_mapping(REQUIRED_NEXTFLOW)
    assert s == make_settings()
    assert s.skip_tools == ()
    assert s.executor == "local"


@pytest.mark.parametrize("missing", ["profile", "work_base", "outdir_base"])
def test_settings_require_deployment_keys(missing):
    raw = {k: v for k, v in REQUIRED_NEXTFLOW.items() if k != missing}
    with pytest.raises(ValueError, match=f"nextflow.{missing} is required"):
        settings_from_mapping(raw)


@pytest.mark.parametrize("raw", [None, {}])
def test_settings_reject_empty_config(raw):
    with pytest.raises(ValueError):
        settings_from_mapping(raw)


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
        {"executor": "gridengine"},  # unknown executor
        {"sbatch_cpus": 0},
        {"sbatch_cpus": "4"},
        {"sbatch_cpus": True},
    ],
)
def test_settings_reject_invalid_config(raw):
    with pytest.raises(ValueError):
        settings_from_mapping({**REQUIRED_NEXTFLOW, **raw})


def test_settings_reject_sbatch_mem():
    # Not configurable: SLURM jobs always get the 4G default.
    with pytest.raises(ValueError, match="unknown key"):
        settings_from_mapping({**REQUIRED_NEXTFLOW, "sbatch_mem": "16G"})


def test_settings_reject_non_object():
    with pytest.raises(ValueError):
        settings_from_mapping(["not", "an", "object"])


def test_settings_slurm_executor_defaults():
    s = settings_from_mapping({**REQUIRED_NEXTFLOW, "executor": "slurm"})
    assert s.executor == "slurm"
    assert s.sbatch_cpus == 4
    assert s.sbatch_mem == "4G"
    assert s.sbatch_time == "240:00:00"
    assert s.sbatch_partition == "local"


def test_settings_slurm_executor_overrides():
    s = settings_from_mapping(
        {
            **REQUIRED_NEXTFLOW,
            "executor": "slurm",
            "sbatch_cpus": 8,
            "sbatch_time": "8:00:00",
            "sbatch_partition": "comp",
        }
    )
    assert (
        s.sbatch_cpus,
        s.sbatch_mem,
        s.sbatch_time,
        s.sbatch_partition,
    ) == (8, "4G", "8:00:00", "comp")


def test_realm_config_defaults_and_hpc_base_path():
    assert realm_config_from_mapping({"nextflow": REQUIRED_NEXTFLOW}) == RealmConfig(
        hpc_base_path="", nextflow=make_settings()
    )
    with pytest.raises(ValueError):
        realm_config_from_mapping(None)
    cfg = realm_config_from_mapping(
        {"hpc_base_path": "/work", "nextflow": {**REQUIRED_NEXTFLOW, "profile": "slurm"}}
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
    # combined output is returned for error reporting. (stdout and stderr
    # share one pipe, so the two lines may arrive in either order.)
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
    assert sorted(proc.stdout.splitlines()) == ["line one", "line two"]
    assert sorted(log_text.splitlines()) == ["line one", "line two"]


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


def test_batch_script_wraps_command_with_directives():
    script = build_slurm_batch_script(
        ["nextflow", "run", "nf-core/demultiplex", "--flowcell_path", "/a b/c"],
        job_name="dmx-test",
        log_path=Path("/work/plan/logs/run.log"),
        cpus=4,
        mem="4G",
        time="240:00:00",
        partition="local",
    )
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=dmx-test" in lines
    assert "#SBATCH --cpus-per-task=4" in lines
    assert "#SBATCH --mem=4G" in lines
    assert "#SBATCH --time=240:00:00" in lines
    assert "#SBATCH --partition=local" in lines
    # All job output goes to the shared-volume log the step tails.
    assert "#SBATCH --output=/work/plan/logs/run.log" in lines
    assert "#SBATCH --error=/work/plan/logs/run.log" in lines
    # The node environment is the deployment's business: no setup here.
    assert "docker" not in script.lower()
    assert not any(line.startswith("export ") for line in lines)
    # The argv is shlex-quoted, so a path with a space survives the node shell.
    assert lines[-1] == "nextflow run nf-core/demultiplex --flowcell_path '/a b/c'"


def test_slurm_job_name_sanitizes_and_bounds():
    assert slurm_job_name("FC/lane_1_settings_0") == "dmx-FC-lane_1_settings_0"
    assert slurm_job_name("A B/lane_1_settings_0") == "dmx-A-B-lane_1_settings_0"
    long = slurm_job_name("F" * 200 + "/lane_1_settings_0")
    assert len(long) == 80
    assert long.startswith("dmx-")


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
SHEET_REL = f"lane_1_settings_0__generate_samplesheet/{SAMPLESHEET.filename}"


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
                "profile": "docker",
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

    result = steps.execute_demux.__wrapped__(
        ctx, scenario=step_scenario(runfolder), samplesheet=sheet
    )

    assert result.metrics["exit_status"] == 0
    assert result.metrics["executor"] == "local"
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
    assert "logs" in [ref.folder for ref, _ in recorded]


def test_execute_demux_without_options_skips_nothing(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)
    captured: dict = {}

    def fake_run(cmd, log_path):
        captured["cmd"] = cmd
        log_path.write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)
    scenario = step_scenario(runfolder)
    del scenario["nextflow"]

    steps.execute_demux.__wrapped__(
        ctx, scenario=scenario, samplesheet=sheet
    )

    assert "--skip_tools" not in captured["cmd"]


def test_execute_demux_nonzero_exit_fails_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)

    def fake_run(cmd, log_path):
        log_path.write_text("ERROR: something\n")
        return subprocess.CompletedProcess(cmd, 1, stdout="ERROR: something\n")

    monkeypatch.setattr(steps, "run_nextflow", fake_run)

    with pytest.raises(PermanentStepError):
        steps.execute_demux.__wrapped__(
            ctx, scenario=step_scenario(runfolder), samplesheet=sheet
        )


def test_execute_demux_invalid_options_fail_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    ctx, _ = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)
    with pytest.raises(PermanentStepError, match="nextflow options"):
        steps.execute_demux.__wrapped__(
            ctx,
            scenario=step_scenario(
                tmp_path / "fc", nextflow={"skip_tools": "samshee"}
            ),
            samplesheet=sheet,
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


# ---------------------------------------------------------------------------
# execute_demux with the SLURM executor
# ---------------------------------------------------------------------------


class StubSlurmManager:
    """A SlurmJobManager stand-in: no sbatch/sacct, script path captured."""

    def __init__(self, job_id, final_state):
        self._job_id = job_id
        self._final_state = final_state
        self.submitted: list[Path] = []
        self.wait_args = None

    async def submit_job(self, script_path):
        self.submitted.append(Path(script_path))
        return self._job_id

    async def wait_for_completion(self, job_id, poll_interval=None):
        self.wait_args = (job_id, poll_interval)
        return self._final_state


def test_execute_demux_slurm_submits_and_waits(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, executor="slurm")
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, recorded = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)

    stub = StubSlurmManager("42", "COMPLETED")
    monkeypatch.setattr(
        SlurmManagerFactory, "get_manager", lambda is_dev: stub
    )

    result = steps.execute_demux.__wrapped__(
        ctx, scenario=step_scenario(runfolder), samplesheet=sheet
    )

    tag = "FC/lane_1_settings_0"
    assert result.metrics == {
        "branch": tag,
        "executor": "slurm",
        "job_id": "42",
        "final_state": "COMPLETED",
        "script": str(ctx.workdir / "nextflow_slurm.sh"),
        "work_dir": str(tmp_path / "nf-work" / tag),
        "outdir": str(tmp_path / "nf-results" / tag),
    }
    # The batch script carries the resource directives and the real argv.
    script = ctx.workdir / "nextflow_slurm.sh"
    assert script.exists()
    text = script.read_text()
    assert f"#SBATCH --job-name={slurm_job_name(tag)}" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert (
        f"--flowcell_samplesheet {shlex.quote(str(sheet.resolve()))}" in text
    )
    assert stub.submitted == [script]
    # One loop for submit + wait, with the explicit poll interval.
    assert stub.wait_args == ("42", 5.0)
    # The (shared-volume) log is recorded like on the local path.
    assert "logs" in [ref.folder for ref, _ in recorded]


def test_execute_demux_slurm_failed_state_fails_permanently(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, executor="slurm")
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)
    # The node writes the shared-volume log; simulate a failed run.
    (ctx.workdir / "nextflow_run.log").write_text("ERROR: boom\n")

    monkeypatch.setattr(
        SlurmManagerFactory, "get_manager",
        lambda is_dev: StubSlurmManager("43", "FAILED"),
    )
    with pytest.raises(PermanentStepError, match="FAILED"):
        steps.execute_demux.__wrapped__(
            ctx, scenario=step_scenario(runfolder), samplesheet=sheet
        )


def test_execute_demux_slurm_submission_failure_fails_permanently(
    tmp_path, monkeypatch
):
    use_config(monkeypatch, tmp_path, executor="slurm")
    runfolder = tmp_path / "fc"
    runfolder.mkdir()
    ctx, _ = make_ctx(tmp_path)
    sheet = write_sheet(tmp_path)

    monkeypatch.setattr(
        SlurmManagerFactory, "get_manager",
        lambda is_dev: StubSlurmManager(None, None),
    )
    with pytest.raises(PermanentStepError, match="sbatch submission failed"):
        steps.execute_demux.__wrapped__(
            ctx, scenario=step_scenario(runfolder), samplesheet=sheet
        )
