import logging
from pathlib import Path
from typing import Annotated

from yggdrasil.flow.artifacts import SimpleArtifactRef
from yggdrasil.flow.errors import PermanentStepError
from yggdrasil.flow.model import StepResult
from yggdrasil.flow.step import In, Out, StepContext, step

from .config import load_realm_config
from .nextflow import (
    NextflowSettings,
    branch_outdir,
    branch_run_tag,
    branch_work_dir,
    build_nextflow_command,
    build_slurm_batch_script,
    parse_run_options,
    run_nextflow,
    slurm_job_name,
)
from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    build_lims_lookup,
    build_x_flowcell_payload,
    derive_xflowcell_name,
    flatten_samplesheets,
    parse_run_info_xml,
    parse_run_parameters_xml,
    render_bcl_convert_samplesheet,
    validate_lane_payload,
)

logger = logging.getLogger(__name__)

SAMPLESHEET = SimpleArtifactRef(
    key_name="samplesheet",
    folder="samplesheets",
    filename="SampleSheet.csv",
)


@step
def validate_runfolder(ctx: StepContext, scenario: dict) -> StepResult:
    """Validate that the runfolder is present."""
    hpc_path = Path(scenario["hpc_runfolder_path"])
    if not hpc_path.exists():
        raise FileNotFoundError(f"HPC runfolder not found: {hpc_path}")
    if not hpc_path.is_dir():
        raise NotADirectoryError(f"HPC runfolder path is not a directory: {hpc_path}")

    logger.info(
        "Validated HPC runfolder %s.",
        hpc_path,
    )
    return StepResult(
        metrics={
            "path_validated": str(hpc_path),
        }
    )


@step
def upsert_x_flowcell_pre_demux(ctx: StepContext, scenario: dict) -> StepResult:
    """Builds the pre-demux x_flowcells document and persists it to CouchDB.

    Reads RunInfo.xml and RunParameters.xml from hpc_runfolder_path, flattens
    demux_sample_info samplesheets into samplesheet_csv (enriched with LIMS
    fields), and upserts the result into the x_flowcells database via the
    Yggdrasil DataAccess write API.

    The document is identified by its `name` field (Mango selector). If no
    matching document exists it is created; if exactly one exists its whole body
    is replaced by this payload, keeping only its _id and _rev. Fields written by
    other processes, such as post-demux Json_Stats, are not preserved.
    """
    runfolder_path = Path(scenario["hpc_runfolder_path"])
    runfolder_id = scenario["runfolder_id"]
    samplesheets = scenario["samplesheets"]

    run_info_path = runfolder_path / RUN_INFO_XML
    run_params_path = runfolder_path / RUN_PARAMETERS_XML
    for p in (run_info_path, run_params_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    if not isinstance(samplesheets, list) or not samplesheets:
        raise ValueError("scenario['samplesheets'] must be a non-empty list.")
    for i, entry in enumerate(samplesheets):
        if not isinstance(entry, dict) or not isinstance(
            entry.get("BCLConvert_Data"), list
        ):
            raise ValueError(f"samplesheets[{i}] missing BCLConvert_Data list.")

    name = derive_xflowcell_name(runfolder_id)
    run_info = parse_run_info_xml(run_info_path)
    # <Flowcell> in RunInfo.xml is the chip serial/lot number, not the NGI flowcell ID.
    # Override with the canonical flowcell ID from the scenario.
    run_info["Flowcell"] = scenario["canonical_flowcell_id"]
    run_params = parse_run_parameters_xml(run_params_path)
    lims_lookup = build_lims_lookup(scenario.get("uploaded_lims_info", []))
    samplesheet_csv = flatten_samplesheets(
        samplesheets,
        flowcell_id=scenario["canonical_flowcell_id"],
        lims_lookup=lims_lookup,
    )
    payload = build_x_flowcell_payload(name, run_info, run_params, samplesheet_csv)

    if ctx.data is None:
        raise RuntimeError("StepContext.data (DataAccess) was not injected.")
    client = ctx.data.connection("x_flowcells_db")
    write_result = client.save(
        payload,
        selector={"name": {"$eq": name}},
        mode="upsert",
    )

    logger.info(
        "x_flowcell document '%s' %s (doc_id=%s)",
        name,
        write_result.status,
        write_result.doc_id,
    )
    return StepResult(
        metrics={
            "x_flowcell_name": name,
            "samplesheet_row_count": len(samplesheet_csv),
            "write_status": write_result.status,
            "doc_id": write_result.doc_id,
        }
    )


@step
def materialize_extra_config(ctx: StepContext, scenario: dict) -> StepResult:
    """Materializes the extra demultiplex config file."""
    config_file = ctx.workdir / DEMUX_CONFIG_FILENAME
    logger.info(f"Materializing config to {config_file}")

    # Writing a mock standard config
    config_file.write_text("process {\n  executor = 'local'\n  cpus = 4\n}")

    ctx.record_artifact(SimpleArtifactRef("demux_config", "config"), path=config_file)
    return StepResult(metrics={"config_bytes": config_file.stat().st_size})


@step
def generate_samplesheet(ctx: StepContext, 
                         scenario: dict,
                         samplesheet: Annotated[Path, Out(SAMPLESHEET)]
                         ) -> StepResult:
    """Generates and writes SampleSheet.csv from the lane-specific samplesheet payload."""
    ss_payload = scenario["samplesheet_payload"]
    # The plan is stored as JSON, so a Path param arrives here as a str.
    samplesheet = Path(samplesheet)

    validate_lane_payload(ss_payload, expected_lane=scenario.get("lane_id"))
    ss_text = render_bcl_convert_samplesheet(ss_payload)

    # Write to the path supplied by the plan.
    samplesheet.parent.mkdir(parents=True, exist_ok=True)
    samplesheet.write_text(ss_text, encoding="utf-8")

    logger.info(
        "Wrote SampleSheet.csv (%d bytes) to %s",
        samplesheet.stat().st_size,
        samplesheet,
    )
    ctx.record_artifact(SAMPLESHEET, path=samplesheet)
    return StepResult(metrics={"samplesheet_bytes": samplesheet.stat().st_size})


@step
def execute_demux(ctx: StepContext, scenario: dict, samplesheet: Annotated[Path, In(SAMPLESHEET)]) -> StepResult:
    """Runs one Nextflow nf-core/demultiplex instance for this branch.

    There is one instance per lane/samplesheet branch: the branch's
    samplesheet is the file this branch's generate_samplesheet step wrote
    into its own work directory (see _branch_samplesheet), and Nextflow
    receives work and results directories derived from the branch tag, so
    two branches never share work directories.

    Deployment settings (pipeline, profile, common config, directory bases)
    come from the realm's JSON config; options that differ between runs
    (skip_tools) come from the demux_sample_info document via the scenario's
    ``nextflow`` entry. See demux_realm.config and demux_realm.nextflow.
    """
    runfolder_path = Path(scenario["hpc_runfolder_path"])
    # The plan is stored as JSON, so a Path param arrives here as a str.
    samplesheet = Path(samplesheet)

    try:
        run_options = parse_run_options(scenario.get("nextflow"))
    except ValueError as exc:
        raise PermanentStepError(
            f"Invalid nextflow options in scenario: {exc}",
            advice="Fix the 'nextflow' object of the demux_sample_info document.",
        ) from exc

    settings = load_realm_config().nextflow
    tag = branch_run_tag(
        scenario["canonical_flowcell_id"],
        str(scenario["lane_id"]),
        str(scenario["settings_index"]),
    )
    cmd = build_nextflow_command(
        settings,
        canonical_flowcell_id=scenario["canonical_flowcell_id"],
        lane_id=str(scenario["lane_id"]),
        settings_index=str(scenario["settings_index"]),
        samplesheet=samplesheet,
        flowcell_path=runfolder_path,
        run_options=run_options,
    )
    log_path = ctx.workdir / "nextflow_run.log"
    if settings.executor == "slurm":
        return _execute_on_slurm(ctx, cmd=cmd, log_path=log_path, settings=settings, tag=tag)
    proc = run_nextflow(cmd, log_path)
    ctx.record_artifact(SimpleArtifactRef("nextflow_log", "logs"), path=log_path)

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        raise PermanentStepError(
            f"Nextflow exited with status {proc.returncode} (branch {tag}); "
            f"last log lines:\n{tail}",
            advice=f"Full log: {log_path}",
        )

    logger.info("Nextflow run finished (branch %s).", tag)
    return StepResult(
        metrics={
            "branch": tag,
            "executor": "local",
            "exit_status": proc.returncode,
            "work_dir": str(branch_work_dir(settings, tag)),
            "outdir": str(branch_outdir(settings, tag)),
        }
    )


def _tail_lines(path: Path, n: int) -> str:
    """Return up to the last n lines of a file ('' if it is missing)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-n:])


def _execute_on_slurm(
    ctx: StepContext,
    *,
    cmd: list[str],
    log_path: Path,
    settings: NextflowSettings,
    tag: str,
) -> StepResult:
    """Run the branch's Nextflow command as an SLURM batch job.

    The batch script is written to the step's work directory (on the shared
    work volume) and submitted with sbatch via the Yggdrasil toolkit's
    SlurmJobManager; the job runs on the SLURM node, whose environment
    provides Nextflow. The step blocks until the job reaches a
    terminal state, so a failed run fails this step exactly like the local
    path does.
    """
    import asyncio

    from yggdrasil.config.session import YggSession
    from yggdrasil.toolkit.sjob_manager import SlurmManagerFactory

    script_path = ctx.workdir / "nextflow_slurm.sh"
    script_path.write_text(
        build_slurm_batch_script(
            cmd,
            job_name=slurm_job_name(tag),
            log_path=log_path,
            cpus=settings.sbatch_cpus,
            mem=settings.sbatch_mem,
            time=settings.sbatch_time,
            partition=settings.sbatch_partition,
        ),
        encoding="utf-8",
    )

    manager = SlurmManagerFactory.get_manager(is_dev=YggSession.is_dev())

    async def _submit_and_wait() -> tuple[str | None, str | None]:
        job_id = await manager.submit_job(script_path)
        if job_id is None:
            return None, None
        # One loop for submit + wait: the mock backend completes its jobs on
        # the loop that submitted them (see step_slurm_job in test_realm).
        # The explicit poll interval keeps this step snappy even where
        # main.json sets job_monitor_poll_interval high.
        final_state = await manager.wait_for_completion(job_id, poll_interval=5.0)
        return job_id, final_state

    job_id, final_state = asyncio.run(_submit_and_wait())

    if job_id is None:
        raise PermanentStepError(
            f"sbatch submission failed for branch {tag}; the job script is at "
            f"{script_path}.",
            advice="Check the SLURM cluster (scontrol ping, squeue) and the "
            "yggdrasil daemon log.",
        )

    ctx.record_artifact(SimpleArtifactRef("nextflow_log", "logs"), path=log_path)
    if final_state != "COMPLETED":
        tail = _tail_lines(log_path, 20)
        raise PermanentStepError(
            f"SLURM job {job_id} ended in state {final_state or 'UNKNOWN'} "
            f"(branch {tag}); last log lines:\n{tail}",
            advice=f"Full log: {log_path}",
        )

    logger.info("SLURM job %s finished (branch %s).", job_id, tag)
    return StepResult(
        metrics={
            "branch": tag,
            "executor": "slurm",
            "job_id": job_id,
            "final_state": final_state,
            "script": str(script_path),
            "work_dir": str(branch_work_dir(settings, tag)),
            "outdir": str(branch_outdir(settings, tag)),
        }
    )


@step
def collect_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Collects demultiplexing artifacts and basic metrics from the branch's results directory."""
    settings = load_realm_config().nextflow
    tag = branch_run_tag(
        scenario["canonical_flowcell_id"],
        str(scenario["lane_id"]),
        str(scenario["settings_index"]),
    )
    outdir = branch_outdir(settings, tag)
    if not outdir.is_dir():
        raise PermanentStepError(
            f"Results directory missing: {outdir}",
            advice="Did the branch's execute_demux step succeed?",
        )

    fastqs = sorted(outdir.rglob("*.fastq.gz"))
    multiqc_reports = sorted(outdir.rglob("multiqc_report.html"))
    for report in multiqc_reports:
        ctx.record_artifact(SimpleArtifactRef("multiqc_report", "reports"), path=report)

    logger.info(
        "Collected %d fastq file(s) from %s", len(fastqs), outdir
    )
    return StepResult(
        metrics={
            "outdir": str(outdir),
            "fastq_count": len(fastqs),
            "multiqc_report_count": len(multiqc_reports),
        }
    )


@step
def upload_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Uploads metrics and artifacts to appropriate endpoints (simulated)."""
    logger.info("Uploading metrics to remote API...")
    scenario_metadata = scenario.get("demux_sample_info_doc", {}).get("metadata", {})
    logger.info(
        f"Simulating upload for run {scenario['runfolder_id']} with metadata {scenario_metadata}"
    )

    return StepResult(metrics={"upload_status": "simulated_success"})
