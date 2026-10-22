import logging
from pathlib import Path

from yggdrasil.flow.artifacts import SimpleArtifactRef
from yggdrasil.flow.errors import PermanentStepError
from yggdrasil.flow.model import StepResult
from yggdrasil.flow.step import StepContext, step

from .nextflow import (
    branch_outdir,
    branch_run_tag,
    branch_work_dir,
    build_nextflow_command,
    run_nextflow,
    settings_from_env,
)
from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    SAMPLESHEET_FILENAME,
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


@step
def validate_runfolder(ctx: StepContext, scenario: dict) -> StepResult:
    """Validates that the runfolder exists in the HPC.

    TODO: verify that the FCID derived from RunInfo.xml (<Run Id> last '_'-separated
    segment) matches scenario['canonical_flowcell_id'] (sourced from flowcell_status
    and demux_sample_info) to catch runfolder/sample-sheet mismatches early.
    """
    hpc_path = scenario["hpc_runfolder_path"]
    logger.info(f"Validating expected HPC runfolder path: {hpc_path}")

    # Normally we would os.path.isdir(hpc_path) or ssh verify it
    # We will pretend it exists successfully
    return StepResult(metrics={"path_validated": hpc_path})


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
def generate_samplesheet(ctx: StepContext, scenario: dict) -> StepResult:
    """Generates and writes SampleSheet.csv from the lane-specific samplesheet payload."""
    ss_payload = scenario["samplesheet_payload"]

    validate_lane_payload(ss_payload, expected_lane=scenario.get("lane_id"))
    ss_text = render_bcl_convert_samplesheet(ss_payload)

    ss_file = ctx.workdir / SAMPLESHEET_FILENAME
    ss_file.write_text(ss_text)

    logger.info(
        "Wrote SampleSheet.csv (%d bytes) to %s", ss_file.stat().st_size, ss_file
    )
    ctx.record_artifact(SimpleArtifactRef("samplesheet", "samplesheets"), path=ss_file)
    return StepResult(metrics={"samplesheet_bytes": ss_file.stat().st_size})


@step
def execute_demux(ctx: StepContext, scenario: dict) -> StepResult:
    """Runs one Nextflow nf-core/demultiplex instance for this branch.

    There is one instance per lane/samplesheet branch: the branch's
    samplesheet is re-rendered from the scenario payload into this step's
    own workdir (runs are self-contained and never share files), and
    Nextflow receives work and results directories derived from the branch
    tag, so two branches never share work directories. Deployment settings
    (pipeline, profile, common config, directory bases, skipped tools) come
    from the DMX_NF_* environment; see demux_realm.nextflow.
    """
    runfolder_path = Path(scenario["hpc_runfolder_path"])

    # Re-render the branch samplesheet here rather than reading the
    # generate_samplesheet step's workdir: cross-step paths are engine
    # internals, and rendering is a pure function of the scenario payload.
    ss_file = ctx.workdir / SAMPLESHEET_FILENAME
    ss_file.write_text(render_bcl_convert_samplesheet(scenario["samplesheet_payload"]))
    ctx.record_artifact(
        SimpleArtifactRef("pipeline_samplesheet", "samplesheets"), path=ss_file
    )

    settings = settings_from_env()
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
        samplesheet=ss_file,
        flowcell_path=runfolder_path,
    )
    log_path = ctx.workdir / "nextflow_run.log"
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
            "exit_status": proc.returncode,
            "work_dir": str(branch_work_dir(settings, tag)),
            "outdir": str(branch_outdir(settings, tag)),
        }
    )


@step
def collect_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Collects demultiplexing artifacts and basic metrics from the branch's results directory."""
    settings = settings_from_env()
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
