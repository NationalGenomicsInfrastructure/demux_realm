import csv
import logging
from pathlib import Path
from typing import Any

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


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_demux_metadata(outdir: Path) -> dict[str, list[dict[str, str]]]:
    """Return the parsed Demultiplex_Stats.csv / Quality_Metrics.csv / Top_Unknown_Barcodes.csv rows."""
    stats_dir = outdir / "Stats"
    metadata: dict[str, list[dict[str, str]]] = {}
    for filename in (
        "Demultiplex_Stats.csv",
        "Quality_Metrics.csv",
        "Top_Unknown_Barcodes.csv",
    ):
        rows = _read_csv_rows(stats_dir / filename)
        if rows:
            metadata[filename.removesuffix(".csv")] = rows
    return metadata


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
    if not runfolder_path.is_dir():
        raise PermanentStepError(
            f"Runfolder path does not exist: {runfolder_path}",
            advice="Check hpc_runfolder_path / DMX_HPC_BASE_PATH.",
        )

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
    outdir = branch_outdir(
        settings, tag
    )  # Should be FCID/Demultiplex_lane_<n>_settings_<i>
    if not outdir.is_dir():
        raise PermanentStepError(
            f"Results directory missing: {outdir}",
            advice="Did the branch's execute_demux step succeed?",
        )

    fastqs = sorted(outdir.rglob("*.fastq.gz"))

    stats_dir = outdir / "Stats"
    if not stats_dir.is_dir():
        logger.warning("Stats directory missing: %s", stats_dir)
    demultiplex_stats = stats_dir / "Demultiplex_Stats.csv"
    quality_metrics = stats_dir / "Quality_Metrics.csv"
    top_unknown_barcodes = stats_dir / "Top_Unknown_Barcodes.csv"

    if demultiplex_stats.is_file():
        ctx.record_artifact(
            SimpleArtifactRef("demultiplex_stats", "reports"), path=demultiplex_stats
        )
    if quality_metrics.is_file():
        ctx.record_artifact(
            SimpleArtifactRef("quality_metrics", "reports"), path=quality_metrics
        )
    if top_unknown_barcodes.is_file():
        ctx.record_artifact(
            SimpleArtifactRef("top_unknown_barcodes", "reports"),
            path=top_unknown_barcodes,
        )

    scenario["demultiplexing_metadata"] = _load_demux_metadata(outdir)
    return StepResult(
        metrics={
            "outdir": str(outdir),
            "fastq_count": len(fastqs),
            "demultiplexing_metadata_files": list(scenario["demultiplexing_metadata"]),
        }
    )


@step
def upload_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Fetch the current x_flowcells doc, add this branch's demux CSV metadata
    under its own field, and save it back.

    Each lane/settings branch publishes its Demultiplex_Stats.csv,
    Quality_Metrics.csv and Top_Unknown_Barcodes.csv rows under a dedicated
    top-level field named Demultiplex_lane_<lane_id>_settings_<settings_index>.
    Since the field is unique per branch, branches never collide and a
    branch can be re-run without touching any other branch's results.
    """
    logger.info("Uploading metrics to remote API...")
    if ctx.data is None:
        raise RuntimeError("StepContext.data (DataAccess) was not injected.")

    runfolder_id = scenario["runfolder_id"]
    name = derive_xflowcell_name(runfolder_id)
    metadata = scenario.get("demultiplexing_metadata")
    if metadata is None:
        settings = settings_from_env()
        tag = branch_run_tag(
            scenario["canonical_flowcell_id"],
            str(scenario["lane_id"]),
            str(scenario["settings_index"]),
        )
        metadata = _load_demux_metadata(branch_outdir(settings, tag))

    client = ctx.data.connection("x_flowcells_db")
    existing: dict[str, Any] | None = None
    if hasattr(client, "find_one"):
        existing = client.find_one({"name": {"$eq": name}})
    elif hasattr(client, "get"):
        existing = client.get(name)
    if existing is None:
        existing = {"name": name}
    elif not isinstance(existing, dict):
        raise TypeError(f"x_flowcells lookup returned a non-dict value: {existing!r}")

    branch_field = (
        f"Demultiplex_lane_{scenario['lane_id']}_settings_{scenario['settings_index']}"
    )
    updated_doc = dict(existing)
    updated_doc[branch_field] = metadata

    write_result = client.save(
        updated_doc,
        selector={"name": {"$eq": name}},
        mode="upsert",
    )
    logger.info(
        "Updated x_flowcell document '%s' %s (doc_id=%s)",
        name,
        write_result.status,
        write_result.doc_id,
    )
    return StepResult(
        metrics={
            "upload_status": "simulated_success",
            "x_flowcell_name": name,
            "demultiplex_field": branch_field,
            "demultiplexing_metadata_files": list(metadata),
            "doc_id": getattr(write_result, "doc_id", None),
        }
    )
