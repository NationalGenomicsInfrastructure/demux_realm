import logging
from pathlib import Path

from yggdrasil.flow.artifacts import SimpleArtifactRef
from yggdrasil.flow.model import StepResult
from yggdrasil.flow.step import StepContext, step

from .utils import (
    DEMUX_CONFIG_FILENAME,
    RUN_INFO_XML,
    RUN_PARAMETERS_XML,
    SAMPLESHEET_FILENAME,
    build_lims_lookup,
    build_x_flowcell_payload,
    derive_xflowcell_name,
    flatten_samplesheets,
    normalize_flowcell_id,
    parse_run_info_xml,
    parse_run_parameters_xml,
    render_bcl_convert_samplesheet,
    validate_lane_payload,
)

logger = logging.getLogger(__name__)


@step
def validate_runfolder(ctx: StepContext, scenario: dict) -> StepResult:
    """Validate that the runfolder is present and matches the canonical FCID.

    The FCID is derived from the last '_'-separated part of RunInfo.xml's
    <Run Id> field (e.g. ASC2177698-SC3 → SC2177698-SC3) and must match the
    scenario's canonical flowcell ID. This catches mismatched runfolders before
    demultiplexing proceeds.
    """
    hpc_path = Path(scenario["hpc_runfolder_path"])
    if not hpc_path.exists():
        raise FileNotFoundError(f"HPC runfolder not found: {hpc_path}")
    if not hpc_path.is_dir():
        raise NotADirectoryError(f"HPC runfolder path is not a directory: {hpc_path}")

    canonical_fcid = scenario.get("canonical_flowcell_id")
    if not canonical_fcid:
        raise ValueError("scenario['canonical_flowcell_id'] is required.")

    run_info_path = hpc_path / RUN_INFO_XML
    if not run_info_path.exists():
        raise FileNotFoundError(f"Required file missing: {run_info_path}")

    run_info = parse_run_info_xml(run_info_path)
    run_id = str(run_info.get("Id", "")).strip()
    if not run_id or "_" not in run_id:
        raise ValueError(f"RunInfo.xml <Run Id> is malformed: {run_id!r}")

    derived_fcid = normalize_flowcell_id(run_id.rsplit("_", 1)[-1])
    expected_fcid = normalize_flowcell_id(canonical_fcid)
    if derived_fcid != expected_fcid:
        raise ValueError(
            "FCID mismatch: RunInfo.xml derived FCID "
            f"'{derived_fcid}' from '{run_id}' does not match canonical flowcell ID "
            f"'{expected_fcid}'."
        )

    logger.info(
        "Validated HPC runfolder %s against canonical flowcell ID %s",
        hpc_path,
        expected_fcid,
    )
    return StepResult(
        metrics={
            "path_validated": str(hpc_path),
            "flowcell_id_validated": expected_fcid,
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
    """Simulates Nextflow pipeline execution."""
    logger.info("Executing demux pipeline (Nextflow)...")
    logger.info(f"Using runfolder: {scenario['hpc_runfolder_path']}")
    logger.info("Simulation mode: Pipeline executed successfully.")

    return StepResult(metrics={"execution_status": "simulated_success"})


@step
def collect_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Collects demultiplexing metrics, artifacts and logs."""
    logger.info("Collecting metrics from demux result directory...")

    return StepResult(metrics={"yield": 120000000, "q30_percentage": 98.4})


@step
def upload_results(ctx: StepContext, scenario: dict) -> StepResult:
    """Uploads metrics and artifacts to appropriate endpoints (simulated)."""
    logger.info("Uploading metrics to remote API...")
    scenario_metadata = scenario.get("demux_sample_info_doc", {}).get("metadata", {})
    logger.info(
        f"Simulating upload for run {scenario['runfolder_id']} with metadata {scenario_metadata}"
    )

    return StepResult(metrics={"upload_status": "simulated_success"})
