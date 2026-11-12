import csv
import logging
import re
from pathlib import Path
from typing import Annotated, Any

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


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _branch_reports_dir(outdir: Path, canonical_flowcell_id: str) -> Path:
    """Return the branch's Reports directory: outdir/<FCID>/Reports."""
    return outdir / canonical_flowcell_id / "Reports"


def _load_demux_metadata(reports_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Return the parsed Demultiplex_Stats.csv / Quality_Metrics.csv / Top_Unknown_Barcodes.csv rows.

    `reports_dir` must already point at the branch's Reports directory (see
    `_branch_reports_dir`); it is not appended here.
    """
    metadata: dict[str, list[dict[str, str]]] = {}
    for filename in (
        "Demultiplex_Stats.csv",
        "Quality_Metrics.csv",
        "Top_Unknown_Barcodes.csv",
    ):
        rows = _read_csv_rows(reports_dir / filename)
        if rows:
            metadata[filename.removesuffix(".csv")] = rows
    return metadata


# Matches branch fields written by upload_results, e.g. "Demultiplex_lane_1_settings_0".
_BRANCH_FIELD_RE = re.compile(r"^Demultiplex_lane_.+_settings_.+$")


def _parse_int(value: str | None) -> int:
    """Parse a CSV integer cell, tolerating thousands separators and blanks."""
    text = (value or "").replace(",", "").strip()
    return int(text) if text else 0


def _parse_float(value: str | None) -> float:
    """Parse a CSV numeric cell, tolerating thousands separators and blanks."""
    text = (value or "").replace(",", "").strip()
    return float(text) if text else 0.0


def _fmt_int(value: float) -> str:
    """Format a count with thousands separators, e.g. 2101589 -> '2,101,589'."""
    return f"{round(value):,}"


def _fmt_pct(value: float) -> str:
    """Format a percentage/score to 2 decimal places, e.g. 98.2 -> '98.20'."""
    return f"{value:.2f}"


def _format_barcode_sequence(index: str) -> str:
    """Format a Demultiplex_Stats.csv "Index" value as "index1+index2" when
    dual-indexed, or just the single index when only one side is present.

    e.g. "ATCG-GCTA" -> "ATCG+GCTA", "ATCG-" -> "ATCG", "-GCTA" -> "GCTA".
    """
    parts = [part for part in index.split("-") if part]
    return "+".join(parts)


def _combine_quality_metrics(
    quality_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Sum Quality_Metrics.csv rows per (Lane, SampleID), read 1 and 2 only.

    Quality_Metrics.csv has up to one row per ReadNumber (1, I1, I2, 2); index
    reads (I1/I2) are excluded so Yield/YieldQ30/QualityScoreSum reflect only
    the actual sequencing reads.
    """
    combined: dict[tuple[str, str], dict[str, float]] = {}
    for row in quality_rows:
        if row.get("ReadNumber") not in ("1", "2"):
            continue
        key = (row.get("Lane", ""), row.get("SampleID", ""))
        bucket = combined.setdefault(
            key, {"yield": 0.0, "yield_q30": 0.0, "quality_sum": 0.0}
        )
        bucket["yield"] += _parse_float(row.get("Yield"))
        bucket["yield_q30"] += _parse_float(row.get("YieldQ30"))
        bucket["quality_sum"] += _parse_float(row.get("QualityScoreSum"))
    return combined


def _build_illumina_stats(
    demux_stats_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    sample_project: dict[str, str],
) -> dict[str, Any]:
    """Combine Demultiplex_Stats.csv and Quality_Metrics.csv rows (all lanes)
    into the legacy 'illumina.Demultiplex_Stats' document shape.

    Limitation: Demultiplex_Stats.csv/Quality_Metrics.csv only report reads
    that already passed filter, so there is no pre-filter ("raw") cluster
    count available from these two files. "% PFClusters" is reused from
    "% Reads" at the sample level; at the lane/flowcell level, where no
    equivalent column exists, raw is set to 0 as a placeholder until we
    know if its used anywhere.
    """
    quality_by_sample = _combine_quality_metrics(quality_rows)

    barcode_lane_statistics: list[dict[str, str]] = []
    lane_totals: dict[str, dict[str, float]] = {}

    for row in demux_stats_rows:
        lane = row.get("Lane", "")
        sample = row.get("SampleID", "")
        reads = _parse_int(row.get("# Reads"))
        perfect = _parse_int(row.get("# Perfect Index Reads"))
        one_mismatch = _parse_int(row.get("# One Mismatch Index Reads"))
        pct_reads = _parse_float(row.get("% Reads"))
        pct_perfect = _parse_float(row.get("% Perfect Index Reads"))
        pct_one_mismatch = _parse_float(row.get("% One Mismatch Index Reads"))

        qm = quality_by_sample.get(
            (lane, sample), {"yield": 0.0, "yield_q30": 0.0, "quality_sum": 0.0}
        )
        yield_bases = qm["yield"]
        mean_quality = (qm["quality_sum"] / yield_bases) if yield_bases else 0.0
        pct_q30 = (qm["yield_q30"] / yield_bases * 100) if yield_bases else 0.0

        barcode_lane_statistics.append(
            {
                "Lane": lane,
                "Sample": sample,
                "Barcode sequence": _format_barcode_sequence(row.get("Index") or "")
                or "unknown",
                "Project": sample_project.get(sample, "default"),
                "PF Clusters": _fmt_int(reads),
                "% of thelane": _fmt_pct(pct_reads),
                "% PFClusters": _fmt_pct(pct_reads),
                "% Perfectbarcode": _fmt_pct(pct_perfect),
                "% One mismatchbarcode": _fmt_pct(pct_one_mismatch),
                "Yield (Mbases)": _fmt_int(yield_bases / 1_000_000),
                "Mean QualityScore": _fmt_pct(mean_quality),
                "% >= Q30bases": _fmt_pct(pct_q30),
            }
        )

        totals = lane_totals.setdefault(
            lane,
            {
                "reads": 0.0,
                "perfect": 0.0,
                "one_mismatch": 0.0,
                "yield": 0.0,
                "yield_q30": 0.0,
                "quality_sum": 0.0,
            },
        )
        totals["reads"] += reads
        totals["perfect"] += perfect
        totals["one_mismatch"] += one_mismatch
        totals["yield"] += yield_bases
        totals["yield_q30"] += qm["yield_q30"]
        totals["quality_sum"] += qm["quality_sum"]

    lanes_stats: list[dict[str, str]] = []
    total_reads = 0.0
    total_yield = 0.0
    for lane, totals in lane_totals.items():
        reads = totals["reads"]
        yield_bases = totals["yield"]
        lanes_stats.append(
            {
                "Lane": lane,
                "PF Clusters": _fmt_int(reads),
                "% of thelane": _fmt_pct(100.0),
                "% Perfectbarcode": _fmt_pct(
                    (totals["perfect"] / reads * 100) if reads else 0.0
                ),
                "% One mismatchbarcode": _fmt_pct(
                    (totals["one_mismatch"] / reads * 100) if reads else 0.0
                ),
                "Yield (Mbases)": _fmt_int(yield_bases / 1_000_000),
                # No raw (pre-filter) cluster count is available from these CSVs; see
                # the docstring above for the placeholder rationale.
                "% PFClusters": _fmt_pct(100.0),
                "% >= Q30bases": _fmt_pct(
                    (totals["yield_q30"] / yield_bases * 100) if yield_bases else 0.0
                ),
                "Mean QualityScore": _fmt_pct(
                    (totals["quality_sum"] / yield_bases) if yield_bases else 0.0
                ),
            }
        )
        total_reads += reads
        total_yield += yield_bases

    flowcell_stats = {
        # Clusters (Raw) == "0": placeholder
        "Clusters (Raw)": "0",
        "Clusters(PF)": _fmt_int(total_reads),
        "Yield (MBases)": _fmt_int(total_yield / 1_000_000),
    }

    return {
        "Demultiplex_Stats": {
            "Barcode_lane_statistics": barcode_lane_statistics,
            "Flowcell_stats": flowcell_stats,
            "Lanes_stats": lanes_stats,
        }
    }


def _build_undetermined_stats(
    top_unknown_rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """Group Top_Unknown_Barcodes.csv rows (all lanes) into the
    {lane: {barcode: read_count}} shape expected under the document's
    "Undetermined" field.

    Barcode keys combine index/index2 as "<index>+<index2>" (or just
    "<index>" when there is no index2). Duplicate (Lane, barcode) rows -
    which should not normally occur within a single branch - are summed.
    """
    undetermined: dict[str, dict[str, int]] = {}
    for row in top_unknown_rows:
        lane = row.get("Lane", "")
        index1 = row.get("index", "")
        index2 = row.get("index2", "")
        barcode = f"{index1}+{index2}" if index2 else index1
        reads = _parse_int(row.get("# Reads"))
        lane_bucket = undetermined.setdefault(lane, {})
        lane_bucket[barcode] = lane_bucket.get(barcode, 0) + reads
    return undetermined


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
def setup_common_demux_dir(ctx: StepContext, scenario: dict) -> StepResult:
    """Create the common demux directory for all branches."""
    hpc_path = Path(scenario["hpc_runfolder_path"])
    demux_dir = hpc_path / "Demultiplexing"
    demux_dir.mkdir(exist_ok=True)
    logger.info("Created common demux directory %s.", demux_dir)
    return StepResult(
        metrics={
            "demux_dir": str(demux_dir),
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
def generate_samplesheet(
    ctx: StepContext, scenario: dict, samplesheet: Annotated[Path, Out(SAMPLESHEET)]
) -> StepResult:
    """Generates and writes SampleSheet.csv from the lane-specific samplesheet payload."""
    ss_payload = scenario["samplesheet_payload"]

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
def execute_demux(
    ctx: StepContext, scenario: dict, samplesheet: Annotated[Path, In(SAMPLESHEET)]
) -> StepResult:
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
    outdir = branch_outdir(
        settings, tag
    )  # Should be FCID/Demultiplex_lane_<n>_settings_<i>
    if not outdir.is_dir():
        raise PermanentStepError(
            f"Results directory missing: {outdir}",
            advice="Did the branch's execute_demux step succeed?",
        )

    stats_dir = _branch_reports_dir(outdir, scenario["canonical_flowcell_id"])
    if not stats_dir.is_dir():
        raise PermanentStepError(
            f"Reports directory missing: {stats_dir}",
            advice="Did the branch's execute_demux step succeed?",
        )
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

    scenario["demultiplexing_metadata"] = _load_demux_metadata(stats_dir)
    return StepResult(
        metrics={
            "outdir": str(outdir),
            "demultiplexing_metadata_files": list(scenario["demultiplexing_metadata"]),
        }
    )


# TODO: Add a step to aggregate demux stats for lanes with multiple demuxes


@step
def symlink_fastqs(ctx: StepContext, scenario: dict) -> StepResult:
    """Symlinks fastq files from the branch's results directory to the common demux directory."""
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
    results_dir = outdir / scenario["canonical_flowcell_id"]
    fastqs = sorted(results_dir.rglob("*.fastq.gz"))
    demux_dir = Path(scenario["hpc_runfolder_path"]) / "Demultiplexing"
    for fq in fastqs:
        symlink_path = (
            demux_dir / fq.name
        )  # TODO: should be demux_dir / project / sample / fq.name
        if not symlink_path.exists():
            symlink_path.symlink_to(fq)
            logger.info("Symlinked %s to %s", fq, symlink_path)
        else:
            logger.warning("Symlink already exists: %s", symlink_path)

    return StepResult(
        metrics={
            "symlinked_fastq_count": len(fastqs),
            "demux_dir": str(demux_dir),
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
        settings = load_realm_config().nextflow
        tag = branch_run_tag(
            scenario["canonical_flowcell_id"],
            str(scenario["lane_id"]),
            str(scenario["settings_index"]),
        )
        outdir = branch_outdir(settings, tag)
        metadata = _load_demux_metadata(
            _branch_reports_dir(outdir, scenario["canonical_flowcell_id"])
        )

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
    # existing may carry CouchDB's own _id/_rev; DataAccess.save() manages
    # those internally via the selector (fetching the current _rev and
    # retrying on a 409 conflict itself), so the caller's job is only to
    # strip them before passing the body back in — never to resolve conflicts.
    if hasattr(client, "clean_doc"):
        updated_doc = client.clean_doc(existing)
    else:
        updated_doc = {k: v for k, v in existing.items() if k not in ("_id", "_rev")}
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


@step
def aggregate_illumina_stats(ctx: StepContext, scenario: dict) -> StepResult:
    """Recompute the combined 'illumina' demux stats field from every branch
    already stored on the x_flowcells document, and save it back.

    This is a common step, run once after all lane/settings branches'
    upload_results steps (not a per-branch step): it reads every
    Demultiplex_lane_<lane>_settings_<idx> field already persisted on the
    document - including branches from earlier runs that were not rerun this
    time - and rebuilds illumina.Demultiplex_Stats from their union. That
    makes a rerun of a single lane safe: the aggregate always reflects
    whatever is currently stored for every lane, not just the lanes that ran
    in this invocation. Running this per-branch instead would require
    merging a shared array field from concurrently-completing branches,
    which risks lost updates/duplicate rows under _rev conflicts.
    """
    if ctx.data is None:
        raise RuntimeError("StepContext.data (DataAccess) was not injected.")

    runfolder_id = scenario["runfolder_id"]
    name = derive_xflowcell_name(runfolder_id)
    client = ctx.data.connection("x_flowcells_db")

    existing: dict[str, Any] | None = None
    if hasattr(client, "find_one"):
        existing = client.find_one({"name": {"$eq": name}})
    elif hasattr(client, "get"):
        existing = client.get(name)
    if existing is None:
        raise PermanentStepError(
            f"x_flowcells document '{name}' not found.",
            advice="Did upsert_x_flowcell_pre_demux run for this flowcell?",
        )
    if not isinstance(existing, dict):
        raise TypeError(f"x_flowcells lookup returned a non-dict value: {existing!r}")

    demux_stats_rows: list[dict[str, str]] = []
    quality_rows: list[dict[str, str]] = []
    for key, value in existing.items():
        if not _BRANCH_FIELD_RE.match(key) or not isinstance(value, dict):
            continue
        demux_stats_rows.extend(value.get("Demultiplex_Stats") or [])
        quality_rows.extend(value.get("Quality_Metrics") or [])

    sample_project = {
        row["Sample_ID"]: row.get("Sample_Project", "default")
        for row in existing.get("samplesheet_csv") or []
        if row.get("Sample_ID")
    }

    illumina = _build_illumina_stats(demux_stats_rows, quality_rows, sample_project)

    if hasattr(client, "clean_doc"):
        updated_doc = client.clean_doc(existing)
    else:
        updated_doc = {k: v for k, v in existing.items() if k not in ("_id", "_rev")}
    updated_doc["illumina"] = illumina

    write_result = client.save(
        updated_doc,
        selector={"name": {"$eq": name}},
        mode="upsert",
    )
    logger.info(
        "Updated x_flowcell document '%s' illumina stats %s (doc_id=%s)",
        name,
        write_result.status,
        write_result.doc_id,
    )
    return StepResult(
        metrics={
            "x_flowcell_name": name,
            "lane_count": len(
                {row["Lane"] for row in demux_stats_rows if row.get("Lane")}
            ),
            "sample_row_count": len(demux_stats_rows),
            "doc_id": getattr(write_result, "doc_id", None),
        }
    )


@step
def aggregate_undetermined_stats(ctx: StepContext, scenario: dict) -> StepResult:
    """Recompute the combined 'Undetermined' demux stats field from every
    branch already stored on the x_flowcells document, and save it back.

    This is a common step, run once after all lane/settings branches'
    upload_results steps (same pattern as aggregate_illumina_stats): it reads
    every Demultiplex_lane_<lane>_settings_<idx> field's Top_Unknown_Barcodes
    rows already persisted on the document - including branches from earlier
    runs that were not rerun this time - and rebuilds Undetermined from their
    union, keyed by lane then by "<index>+<index2>" barcode with its read
    count. That makes a rerun of a single lane safe: the aggregate always
    reflects whatever is currently stored for every lane, not just the lanes
    that ran in this invocation.
    """
    if ctx.data is None:
        raise RuntimeError("StepContext.data (DataAccess) was not injected.")

    runfolder_id = scenario["runfolder_id"]
    name = derive_xflowcell_name(runfolder_id)
    client = ctx.data.connection("x_flowcells_db")

    existing: dict[str, Any] | None = None
    if hasattr(client, "find_one"):
        existing = client.find_one({"name": {"$eq": name}})
    elif hasattr(client, "get"):
        existing = client.get(name)
    if existing is None:
        raise PermanentStepError(
            f"x_flowcells document '{name}' not found.",
            advice="Did upsert_x_flowcell_pre_demux run for this flowcell?",
        )
    if not isinstance(existing, dict):
        raise TypeError(f"x_flowcells lookup returned a non-dict value: {existing!r}")

    top_unknown_rows: list[dict[str, str]] = []
    for key, value in existing.items():
        if not _BRANCH_FIELD_RE.match(key) or not isinstance(value, dict):
            continue
        top_unknown_rows.extend(value.get("Top_Unknown_Barcodes") or [])

    undetermined = _build_undetermined_stats(top_unknown_rows)

    if hasattr(client, "clean_doc"):
        updated_doc = client.clean_doc(existing)
    else:
        updated_doc = {k: v for k, v in existing.items() if k not in ("_id", "_rev")}
    updated_doc["Undetermined"] = undetermined

    write_result = client.save(
        updated_doc,
        selector={"name": {"$eq": name}},
        mode="upsert",
    )
    logger.info(
        "Updated x_flowcell document '%s' Undetermined stats %s (doc_id=%s)",
        name,
        write_result.status,
        write_result.doc_id,
    )
    return StepResult(
        metrics={
            "x_flowcell_name": name,
            "lane_count": len(undetermined),
            "barcode_row_count": len(top_unknown_rows),
            "doc_id": getattr(write_result, "doc_id", None),
        }
    )
