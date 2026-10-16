"""Launching nf-core/demultiplex (Nextflow) from the realm.

One Nextflow run is launched per lane/samplesheet branch of the demux plan;
the two never share work or result directories.

Deployment-specific values come from the environment (DMX_NF_* variables,
following the DMX_HPC_BASE_PATH convention). The defaults target the local
compose/DinD development stack; the HPC deployment overrides them (e.g. a
provisioned, offline pipeline path and an HPC-sized resource config). No
value here is per-run: everything per-run is derived from the step scenario.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_PIPELINE = "DMX_NF_PIPELINE"
ENV_PROFILE = "DMX_NF_PROFILE"
ENV_CONFIG = "DMX_NF_CONFIG"
ENV_WORK_BASE = "DMX_NF_WORK_BASE"
ENV_OUTDIR_BASE = "DMX_NF_OUTDIR_BASE"
ENV_SKIP_TOOLS = "DMX_NF_SKIP_TOOLS"
ENV_EXTRA_ARGS = "DMX_NF_EXTRA_ARGS"


@dataclass(frozen=True)
class NextflowSettings:
    """Deployment-level Nextflow settings (not per-run)."""

    #: Pipeline to run: an nf-core name (resolved online, dev) or the path
    #: of a provisioned local repository (offline HPC).
    pipeline: str
    #: Nextflow profile, e.g. "docker" (DinD dev) or a Slurm/singularity
    #: profile on the HPC.
    profile: str
    #: Optional common config file passed with -c (e.g. HPC-sized resource
    #: requirements). Empty means none.
    common_config: Path | None
    #: Base directory for per-branch Nextflow work directories.
    work_base: Path
    #: Base directory for per-branch published results (--outdir).
    outdir_base: Path
    #: Tools to skip, e.g. ("samshee",) while its samplesheet validation is
    #: out of date. Empty means none.
    skip_tools: tuple[str, ...]
    #: Extra nextflow arguments appended verbatim (last, so they win).
    extra_args: tuple[str, ...]


def settings_from_env(env: Mapping[str, str] | None = None) -> NextflowSettings:
    """Build NextflowSettings from DMX_NF_* environment variables.

    Defaults target the local compose/DinD development stack:
    nextflow resolves the pipeline online (cached under NXF_HOME) and work
    and results live on the shared /work volume.

    Args:
        env: Environment mapping; defaults to os.environ.

    Returns:
        The resolved settings.
    """
    env = os.environ if env is None else env

    def split_csv(value: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in value.split(",") if part.strip())

    common_config = env.get(ENV_CONFIG, "").strip()
    return NextflowSettings(
        pipeline=env.get(ENV_PIPELINE, "nf-core/demultiplex").strip(),
        profile=env.get(ENV_PROFILE, "docker").strip(),
        common_config=Path(common_config) if common_config else None,
        work_base=Path(env.get(ENV_WORK_BASE, "/work/nf-realm-work")),
        outdir_base=Path(env.get(ENV_OUTDIR_BASE, "/work/nf-realm-results")),
        skip_tools=split_csv(env.get(ENV_SKIP_TOOLS, "samshee")),
        extra_args=tuple(shlex.split(env.get(ENV_EXTRA_ARGS, ""))),
    )


def branch_run_tag(
    canonical_flowcell_id: str, lane_id: str, settings_index: str
) -> str:
    """Return the per-branch path tag, e.g. SC2177698-SC3/lane_2_settings_0.

    The tag mirrors the plan's branch namespace (lane_<n>_settings_<i>) so a
    branch's directories are always unique within a flowcell and stable
    across re-runs of the same branch.
    """
    return f"{canonical_flowcell_id}/lane_{lane_id}_settings_{settings_index}"


def sanitize_run_name(raw: str) -> str:
    """Sanitize a string into a valid Nextflow run name.

    Nextflow requires ^[a-z](?:[a-z\\d]|[-_](?=[a-z\\d])){0,79}$: lowercase,
    starting with a letter, at most 80 characters, and no leading/trailing or
    consecutive separators. Flowcell ids are typically upper-case and start
    with digits, so they never pass through unchanged.
    """
    name = re.sub(r"[^a-z0-9_-]", "_", raw.lower())
    name = re.sub(r"[_-]+", "_", name).strip("_-")
    if not name or not name[0].isalpha():
        name = f"run_{name}"
    return name[:80].rstrip("_-")


def branch_work_dir(settings: NextflowSettings, tag: str) -> Path:
    """Return the branch's Nextflow work directory (unique per branch)."""
    return settings.work_base / tag


def branch_outdir(settings: NextflowSettings, tag: str) -> Path:
    """Return the branch's published-results directory (unique per branch)."""
    return settings.outdir_base / tag


def build_nextflow_command(
    settings: NextflowSettings,
    *,
    canonical_flowcell_id: str,
    lane_id: str,
    settings_index: str,
    samplesheet: Path,
    flowcell_path: Path,
) -> list[str]:
    """Build the argv of one per-branch Nextflow run.

    Per-run values come from the branch (flowcell, lane, the rendered
    samplesheet and the runfolder path); per-branch work and result
    directories are derived from the settings so two branches never share
    them. -resume is included so a retried step reuses its own branch's
    completed tasks (and only those).

    Returns:
        The command as a list (no shell involved).
    """
    tag = branch_run_tag(canonical_flowcell_id, lane_id, settings_index)
    cmd = [
        "nextflow",
        "run",
        settings.pipeline,
        # Pipeline parameters (double dash).
        "--flowcell_id",
        canonical_flowcell_id,
        "--flowcell_samplesheet",
        str(samplesheet),
        "--flowcell_path",
        str(flowcell_path),
    ]
    if settings.skip_tools:
        cmd += ["--skip_tools", ",".join(settings.skip_tools)]
    if settings.common_config is not None:
        # Nextflow CLI options (single dash); later -c files override earlier
        # ones, so the common config comes before the per-run values would.
        cmd += ["-c", str(settings.common_config)]
    cmd += [
        "-profile",
        settings.profile,
        "-work-dir",
        str(branch_work_dir(settings, tag)),
        "--outdir",
        str(branch_outdir(settings, tag)),
        "-name",
        sanitize_run_name(tag.replace("/", "_")),
        "-resume",
    ]
    cmd += list(settings.extra_args)
    return cmd


def run_nextflow(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    """Run a Nextflow command, teeing its combined output to log_path.

    Args:
        cmd: The argv from build_nextflow_command.
        log_path: Where to write the combined stdout/stderr (created).

    Returns:
        The CompletedProcess (caller inspects returncode).
    """
    logger.info("Running: %s", " ".join(shlex.quote(part) for part in cmd))
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout or "")
    return proc
