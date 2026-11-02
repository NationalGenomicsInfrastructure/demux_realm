"""Launching nf-core/demultiplex (Nextflow) from the realm.

One Nextflow run is launched per lane/samplesheet branch of the demux plan;
the two never share work or result directories.

Two kinds of input shape a run:

* NextflowSettings: deployment-level values (pipeline, profile, directory
  bases, ...) read from the realm's JSON config (see demux_realm.config).
  The defaults target the local compose/DinD development stack; the HPC
  deployment overrides them (e.g. a provisioned, offline pipeline path and
  an HPC-sized resource config).
* NextflowRunOptions: values that may differ between samplesheets (e.g. the
  tools to skip). Resolved per branch by resolve_run_options: the
  samplesheet entry's own ``commandline_options.nextflow.skip_tools``,
  else the realm config's default. The resolved value travels in the plan's
  step scenario, so it takes part in fingerprints.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Accepted spelling of a pipeline tool name in skip_tools.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]+")

_SETTINGS_KEYS = frozenset(
    {
        "pipeline",
        "profile",
        "common_config",
        "work_base",
        "outdir_base",
        "extra_args",
        "skip_tools",
    }
)
_RUN_OPTION_KEYS = frozenset({"skip_tools"})

#: Tool sections accepted in a samplesheet entry's commandline_options.
_CMDLINE_TOOLS = frozenset({"nextflow"})
#: Keys accepted in the nextflow section of commandline_options.
_CMDLINE_NXF_KEYS = frozenset({"skip_tools"})


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
    #: requirements). None means none.
    common_config: Path | None
    #: Base directory for per-branch Nextflow work directories.
    work_base: Path
    #: Base directory for per-branch published results (--outdir).
    outdir_base: Path
    #: Extra nextflow arguments appended verbatim (last, so they win).
    extra_args: tuple[str, ...]
    #: Default tools to skip for a branch whose samplesheet entry does not
    #: specify skip_tools. Empty means skip nothing (see
    #: NextflowRunOptions and resolve_run_options).
    skip_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class NextflowRunOptions:
    """The effective per-branch Nextflow options.

    One instance per lane/samplesheet branch; resolved by
    resolve_run_options from the samplesheet entry, with the realm config
    as the default.
    """

    #: Tools to skip, e.g. ("samshee",) while its samplesheet validation is
    #: out of date. Empty means none.
    skip_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form carried in the step scenario."""
        return {"skip_tools": list(self.skip_tools)}


def as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object, got {type(value).__name__}.")
    return value


def reject_unknown(
    raw: Mapping[str, Any], allowed: frozenset[str], label: str
) -> None:
    """Reject unknown keys so a misspelt option cannot be silently ignored."""
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"{label}: unknown key(s) {unknown}; allowed: {sorted(allowed)}."
        )


def _text(raw: Mapping[str, Any], key: str, default: str, label: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{label} must be a list of strings.")
    return tuple(value)


def parse_skip_tools(value: object, label: str) -> tuple[str, ...]:
    """Validate a skip_tools value and return it as a tuple of tool names.

    Args:
        value: The raw value; must be a list of strings.
        label: Where the value came from, for error messages.

    Raises:
        ValueError: If value is not a list of strings or a name is invalid.
    """
    tools = _string_list(value, label)
    for tool in tools:
        if not _TOOL_NAME.fullmatch(tool):
            raise ValueError(f"{label}: invalid tool name {tool!r}.")
    return tools


def settings_from_mapping(raw: Mapping[str, Any] | None = None) -> NextflowSettings:
    """Build NextflowSettings from the ``nextflow`` object of the realm config.

    Every key is optional. Defaults target the local compose/DinD development
    stack: nextflow resolves the pipeline online (cached under NXF_HOME) and
    work and results live on the shared /work volume.

    Args:
        raw: The config's ``nextflow`` object, or None for all defaults.

    Returns:
        The resolved settings.

    Raises:
        ValueError: On an unknown key or a value of the wrong type.
    """
    label = "nextflow"
    raw = as_mapping(raw, label)
    reject_unknown(raw, _SETTINGS_KEYS, label)

    common_config = raw.get("common_config")
    return NextflowSettings(
        pipeline=_text(raw, "pipeline", "nf-core/demultiplex", label),
        profile=_text(raw, "profile", "docker", label),
        common_config=(
            None
            if common_config is None
            else Path(_text(raw, "common_config", "", label))
        ),
        work_base=Path(_text(raw, "work_base", "/work/nf-realm-work", label)),
        outdir_base=Path(_text(raw, "outdir_base", "/work/nf-realm-results", label)),
        extra_args=_string_list(raw.get("extra_args", []), f"{label}.extra_args"),
        skip_tools=parse_skip_tools(raw.get("skip_tools", []), f"{label}.skip_tools"),
    )


def parse_run_options(raw: object) -> NextflowRunOptions:
    """Validate a scenario's ``nextflow`` object into run options.

    Args:
        raw: The scenario's ``nextflow`` object (NextflowRunOptions.to_dict
            form), or None when it has none.

    Returns:
        The options; all defaults when raw is None.

    Raises:
        ValueError: On an unknown key, a non-list skip_tools, or a tool name
            that is not a plain identifier-like word.
    """
    label = "nextflow"
    options = as_mapping(raw, label)
    reject_unknown(options, _RUN_OPTION_KEYS, label)
    return NextflowRunOptions(
        skip_tools=parse_skip_tools(options.get("skip_tools", []), f"{label}.skip_tools")
    )


def resolve_run_options(
    samplesheet_entry: object,
    settings: NextflowSettings,
) -> NextflowRunOptions:
    """Resolve the effective run options of one samplesheet branch.

    The samplesheet entry's own options win; they are nested under
    ``commandline_options.nextflow``, e.g.
    ``{"commandline_options": {"nextflow": {"skip_tools": ["samshee"]}}}``.
    Unknown tool sections or keys are rejected so a misspelling cannot be
    silently ignored. When the entry names no skip_tools, the realm
    config's default (``settings.skip_tools``) applies; an explicit empty
    list means "skip nothing".

    Args:
        samplesheet_entry: One demux_sample_info samplesheets entry.
        settings: The realm's NextflowSettings (provides the default).

    Returns:
        The branch's run options.

    Raises:
        ValueError: On a malformed commandline_options object, an unknown
            key, a non-list skip_tools, or an invalid tool name.
    """
    entry = as_mapping(samplesheet_entry, "samplesheet entry")
    cmdline = entry.get("commandline_options")
    if cmdline is not None:
        cmdline = as_mapping(cmdline, "commandline_options")
        reject_unknown(cmdline, _CMDLINE_TOOLS, "commandline_options")
        nxf = cmdline.get("nextflow")
        if nxf is not None:
            nxf = as_mapping(nxf, "commandline_options.nextflow")
            reject_unknown(nxf, _CMDLINE_NXF_KEYS, "commandline_options.nextflow")
            if "skip_tools" in nxf:
                return NextflowRunOptions(
                    skip_tools=parse_skip_tools(
                        nxf["skip_tools"],
                        "commandline_options.nextflow.skip_tools",
                    )
                )
    return NextflowRunOptions(skip_tools=settings.skip_tools)


def branch_run_tag(
    canonical_flowcell_id: str, lane_id: str, settings_index: str
) -> str:
    """Return the per-branch path tag, e.g. SC2177698-SC3/lane_2_settings_0.

    The tag mirrors the plan's branch namespace (lane_<n>_settings_<i>) so a
    branch's directories are always unique within a flowcell and stable
    across re-runs of the same branch.
    """
    return f"{canonical_flowcell_id}/lane_{lane_id}_settings_{settings_index}"


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
    run_options: NextflowRunOptions,
) -> list[str]:
    """Build the argv of one per-branch Nextflow run.

    Per-run values come from the branch (flowcell, lane, the rendered
    samplesheet and the runfolder path) and from run_options (taken from the
    demux_sample_info document); per-branch work and result
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
    if run_options.skip_tools:
        cmd += ["--skip_tools", ",".join(run_options.skip_tools)]
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
        # Deliberately no -name: Nextflow indexes run names per user home
        # (HOME/.nextflow), so a stable name would collide every time a
        # branch re-runs (e.g. after a failure). -resume is keyed on the
        # work dir, not the run name.
        "-resume",
    ]
    cmd += list(settings.extra_args)
    return cmd


def run_nextflow(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    """Run a Nextflow command, streaming its combined output to log_path.

    The log is written line-by-line while the run is in progress (so it can
    be tailed live), and the combined output is also returned in
    CompletedProcess.stdout for the caller's error reporting.

    Args:
        cmd: The argv from build_nextflow_command.
        log_path: Where to write the combined stdout/stderr (created).

    Returns:
        The CompletedProcess (caller inspects returncode/stdout).
    """
    logger.info("Starting Nextflow run: %s", " ".join(shlex.quote(part) for part in cmd))
    chunks: list[str] = []
    with open(log_path, "w") as log:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line)
                log.write(line)
                log.flush()
                logger.debug("nextflow: %s", line.rstrip("\n"))
            returncode = proc.wait()
    return subprocess.CompletedProcess(cmd, returncode, stdout="".join(chunks))
