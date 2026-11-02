"""Deployment configuration of the realm, read from a JSON file.

The file is resolved by Yggdrasil's ConfigLoader, so it lives beside main.json
in the workspace's ``common/configurations`` directory and, in dev mode, a
``dev_dmx_realm.json`` next to it takes precedence. Credentials do not belong
here: they stay in Yggdrasil's own configuration and environment.

Example::

    {
      "hpc_base_path": "/work",
      "nextflow": {
        "pipeline": "nf-core/demultiplex",
        "profile": "docker",
        "common_config": "/work/demultiplex_local.config",
        "work_base": "/work/nf-realm-work",
        "outdir_base": "/work/nf-realm-results",
        "extra_args": []
      }
    }

Every key is optional; the defaults target the local compose/DinD stack.
``nextflow.skip_tools`` is the default for a branch whose samplesheet entry
does not specify one (see resolve_run_options).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from yggdrasil.config import ConfigLoader

from .nextflow import (
    NextflowSettings,
    as_mapping,
    reject_unknown,
    settings_from_mapping,
)

CONFIG_FILENAME = "dmx_realm.json"

_CONFIG_KEYS = frozenset({"hpc_base_path", "nextflow"})


@dataclass(frozen=True)
class RealmConfig:
    """Deployment-level settings of the realm."""

    #: Prefix of the runfolder path: the HPC mount point as seen by Yggdrasil.
    #: Empty means flowcell_status destination paths are used as they are.
    hpc_base_path: str
    nextflow: NextflowSettings


def realm_config_from_mapping(raw: Mapping[str, Any] | None = None) -> RealmConfig:
    """Validate a parsed config file and apply defaults.

    Raises:
        ValueError: On an unknown key or a value of the wrong type.
    """
    raw = as_mapping(raw, "config")
    reject_unknown(raw, _CONFIG_KEYS, "config")
    hpc_base_path = raw.get("hpc_base_path", "")
    if not isinstance(hpc_base_path, str):
        raise ValueError("config.hpc_base_path must be a string.")
    return RealmConfig(
        hpc_base_path=hpc_base_path.strip(),
        nextflow=settings_from_mapping(raw.get("nextflow")),
    )


def load_realm_config() -> RealmConfig:
    """Load the realm's JSON config, falling back to defaults if it is absent.

    Raises:
        ValueError: If the file exists but is invalid.
    """
    return realm_config_from_mapping(ConfigLoader().load_config(CONFIG_FILENAME))
