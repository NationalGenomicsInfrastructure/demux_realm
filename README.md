# demux_realm

`demux_realm` contains the active Yggdrasil realm for demultiplexing planning and
pre-demux metadata preparation.

The realm currently:

* registers a `ygg.realm` entry point named `dmx_realm`;
* watches `demux_sample_info` and `flowcell_status` CouchDB changes;
* builds Yggdrasil plan drafts for flowcell-level initialization and lane-level
  demultiplexing work;
* prepares the pre-demux `x_flowcells` document from runfolder XML files,
  sample-sheet payloads, and uploaded LIMS metadata.

## Setup

### Create and activate a Python 3.11 environment:

```bash
conda create -n ygg python=3.11
conda activate ygg
```

### Install Yggdrasil and this repository in editable mode:

```bash
git clone https://github.com/NationalGenomicsInfrastructure/Yggdrasil.git
pip install -e Yggdrasil

git clone https://github.com/NationalGenomicsInfrastructure/demux_realm.git
pip install -e demux_realm
```

### Install local development dependencies:

```bash
pip install -e ".[dev]"
```

Yggdrasil is required to run the realm. It is kept out of the base install so
the package can still be installed without fetching a Git dependency. When
network and GitHub access are available, install Yggdrasil through the `ygg`
extra:

```bash
pip install -e ".[ygg,dev]"
```

## Yggdrasil Entry Point

On startup, Yggdrasil discovers the realm through `pyproject.toml`:

```toml
[project.entry-points."ygg.realm"]
dmx_realm = "demux_realm.descriptor:get_realm_descriptor"
```

The descriptor returns the `dmx_realm` realm with:

* `DemuxHandler`
* CouchDB watch specs for `demux_sample_info_db`
* CouchDB watch specs for `flowcell_status_db`

## Repository Layout

```text
demux_realm/
├── demux_realm/
│   ├── __init__.py
│   ├── descriptor.py
│   ├── handler.py
│   ├── recipes.py
│   ├── steps.py
│   └── utils.py
├── tests/
├── README.md
├── pyproject.toml
└── LICENSE
```

## Development

Run the test suite:

```bash
pytest
```

Check that the package imports from the repository root:

```bash
python -c "import demux_realm; from demux_realm.descriptor import get_realm_descriptor"
```
