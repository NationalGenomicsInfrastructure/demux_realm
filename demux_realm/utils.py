import io
import xml.etree.ElementTree as ET
from pathlib import Path

# Required fields on every BCLConvert_Data row.
_BCL_DATA_REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "Lane",
    "Sample_ID",
    "Sample_Name",
    "index",
    "Sample_Project",
)

# Optional fields whose presence is checked for consistency across rows.
_BCL_DATA_CONSISTENCY_CHECKED_FIELDS: tuple[str, ...] = ("index2", "OverrideCycles")


def resolve_settings_index(entries: list[dict]) -> list[tuple[str, dict]]:
    """Resolve (settings_idx_str, entry) pairs for samplesheet entries sharing a lane.

    Rules:
      1. Single entry with settings_index    → [(str(settings_index), entry)]
      2. Single entry without settings_index → [("0", entry)]
      3. Multiple entries, ANY missing settings_index → raise ValueError (ambiguous)
      4. Multiple entries, ALL have settings_index   → [(str(si), entry), ...]

    Raises:
        ValueError: if multiple entries are present and any lack settings_index.
    """
    if len(entries) == 1:
        raw = entries[0].get("settings_index")
        return [(str(raw) if raw is not None else "0", entries[0])]

    missing_count = sum(1 for e in entries if e.get("settings_index") is None)
    if missing_count:
        raise ValueError(
            f"{missing_count} of {len(entries)} entries lack settings_index "
            f"(ambiguous; all entries for this lane are skipped)."
        )

    return [(str(e["settings_index"]), e) for e in entries]


def normalize_flowcell_id(fcid: str) -> str:
    """Canonical matching of SC... vs ASC..."""
    fcid = fcid or ""
    if fcid.startswith("ASC"):
        return fcid[1:]
    return fcid


def validate_lane_payload(lane_payload: dict) -> None:
    """Validate a single lane samplesheet payload before rendering.

    Checks:
    - Top-level required keys are present.
    - BCLConvert_Data is a non-empty list of dicts.
    - Every row contains the required row fields.
    - Optional fields (index2, OverrideCycles) are consistent: present in all
      rows or absent from all rows.

    Raises:
        ValueError: on any structural or content problem.
    """
    for key in ("Header", "raw_samplesheet_settings", "BCLConvert_Data"):
        if key not in lane_payload:
            raise ValueError(f"Lane payload missing required key '{key}'.")

    data = lane_payload["BCLConvert_Data"]
    if not isinstance(data, list):
        raise ValueError(f"BCLConvert_Data must be a list, got {type(data).__name__}.")
    if not data:
        raise ValueError("BCLConvert_Data is empty.")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"BCLConvert_Data row {i} is not a dict.")
        missing = [f for f in _BCL_DATA_REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"BCLConvert_Data row {i} missing required field(s): {missing}."
            )

    # Consistency check: optional fields must be uniformly present or absent.
    for field in _BCL_DATA_CONSISTENCY_CHECKED_FIELDS:
        presence = [field in row for row in data]
        if any(presence) and not all(presence):
            raise ValueError(
                f"BCLConvert_Data rows are inconsistent: '{field}' is present "
                f"in some rows but not all."
            )


def render_bcl_convert_samplesheet(lane_payload: dict) -> str:
    """Render a bcl-convert SampleSheet.csv text from a lane payload dict.

    Sections emitted (in order):
      [Header]          — key/value pairs
      [BCLConvert_Settings] — key/value pairs (source field: raw_samplesheet_settings)
      [BCLConvert_Data] — CSV table

    Column order in [BCLConvert_Data]: required fields first (in canonical
    order), then any extra fields found in the rows, in their natural dict
    iteration order from the first row.

    Args:
        lane_payload: A validated lane payload dict.  Call validate_lane_payload
            first to get a clear error on bad input.

    Returns:
        The full samplesheet text, ready to be written to SampleSheet.csv.
    """
    buf = io.StringIO()

    def _write_kv_section(section_name: str, mapping: dict) -> None:
        buf.write(f"[{section_name}]\n")
        for k, v in mapping.items():
            buf.write(f"{k},{v}\n")

    _write_kv_section("Header", lane_payload["Header"])
    buf.write("\n")

    # The field is currently named raw_samplesheet_settings; it will likely
    # be renamed to BCLConvert_Settings upstream.
    _write_kv_section("BCLConvert_Settings", lane_payload["raw_samplesheet_settings"])
    buf.write("\n")

    data_rows = lane_payload["BCLConvert_Data"]
    # Build column order: required columns first, then extra columns in
    # first-row dict order (stable across Python 3.7+).
    extra_cols = [k for k in data_rows[0] if k not in _BCL_DATA_REQUIRED_ROW_FIELDS]
    columns = list(_BCL_DATA_REQUIRED_ROW_FIELDS) + extra_cols

    buf.write("[BCLConvert_Data]\n")
    buf.write(",".join(columns) + "\n")
    for row in data_rows:
        buf.write(",".join(str(row.get(col, "")) for col in columns) + "\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# x_flowcells helpers
# ---------------------------------------------------------------------------

# Fields from BCLConvert_Data rows projected into samplesheet_csv.
# Sample_Ref is included as a passthrough: added to each row when present in the
# source BCLConvert_Data, omitted silently when absent.
_SAMPLESHEET_CSV_FIELDS: frozenset[str] = frozenset(
    (
        "Lane",
        "Sample_ID",
        "Sample_Name",
        "index",
        "index2",
        "Sample_Project",
        "Sample_Ref",
    )
)

# Fields sourced from demux_sample_info.uploaded_lims_info (lowercase keys) mapped
# to their samplesheet_csv output names (Title_Case keys).
_LIMS_CSV_FIELDS: dict[str, str] = {
    "sample_ref": "Sample_Ref",
    "description": "Description",
    "control": "Control",
    "recipe": "Recipe",
    "operator": "Operator",
}


def _xml_element_to_value(elem: ET.Element) -> "str | dict":
    """Recursively convert an XML element to a str (leaf) or dict (node).

    Rules:
    - No children, no attributes → stripped text content (or "" if empty).
    - No children, has attributes → attribute dict; non-empty text added under "_text".
    - Has children → dict keyed by child tag; repeated sibling tags become a list.
    """
    children = list(elem)
    if not children:
        if elem.attrib:
            attrs = dict(elem.attrib)
            text = (elem.text or "").strip()
            if text:
                attrs["_text"] = text
            return attrs
        return (elem.text or "").strip()

    result: dict = {}
    for child in children:
        val = _xml_element_to_value(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(val)
        else:
            result[child.tag] = val
    return result


def _normalize_run_date(raw: str) -> str:
    """Normalize a RunInfo Date value to YYMMDD.

    Handles both ISO format ('2026-03-12T14:25:57Z' → '260312') and
    already-short YYMMDD values ('260312' → '260312').
    """
    date_part = raw.split("T")[0].replace("-", "")  # "20260312" or "260312"
    if len(date_part) == 8:  # YYYYMMDD → strip century
        return date_part[2:]
    return date_part


def parse_run_info_xml(xml_path: Path) -> dict:
    """Parse RunInfo.xml and return a TACA-compatible RunInfo dict.

    Extracts: Id, Number, Flowcell, Instrument, Date (YYMMDD),
    Reads (list of attribute dicts), FlowcellLayout (attribute dict).

    Raises:
        ValueError: if the file is missing or cannot be parsed.
    """
    if not xml_path.exists():
        raise ValueError(f"RunInfo.xml not found: {xml_path}")
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse RunInfo.xml: {exc}") from exc

    root = tree.getroot()
    run_elem = root.find("Run")
    if run_elem is None:
        raise ValueError(f"No <Run> element in {xml_path}")

    result: dict = {
        "Id": run_elem.get("Id", ""),
        "Number": run_elem.get("Number", ""),
    }

    for tag in ("Flowcell", "Instrument"):
        elem = run_elem.find(tag)
        result[tag] = elem.text.strip() if (elem is not None and elem.text) else ""

    date_elem = run_elem.find("Date")
    raw_date = (date_elem.text or "").strip() if date_elem is not None else ""
    result["Date"] = _normalize_run_date(raw_date) if raw_date else ""

    reads_elem = run_elem.find("Reads")
    result["Reads"] = (
        [dict(r.attrib) for r in reads_elem.findall("Read")]
        if reads_elem is not None
        else []
    )

    layout_elem = run_elem.find("FlowcellLayout")
    result["FlowcellLayout"] = (
        dict(layout_elem.attrib) if layout_elem is not None else {}
    )

    return result


def parse_run_parameters_xml(xml_path: Path) -> dict:
    """Parse RunParameters.xml and return a full RunParameters dict.

    All direct children of the root element are included. Nested elements are
    recursively converted: leaf elements become strings, parent elements become
    dicts, and repeated sibling tags become lists.

    Raises:
        ValueError: if the file is missing or cannot be parsed.
    """
    if not xml_path.exists():
        raise ValueError(f"RunParameters.xml not found: {xml_path}")
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse RunParameters.xml: {exc}") from exc

    root = tree.getroot()
    result: dict = {}
    for child in root:
        val = _xml_element_to_value(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(val)
        else:
            result[child.tag] = val
    return result


def derive_xflowcell_name(runfolder_id: str) -> str:
    """Return the TACA-compatible x_flowcells name for a runfolder.

    Format: '{date}_{flowcell_id}' — first and last '_'-separated segments.
    E.g. '20260312_SH01140_0005_ASC2177698-SC3' → '20260312_ASC2177698-SC3'.

    Raises:
        ValueError: if runfolder_id has fewer than 2 '_'-separated parts.
    """
    parts = runfolder_id.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot derive x_flowcells name from '{runfolder_id}': "
            f"expected at least 2 '_'-separated parts."
        )
    return f"{parts[0]}_{parts[-1]}"


def build_lims_lookup(
    uploaded_lims_info: list[dict],
) -> dict[tuple[str, str], dict]:
    """Build a (lane, sample_name) → lims_entry lookup from uploaded_lims_info.

    Both lane and sample_name are coerced to str so callers need not worry about
    int vs. string lane values.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for entry in uploaded_lims_info:
        key = (str(entry.get("lane", "")), str(entry.get("sample_name", "")))
        lookup[key] = entry
    return lookup


def flatten_samplesheets(
    samplesheets: list[dict],
    flowcell_id: str = "",
    lims_lookup: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Flatten demux_sample_info['samplesheets'] into an x_flowcells samplesheet_csv list.

    For each entry, iterates BCLConvert_Data rows and emits a dict with:
    - FCID prepended (when flowcell_id is provided)
    - Fields from _SAMPLESHEET_CSV_FIELDS that are present in the row
      (Lane, Sample_ID, Sample_Name, index, index2, Sample_Project, Sample_Ref)
    - Fields from _LIMS_CSV_FIELDS enriched from lims_lookup when a match is found
      (Sample_Ref, Description, Control, Recipe, Operator)
    """
    rows: list[dict] = []
    for entry in samplesheets:
        for row in entry.get("BCLConvert_Data", []):
            flat_row: dict = {}
            if flowcell_id:
                flat_row["FCID"] = flowcell_id
            flat_row.update(
                {k: v for k, v in row.items() if k in _SAMPLESHEET_CSV_FIELDS}
            )
            if lims_lookup is not None:
                key = (str(row.get("Lane", "")), str(row.get("Sample_Name", "")))
                lims_entry = lims_lookup.get(key)
                if lims_entry:
                    for lims_key, csv_key in _LIMS_CSV_FIELDS.items():
                        if lims_key in lims_entry:
                            flat_row[csv_key] = lims_entry[lims_key]
            rows.append(flat_row)
    return rows


def build_x_flowcell_payload(
    name: str,
    run_info: dict,
    run_params: dict,
    samplesheet_csv: list[dict],
) -> dict:
    """Assemble the pre-demux x_flowcells document payload (no _id/_rev).

    Fields: name, RunInfo, RunParameters, samplesheet_csv.
    Does not include run_setup or any post-demux fields.
    """
    return {
        "name": name,
        "RunInfo": run_info,
        "RunParameters": run_params,
        "samplesheet_csv": samplesheet_csv,
    }
