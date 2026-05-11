import textwrap

import pytest

from dataflow_demux.yggdrasil_realm.utils import (
    build_lims_lookup,
    build_x_flowcell_payload,
    derive_xflowcell_name,
    flatten_samplesheets,
    parse_run_info_xml,
    parse_run_parameters_xml,
)

# ---------------------------------------------------------------------------
# Minimal XML fixtures
# ---------------------------------------------------------------------------

_RUN_INFO_ISO_DATE = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunInfo Version="7">
      <Run Id="20260312_SH01140_0005_ASC2177698-SC3" Number="5">
        <Flowcell>BXA66715-2010</Flowcell>
        <Instrument>SH01140</Instrument>
        <Date>2026-03-12T14:25:57Z</Date>
        <Reads>
          <Read Number="1" NumCycles="301" IsIndexedRead="N" IsReverseComplement="N"/>
          <Read Number="2" NumCycles="10" IsIndexedRead="Y" IsReverseComplement="N"/>
        </Reads>
        <FlowcellLayout LaneCount="1" SurfaceCount="1" SwathCount="9" TileCount="2"/>
      </Run>
    </RunInfo>
""")

_RUN_INFO_SHORT_DATE = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunInfo Version="2">
      <Run Id="260312_A00000_0001_ASC123" Number="1">
        <Flowcell>ASC123</Flowcell>
        <Instrument>A00000</Instrument>
        <Date>260312</Date>
        <Reads>
          <Read Number="1" NumCycles="151" IsIndexedRead="N"/>
        </Reads>
        <FlowcellLayout LaneCount="4" SurfaceCount="2" SwathCount="1" TileCount="0"/>
      </Run>
    </RunInfo>
""")

_RUN_PARAMS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunParameters>
      <InstrumentType>MiSeqi100Plus</InstrumentType>
      <InstrumentSerialNumber>SH01140</InstrumentSerialNumber>
      <RunId>20260312_SH01140_0005_ASC2177698-SC3</RunId>
      <RunCounter>5</RunCounter>
    </RunParameters>
""")

_RUN_PARAMS_FULL = textwrap.dedent("""\
    <?xml version="1.0"?>
    <RunParameters>
      <Application>MiSeqi100Series Control Software</Application>
      <SystemSuiteVersion>1.1.0.26158</SystemSuiteVersion>
      <OutputFolder>/mnt/usb/run</OutputFolder>
      <CustomPrimerSelections>
        <ReadOnePrimer>false</ReadOnePrimer>
        <ReadTwoPrimer>false</ReadTwoPrimer>
        <IndexOnePrimer>false</IndexOnePrimer>
        <IndexTwoPrimer>false</IndexTwoPrimer>
      </CustomPrimerSelections>
      <InstrumentType>MiSeqi100Plus</InstrumentType>
      <InstrumentSerialNumber>SH01140</InstrumentSerialNumber>
      <RunId>20260312_SH01140_0005_ASC2177698-SC3</RunId>
      <ConsumableInfo>
        <ConsumableInfo>
          <SerialNumber>SC2177698-SC3</SerialNumber>
          <Type>DryCartridge</Type>
        </ConsumableInfo>
        <ConsumableInfo>
          <SerialNumber>BXA66715-2010</SerialNumber>
          <Type>FlowCell_1</Type>
        </ConsumableInfo>
      </ConsumableInfo>
      <PlannedReads>
        <Read ReadName="Read1" Cycles="301" />
        <Read ReadName="Index1" Cycles="10" />
      </PlannedReads>
      <SecondaryAnalysisInfo />
      <RunCounter>5</RunCounter>
      <RecipeName>5M/600_B_Recipe</RecipeName>
    </RunParameters>
""")


@pytest.fixture
def run_info_iso(tmp_path):
    p = tmp_path / "RunInfo.xml"
    p.write_text(_RUN_INFO_ISO_DATE)
    return p


@pytest.fixture
def run_params_full(tmp_path):
    p = tmp_path / "RunParameters.xml"
    p.write_text(_RUN_PARAMS_FULL)
    return p


@pytest.fixture
def run_info_short(tmp_path):
    p = tmp_path / "RunInfo.xml"
    p.write_text(_RUN_INFO_SHORT_DATE)
    return p


@pytest.fixture
def run_params(tmp_path):
    p = tmp_path / "RunParameters.xml"
    p.write_text(_RUN_PARAMS)
    return p


# ---------------------------------------------------------------------------
# parse_run_info_xml
# ---------------------------------------------------------------------------


def test_parse_run_info_xml_fields(run_info_iso):
    result = parse_run_info_xml(run_info_iso)
    assert result["Id"] == "20260312_SH01140_0005_ASC2177698-SC3"
    assert result["Number"] == "5"
    assert result["Flowcell"] == "BXA66715-2010"
    assert result["Instrument"] == "SH01140"
    assert isinstance(result["Reads"], list)
    assert len(result["Reads"]) == 2
    assert result["Reads"][0]["Number"] == "1"
    assert result["Reads"][0]["NumCycles"] == "301"
    assert isinstance(result["FlowcellLayout"], dict)
    assert result["FlowcellLayout"]["LaneCount"] == "1"


def test_parse_run_info_xml_iso_date_normalized(run_info_iso):
    assert parse_run_info_xml(run_info_iso)["Date"] == "260312"


def test_parse_run_info_xml_short_date_passthrough(run_info_short):
    assert parse_run_info_xml(run_info_short)["Date"] == "260312"


def test_parse_run_info_xml_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        parse_run_info_xml(tmp_path / "RunInfo.xml")


def test_parse_run_info_xml_malformed(tmp_path):
    p = tmp_path / "RunInfo.xml"
    p.write_text("<unclosed")
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_run_info_xml(p)


# ---------------------------------------------------------------------------
# parse_run_parameters_xml
# ---------------------------------------------------------------------------


def test_parse_run_parameters_xml_fields(run_params):
    result = parse_run_parameters_xml(run_params)
    assert result["InstrumentType"] == "MiSeqi100Plus"
    assert result["InstrumentSerialNumber"] == "SH01140"
    assert result["RunId"] == "20260312_SH01140_0005_ASC2177698-SC3"
    assert result["RunCounter"] == "5"


def test_parse_run_parameters_xml_missing_fields_omitted(tmp_path):
    p = tmp_path / "RunParameters.xml"
    p.write_text("<RunParameters><RunId>X</RunId></RunParameters>")
    result = parse_run_parameters_xml(p)
    assert result == {"RunId": "X"}
    assert "InstrumentType" not in result


def test_parse_run_parameters_xml_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        parse_run_parameters_xml(tmp_path / "RunParameters.xml")


def test_parse_run_parameters_xml_malformed(tmp_path):
    p = tmp_path / "RunParameters.xml"
    p.write_text("<unclosed")
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_run_parameters_xml(p)


# ---------------------------------------------------------------------------
# derive_xflowcell_name
# ---------------------------------------------------------------------------


def test_derive_xflowcell_name_four_parts():
    assert (
        derive_xflowcell_name("20260312_SH01140_0005_ASC2177698-SC3")
        == "20260312_ASC2177698-SC3"
    )


def test_derive_xflowcell_name_two_parts():
    assert derive_xflowcell_name("20260312_ASC2177698-SC3") == "20260312_ASC2177698-SC3"


def test_derive_xflowcell_name_single_part_raises():
    with pytest.raises(ValueError):
        derive_xflowcell_name("singlepart")


# ---------------------------------------------------------------------------
# flatten_samplesheets
# ---------------------------------------------------------------------------


def test_flatten_samplesheets_basic():
    samplesheets = [
        {
            "lane": 1,
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "Sample1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                },
                {
                    "Lane": "1",
                    "Sample_ID": "S2",
                    "Sample_Name": "Sample2",
                    "index": "TTTT",
                    "Sample_Project": "P001",
                },
            ],
        }
    ]
    rows = flatten_samplesheets(samplesheets)
    assert len(rows) == 2
    assert rows[0]["Sample_ID"] == "S1"
    assert rows[1]["Sample_ID"] == "S2"
    assert "Lane" in rows[0]
    assert "Sample_Project" in rows[0]


def test_flatten_samplesheets_index2_included_when_present():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "index2": "TTTT",
                    "Sample_Project": "P001",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets)
    assert rows[0]["index2"] == "TTTT"


def test_flatten_samplesheets_extra_fields_stripped():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                    "OverrideCycles": "Y151;I10;I10;Y151",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets)
    assert "OverrideCycles" not in rows[0]


def test_flatten_samplesheets_multi_lane():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "A",
                    "Sample_Project": "P",
                }
            ]
        },
        {
            "BCLConvert_Data": [
                {
                    "Lane": "2",
                    "Sample_ID": "S2",
                    "Sample_Name": "S2",
                    "index": "T",
                    "Sample_Project": "P",
                }
            ]
        },
    ]
    rows = flatten_samplesheets(samplesheets)
    assert len(rows) == 2
    assert rows[0]["Lane"] == "1"
    assert rows[1]["Lane"] == "2"


# ---------------------------------------------------------------------------
# build_x_flowcell_payload
# ---------------------------------------------------------------------------


def test_build_x_flowcell_payload_fields():
    payload = build_x_flowcell_payload(
        name="20260312_ASC123",
        run_info={"Id": "x"},
        run_params={"InstrumentType": "MiSeqi100"},
        samplesheet_csv=[{"Lane": "1", "Sample_ID": "S1"}],
    )
    assert payload["name"] == "20260312_ASC123"
    assert payload["RunInfo"] == {"Id": "x"}
    assert payload["RunParameters"] == {"InstrumentType": "MiSeqi100"}
    assert len(payload["samplesheet_csv"]) == 1


def test_build_x_flowcell_payload_no_id_rev_or_run_setup():
    payload = build_x_flowcell_payload("n", {}, {}, [])
    assert "_id" not in payload
    assert "_rev" not in payload
    assert "run_setup" not in payload


# ---------------------------------------------------------------------------
# parse_run_parameters_xml — nested / full parsing
# ---------------------------------------------------------------------------


def test_parse_run_parameters_xml_nested_custom_primer_selections(run_params_full):
    result = parse_run_parameters_xml(run_params_full)
    assert isinstance(result["CustomPrimerSelections"], dict)
    assert result["CustomPrimerSelections"]["ReadOnePrimer"] == "false"
    assert result["CustomPrimerSelections"]["IndexTwoPrimer"] == "false"


def test_parse_run_parameters_xml_nested_consumable_info(run_params_full):
    result = parse_run_parameters_xml(run_params_full)
    ci = result["ConsumableInfo"]["ConsumableInfo"]
    assert isinstance(ci, list)
    assert len(ci) == 2
    serials = [item["SerialNumber"] for item in ci]
    assert "SC2177698-SC3" in serials
    assert "BXA66715-2010" in serials


def test_parse_run_parameters_xml_nested_planned_reads(run_params_full):
    result = parse_run_parameters_xml(run_params_full)
    reads = result["PlannedReads"]["Read"]
    assert isinstance(reads, list)
    assert reads[0] == {"ReadName": "Read1", "Cycles": "301"}
    assert reads[1] == {"ReadName": "Index1", "Cycles": "10"}


def test_parse_run_parameters_xml_empty_element(run_params_full):
    result = parse_run_parameters_xml(run_params_full)
    assert result["SecondaryAnalysisInfo"] == ""


def test_parse_run_parameters_xml_simple_fields(run_params_full):
    result = parse_run_parameters_xml(run_params_full)
    assert result["Application"] == "MiSeqi100Series Control Software"
    assert result["SystemSuiteVersion"] == "1.1.0.26158"
    assert result["OutputFolder"] == "/mnt/usb/run"
    assert result["RecipeName"] == "5M/600_B_Recipe"


# ---------------------------------------------------------------------------
# flatten_samplesheets — FCID and Sample_Ref
# ---------------------------------------------------------------------------


def test_flatten_samplesheets_adds_fcid():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets, flowcell_id="SC2177698-SC3")
    assert rows[0]["FCID"] == "SC2177698-SC3"
    assert list(rows[0].keys())[0] == "FCID"  # FCID is first


def test_flatten_samplesheets_no_fcid_when_empty():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets)  # flowcell_id="" default
    assert "FCID" not in rows[0]


def test_flatten_samplesheets_includes_sample_ref_when_present():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                    "Sample_Ref": "Human (Homo sapiens GRCh38)",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets, flowcell_id="SC123")
    assert rows[0]["Sample_Ref"] == "Human (Homo sapiens GRCh38)"


def test_flatten_samplesheets_sample_ref_absent_when_not_in_row():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P001",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets, flowcell_id="SC123")
    assert "Sample_Ref" not in rows[0]


# ---------------------------------------------------------------------------
# build_lims_lookup
# ---------------------------------------------------------------------------

_LIMS_ENTRY = {
    "lane": "1",
    "sample_name": "P39402_1001",
    "sample_ref": "Human (GRCh38)",
    "description": "G__Production_26_03",
    "control": "N",
    "recipe": "301-301",
    "operator": "Shan_Chen",
}


def test_build_lims_lookup_basic():
    lookup = build_lims_lookup([_LIMS_ENTRY])
    assert ("1", "P39402_1001") in lookup
    assert lookup[("1", "P39402_1001")]["operator"] == "Shan_Chen"


def test_build_lims_lookup_empty():
    assert build_lims_lookup([]) == {}


def test_build_lims_lookup_int_lane_coerced():
    entry = {**_LIMS_ENTRY, "lane": 1}
    lookup = build_lims_lookup([entry])
    assert ("1", "P39402_1001") in lookup


def test_build_lims_lookup_multiple_samples():
    entries = [
        {**_LIMS_ENTRY, "sample_name": "S1"},
        {**_LIMS_ENTRY, "sample_name": "S2"},
    ]
    lookup = build_lims_lookup(entries)
    assert len(lookup) == 2
    assert ("1", "S1") in lookup
    assert ("1", "S2") in lookup


# ---------------------------------------------------------------------------
# flatten_samplesheets — lims enrichment
# ---------------------------------------------------------------------------

_LIMS_LOOKUP = build_lims_lookup([_LIMS_ENTRY])


def test_flatten_samplesheets_lims_fields_merged():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "Sample_P39402_1001",
                    "Sample_Name": "P39402_1001",
                    "index": "GAACAATTCC",
                    "Sample_Project": "G__Production_26_03",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(
        samplesheets, flowcell_id="SC123", lims_lookup=_LIMS_LOOKUP
    )
    row = rows[0]
    assert row["Sample_Ref"] == "Human (GRCh38)"
    assert row["Description"] == "G__Production_26_03"
    assert row["Control"] == "N"
    assert row["Recipe"] == "301-301"
    assert row["Operator"] == "Shan_Chen"


def test_flatten_samplesheets_lims_no_match_extra_fields_absent():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "Sample_X",
                    "Sample_Name": "UNKNOWN",
                    "index": "AAAA",
                    "Sample_Project": "P",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(
        samplesheets, flowcell_id="SC123", lims_lookup=_LIMS_LOOKUP
    )
    assert "Description" not in rows[0]
    assert "Operator" not in rows[0]


def test_flatten_samplesheets_lims_partial_entry():
    partial_entry = {"lane": "1", "sample_name": "S1", "description": "MyDesc"}
    lookup = build_lims_lookup([partial_entry])
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets, lims_lookup=lookup)
    assert rows[0]["Description"] == "MyDesc"
    assert "Sample_Ref" not in rows[0]
    assert "Control" not in rows[0]


def test_flatten_samplesheets_no_lims_lookup_unchanged():
    samplesheets = [
        {
            "BCLConvert_Data": [
                {
                    "Lane": "1",
                    "Sample_ID": "S1",
                    "Sample_Name": "S1",
                    "index": "AAAA",
                    "Sample_Project": "P",
                }
            ]
        }
    ]
    rows = flatten_samplesheets(samplesheets)
    assert "Description" not in rows[0]
    assert "Operator" not in rows[0]
