"""Tests for OME structure XML metadata file writing."""

import os
import re
from xml.dom.minidom import parseString

from tensorswitch_v2.utils.ome_structure import write_xml_metadata_file


# Sample XML that mimics the Zeiss CZI style: inter-tag whitespace already
# present, plus a <Comment> element whose text content contains newlines that
# must be preserved.
SAMPLE_XML_WITH_INTER_TAG_WHITESPACE = """<?xml version="1.0" encoding="UTF-8"?>
<ImageDocument>
  <Metadata>
    <Version>1.0</Version>
    <Information>
      <User>
        <DisplayName>zeiss</DisplayName>
      </User>
      <Document>
        <Name>sample</Name>
        <Comment>Light sheet thickness: 4.16 um
Illumination mode: single
Pivot scan: on</Comment>
        <UserName>zeiss</UserName>
      </Document>
    </Information>
  </Metadata>
</ImageDocument>
"""


def test_write_xml_metadata_file_no_doubled_whitespace(temp_dir):
    """Writing pre-formatted XML must not produce consecutive blank lines.

    Regression test for the minidom.toprettyxml() whitespace-doubling bug
    reported by Konrad on CZI→Zarr3 output.
    """
    write_xml_metadata_file(
        output_path=temp_dir,
        xml_string=SAMPLE_XML_WITH_INTER_TAG_WHITESPACE,
        source_format="czi",
        zarr_format=3,
        image_key="",
    )

    xml_path = os.path.join(temp_dir, "OME", "METADATA.czi.xml")
    assert os.path.exists(xml_path), "XML file was not written"

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # No consecutive blank lines or whitespace-only-then-blank patterns between tags
    assert "\n\n\n" not in content, "Output has 3+ consecutive newlines"
    blank_line_pairs = re.findall(r"\n[ \t]*\n[ \t]*\n", content)
    assert not blank_line_pairs, (
        f"Output has doubled whitespace lines: {blank_line_pairs[:3]}"
    )


def test_write_xml_metadata_file_preserves_comment_text(temp_dir):
    """Mixed-content elements like <Comment> must keep their text verbatim."""
    write_xml_metadata_file(
        output_path=temp_dir,
        xml_string=SAMPLE_XML_WITH_INTER_TAG_WHITESPACE,
        source_format="czi",
        zarr_format=3,
        image_key="",
    )

    xml_path = os.path.join(temp_dir, "OME", "METADATA.czi.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Original Comment text (with its newlines inside the element) must survive
    assert "Light sheet thickness: 4.16 um" in content
    assert "Illumination mode: single" in content
    assert "Pivot scan: on" in content


def test_write_xml_metadata_file_still_valid_xml(temp_dir):
    """Output must still be parseable as XML after whitespace stripping."""
    write_xml_metadata_file(
        output_path=temp_dir,
        xml_string=SAMPLE_XML_WITH_INTER_TAG_WHITESPACE,
        source_format="czi",
        zarr_format=3,
        image_key="",
    )

    xml_path = os.path.join(temp_dir, "OME", "METADATA.czi.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Should round-trip through minidom without error
    dom = parseString(content)
    root = dom.documentElement
    assert root.tagName == "ImageDocument"


def test_write_xml_metadata_file_ome_source(temp_dir):
    """Non-CZI sources should produce METADATA.ome.xml."""
    write_xml_metadata_file(
        output_path=temp_dir,
        xml_string=SAMPLE_XML_WITH_INTER_TAG_WHITESPACE,
        source_format="tiff",
        zarr_format=3,
        image_key="raw",
    )

    assert os.path.exists(os.path.join(temp_dir, "OME", "METADATA.ome.xml"))
    assert not os.path.exists(os.path.join(temp_dir, "OME", "METADATA.czi.xml"))


def test_write_xml_metadata_file_shrinks_line_count(temp_dir):
    """Verify the fix actually reduces line count on whitespace-heavy input."""
    # Build an XML with the same doubled-whitespace pattern Konrad saw
    inner = "\n    \n    \n    ".join(
        [f"<Field{i}>value{i}</Field{i}>" for i in range(20)]
    )
    bloated_xml = (
        f"<?xml version=\"1.0\"?>\n<Root>\n    \n    \n    {inner}\n  </Root>\n"
    )

    write_xml_metadata_file(
        output_path=temp_dir,
        xml_string=bloated_xml,
        source_format="czi",
        zarr_format=3,
        image_key="",
    )

    xml_path = os.path.join(temp_dir, "OME", "METADATA.czi.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 1 decl + 1 open + 20 fields + 1 close = 23 meaningful lines
    # Allow a small trailing slack but reject if we've ~doubled it
    assert len(lines) < 30, (
        f"Output has {len(lines)} lines — whitespace stripping did not work"
    )
