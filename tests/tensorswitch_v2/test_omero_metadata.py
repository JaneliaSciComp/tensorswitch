"""Tests for OMERO metadata builder and format-specific channel extractors."""

import pytest
from tensorswitch_v2.utils.omero import (
    ChannelInfo,
    build_omero_metadata,
    wavelength_to_color,
    resolve_channel_color,
    resolve_channel_label,
    get_dtype_window,
    extract_channels_from_ome_xml,
    extract_channels_from_czi_xml,
    extract_channel_info,
    _rgba_int_to_hex_rgb,
)


# ============================================================================
# wavelength_to_color
# ============================================================================

class TestWavelengthToColor:
    def test_blue(self):
        assert wavelength_to_color(405) == "0000FF"
        assert wavelength_to_color(488) == "0000FF"
        assert wavelength_to_color(499) == "0000FF"

    def test_green(self):
        assert wavelength_to_color(500) == "00FF00"
        assert wavelength_to_color(525) == "00FF00"
        assert wavelength_to_color(559) == "00FF00"

    def test_red(self):
        assert wavelength_to_color(560) == "FF0000"
        assert wavelength_to_color(590) == "FF0000"
        assert wavelength_to_color(647) == "FF0000"


# ============================================================================
# _rgba_int_to_hex_rgb
# ============================================================================

class TestRgbaIntToHexRgb:
    def test_pure_red(self):
        # RGBA packed: R=FF, G=00, B=00, A=FF → 0xFF0000FF
        assert _rgba_int_to_hex_rgb(0xFF0000FF) == "FF0000"

    def test_negative_int(self):
        # -16776961 = 0xFF0000FF as signed → red
        assert _rgba_int_to_hex_rgb(-16776961) == "FF0000"

    def test_blue(self):
        # RGBA: R=00, G=00, B=FF, A=FF → 0x0000FFFF
        assert _rgba_int_to_hex_rgb(0x0000FFFF) == "0000FF"

    def test_green(self):
        # RGBA: R=00, G=FF, B=00, A=FF → 0x00FF00FF
        assert _rgba_int_to_hex_rgb(0x00FF00FF) == "00FF00"

    def test_zero(self):
        assert _rgba_int_to_hex_rgb(0) == "000000"

    def test_bioformats_real_values(self):
        # Real values from Bio-Formats CZI conversion
        assert _rgba_int_to_hex_rgb(16711935) == "00FF00"    # green
        assert _rgba_int_to_hex_rgb(-16711681) == "FF00FF"   # magenta
        assert _rgba_int_to_hex_rgb(-16776961) == "FF0000"   # red
        assert _rgba_int_to_hex_rgb(16777215) == "00FFFF"    # cyan


# ============================================================================
# resolve_channel_color
# ============================================================================

class TestResolveChannelColor:
    def test_explicit_hex_color(self):
        ch = ChannelInfo(color="FF0000")
        assert resolve_channel_color(ch, 0, 3) == "FF0000"

    def test_explicit_int_color(self):
        # RGBA packed: R=00, G=FF, B=00, A=FF
        ch = ChannelInfo(color=0x00FF00FF)
        assert resolve_channel_color(ch, 0, 3) == "00FF00"

    def test_explicit_int_string_color(self):
        ch = ChannelInfo(color="-16776961")  # 0xFF0000FF signed → red
        assert resolve_channel_color(ch, 0, 3) == "FF0000"

    def test_single_channel_grey(self):
        ch = ChannelInfo()
        assert resolve_channel_color(ch, 0, 1) == "808080"

    def test_emission_wavelength(self):
        ch = ChannelInfo(emission_wavelength=488)
        assert resolve_channel_color(ch, 0, 3) == "0000FF"

    def test_excitation_wavelength(self):
        ch = ChannelInfo(excitation_wavelength=561)
        assert resolve_channel_color(ch, 0, 3) == "FF0000"

    def test_index_rotation(self):
        ch = ChannelInfo()
        assert resolve_channel_color(ch, 0, 3) == "FF0000"
        assert resolve_channel_color(ch, 1, 3) == "00FF00"
        assert resolve_channel_color(ch, 2, 3) == "0000FF"
        assert resolve_channel_color(ch, 3, 4) == "FF00FF"

    def test_priority_explicit_over_wavelength(self):
        ch = ChannelInfo(color="AABBCC", emission_wavelength=488)
        assert resolve_channel_color(ch, 0, 3) == "AABBCC"


# ============================================================================
# resolve_channel_label
# ============================================================================

class TestResolveChannelLabel:
    def test_explicit_name(self):
        ch = ChannelInfo(name="DAPI")
        assert resolve_channel_label(ch, 0) == "DAPI"

    def test_emission_wavelength_integer(self):
        ch = ChannelInfo(emission_wavelength=488.0)
        assert resolve_channel_label(ch, 0) == "488"

    def test_emission_wavelength_float(self):
        ch = ChannelInfo(emission_wavelength=488.5)
        assert resolve_channel_label(ch, 0) == "488.5"

    def test_excitation_wavelength(self):
        ch = ChannelInfo(excitation_wavelength=405.0)
        assert resolve_channel_label(ch, 0) == "405"

    def test_dye_name(self):
        ch = ChannelInfo(dye_name="Alexa Fluor 488")
        assert resolve_channel_label(ch, 0) == "Alexa Fluor 488"

    def test_fallback_channel_n(self):
        ch = ChannelInfo()
        assert resolve_channel_label(ch, 0) == "Channel 0"
        assert resolve_channel_label(ch, 3) == "Channel 3"

    def test_priority_name_over_wavelength(self):
        ch = ChannelInfo(name="GFP", emission_wavelength=488)
        assert resolve_channel_label(ch, 0) == "GFP"


# ============================================================================
# get_dtype_window
# ============================================================================

class TestGetDtypeWindow:
    def test_uint8(self):
        w = get_dtype_window("uint8")
        assert w == {"start": 0, "end": 255, "min": 0, "max": 255}

    def test_uint16(self):
        w = get_dtype_window("uint16")
        assert w == {"start": 0, "end": 65535, "min": 0, "max": 65535}

    def test_int8(self):
        w = get_dtype_window("int8")
        assert w == {"start": -128, "end": 127, "min": -128, "max": 127}

    def test_int16(self):
        w = get_dtype_window("int16")
        assert w == {"start": -32768, "end": 32767, "min": -32768, "max": 32767}

    def test_float32(self):
        w = get_dtype_window("float32")
        assert w == {"start": 0, "end": 1, "min": 0, "max": 1}

    def test_unknown_dtype(self):
        w = get_dtype_window("complex128")
        assert w == {"start": 0, "end": 65535, "min": 0, "max": 65535}


# ============================================================================
# build_omero_metadata
# ============================================================================

class TestBuildOmeroMetadata:
    def test_basic_3channel(self):
        channels = [
            ChannelInfo(name="DAPI", color="0000FF"),
            ChannelInfo(name="GFP", color="00FF00"),
            ChannelInfo(name="RFP", color="FF0000"),
        ]
        result = build_omero_metadata(
            channels, "uint16", (1, 3, 100, 512, 512), ["t", "c", "z", "y", "x"]
        )

        assert "channels" in result
        assert "rdefs" in result
        assert len(result["channels"]) == 3

        ch0 = result["channels"][0]
        assert ch0["label"] == "DAPI"
        assert ch0["color"] == "0000FF"
        assert ch0["active"] is True
        assert ch0["coefficient"] == 1
        assert ch0["family"] == "linear"
        assert ch0["inverted"] is False
        assert ch0["window"] == {"start": 0, "end": 65535, "min": 0, "max": 65535}

        assert result["rdefs"]["model"] == "color"
        assert result["rdefs"]["defaultT"] == 0
        assert result["rdefs"]["defaultZ"] == 50

    def test_single_channel_greyscale(self):
        channels = [ChannelInfo(name="Bright")]
        result = build_omero_metadata(
            channels, "uint8", (100, 512, 512), ["z", "y", "x"]
        )

        assert len(result["channels"]) == 1
        assert result["channels"][0]["color"] == "808080"  # Grey for single channel
        assert result["rdefs"]["model"] == "greyscale"

    def test_8plus_channels_greyscale(self):
        channels = [ChannelInfo() for _ in range(8)]
        result = build_omero_metadata(
            channels, "uint16", (1, 8, 50, 256, 256), ["t", "c", "z", "y", "x"]
        )
        assert result["rdefs"]["model"] == "greyscale"

    def test_none_channels_synthesize(self):
        result = build_omero_metadata(
            None, "uint16", (1, 3, 100, 512, 512), ["t", "c", "z", "y", "x"]
        )
        assert len(result["channels"]) == 3
        # Should get index-based colors
        assert result["channels"][0]["color"] == "FF0000"
        assert result["channels"][1]["color"] == "00FF00"
        assert result["channels"][2]["color"] == "0000FF"

    def test_no_c_axis_single_channel(self):
        result = build_omero_metadata(
            None, "uint16", (100, 512, 512), ["z", "y", "x"]
        )
        assert len(result["channels"]) == 1
        assert result["channels"][0]["label"] == "Channel 0"
        assert result["rdefs"]["model"] == "greyscale"

    def test_active_first_3_only(self):
        channels = [ChannelInfo() for _ in range(5)]
        result = build_omero_metadata(
            channels, "uint16", (1, 5, 50, 256, 256), ["t", "c", "z", "y", "x"]
        )
        assert result["channels"][0]["active"] is True
        assert result["channels"][1]["active"] is True
        assert result["channels"][2]["active"] is True
        assert result["channels"][3]["active"] is False
        assert result["channels"][4]["active"] is False

    def test_uint8_window(self):
        result = build_omero_metadata(
            None, "uint8", (100, 512, 512), ["z", "y", "x"]
        )
        assert result["channels"][0]["window"]["end"] == 255

    def test_float32_window(self):
        result = build_omero_metadata(
            None, "float32", (100, 512, 512), ["z", "y", "x"]
        )
        assert result["channels"][0]["window"]["end"] == 1

    def test_no_z_axis(self):
        result = build_omero_metadata(
            None, "uint16", (512, 512), ["y", "x"]
        )
        assert result["rdefs"]["defaultZ"] == 0  # 1 // 2 = 0

    def test_channel_count_mismatch_pads(self):
        channels = [ChannelInfo(name="DAPI")]
        result = build_omero_metadata(
            channels, "uint16", (1, 3, 100, 512, 512), ["t", "c", "z", "y", "x"]
        )
        assert len(result["channels"]) == 3
        assert result["channels"][0]["label"] == "DAPI"
        assert result["channels"][1]["label"] == "Channel 1"

    def test_channel_count_mismatch_truncates(self):
        channels = [ChannelInfo(name=f"Ch{i}") for i in range(5)]
        result = build_omero_metadata(
            channels, "uint16", (1, 2, 100, 512, 512), ["t", "c", "z", "y", "x"]
        )
        assert len(result["channels"]) == 2
        assert result["channels"][0]["label"] == "Ch0"
        assert result["channels"][1]["label"] == "Ch1"


# ============================================================================
# extract_channels_from_ome_xml
# ============================================================================

class TestExtractChannelsFromOmeXml:
    def test_basic_channels(self):
        ome_xml = '''<OME>
          <Image><Pixels>
            <Channel Name="DAPI" Color="-16776961" EmissionWavelength="461" Fluor="DAPI"/>
            <Channel Name="GFP" Color="16711935" EmissionWavelength="509"/>
          </Pixels></Image>
        </OME>'''
        channels = extract_channels_from_ome_xml(ome_xml)
        assert len(channels) == 2
        assert channels[0].name == "DAPI"
        assert channels[0].color == "FF0000"  # -16776961 = 0xFF0000FF RGBA → red
        assert channels[0].emission_wavelength == 461.0
        assert channels[0].dye_name == "DAPI"
        assert channels[1].name == "GFP"
        assert channels[1].color == "00FF00"  # 16711935 = 0x00FF00FF RGBA → green
        assert channels[1].emission_wavelength == 509.0

    def test_channel_without_name(self):
        ome_xml = '<OME><Image><Pixels><Channel Color="-16776961"/></Pixels></Image></OME>'
        channels = extract_channels_from_ome_xml(ome_xml)
        assert len(channels) == 1
        assert channels[0].name is None
        assert channels[0].color == "FF0000"

    def test_channel_without_color(self):
        ome_xml = '<OME><Image><Pixels><Channel Name="DAPI"/></Pixels></Image></OME>'
        channels = extract_channels_from_ome_xml(ome_xml)
        assert len(channels) == 1
        assert channels[0].name == "DAPI"
        assert channels[0].color is None

    def test_ome_namespace(self):
        ome_xml = '<OME><OME:Image><OME:Pixels><OME:Channel Name="Test" Color="255"/></OME:Pixels></OME:Image></OME>'
        channels = extract_channels_from_ome_xml(ome_xml)
        assert len(channels) == 1
        assert channels[0].name == "Test"

    def test_empty_string(self):
        assert extract_channels_from_ome_xml("") is None
        assert extract_channels_from_ome_xml(None) is None

    def test_no_channels(self):
        assert extract_channels_from_ome_xml("<OME></OME>") is None

    def test_self_closing_channel(self):
        ome_xml = '<OME><Image><Pixels><Channel Name="DAPI" Color="-1" EmissionWavelength="461"/></Pixels></Image></OME>'
        channels = extract_channels_from_ome_xml(ome_xml)
        assert len(channels) == 1
        assert channels[0].name == "DAPI"

    def test_excitation_wavelength(self):
        ome_xml = '<OME><Image><Pixels><Channel Name="405" ExcitationWavelength="405.0"/></Pixels></Image></OME>'
        channels = extract_channels_from_ome_xml(ome_xml)
        assert channels[0].excitation_wavelength == 405.0


# ============================================================================
# extract_channels_from_czi_xml
# ============================================================================

class TestExtractChannelsFromCziXml:
    def test_basic_czi_xml(self):
        czi_xml = '''<?xml version="1.0"?>
        <ImageDocument>
          <Metadata>
            <Information>
              <Image>
                <Dimensions>
                  <Channels>
                    <Channel Name="DAPI" Color="#FF0000FF">
                      <EmissionWavelength>461</EmissionWavelength>
                      <ExcitationWavelength>405</ExcitationWavelength>
                      <DyeName>DAPI</DyeName>
                    </Channel>
                    <Channel Name="GFP" Color="#FF00FF00">
                      <EmissionWavelength>509</EmissionWavelength>
                    </Channel>
                  </Channels>
                </Dimensions>
              </Image>
            </Information>
          </Metadata>
        </ImageDocument>'''
        channels = extract_channels_from_czi_xml(czi_xml)
        assert len(channels) == 2
        assert channels[0].name == "DAPI"
        assert channels[0].color == "0000FF"  # #FF0000FF → strip #AA → 0000FF
        assert channels[0].emission_wavelength == 461.0
        assert channels[0].excitation_wavelength == 405.0
        assert channels[0].dye_name == "DAPI"
        assert channels[1].name == "GFP"
        assert channels[1].color == "00FF00"  # #FF00FF00 → strip #AA → 00FF00
        assert channels[1].emission_wavelength == 509.0

    def test_czi_integer_colors(self):
        czi_xml = '''<ImageDocument><Metadata><Information><Image><Dimensions><Channels>
            <Channel Name="Ch1" Color="-16776961">
              <EmissionWavelength>461</EmissionWavelength>
            </Channel>
        </Channels></Dimensions></Image></Information></Metadata></ImageDocument>'''
        channels = extract_channels_from_czi_xml(czi_xml)
        assert len(channels) == 1
        assert channels[0].color == "FF0000"  # -16776961 = 0xFF0000FF RGBA → red

    def test_empty_string(self):
        assert extract_channels_from_czi_xml("") is None
        assert extract_channels_from_czi_xml(None) is None

    def test_invalid_xml(self):
        assert extract_channels_from_czi_xml("<not valid xml") is None

    def test_no_channels(self):
        assert extract_channels_from_czi_xml("<ImageDocument></ImageDocument>") is None


# ============================================================================
# extract_channel_info (router)
# ============================================================================

class TestExtractChannelInfo:
    def test_czi_routing(self):
        raw_xml = '''<ImageDocument><Metadata><Information><Image><Dimensions><Channels>
            <Channel Name="DAPI" Color="-16776961"/>
        </Channels></Dimensions></Image></Information></Metadata></ImageDocument>'''
        result = extract_channel_info({"raw_xml": raw_xml}, "czi")
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "DAPI"

    def test_tiff_routing_via_ome_xml(self):
        ome_xml = '<OME><Image><Pixels><Channel Name="GFP" Color="16711935"/></Pixels></Image></OME>'
        result = extract_channel_info({"ome_xml": ome_xml}, "tiff")
        assert result is not None
        assert result[0].name == "GFP"

    def test_fallback_to_ome_xml(self):
        ome_xml = '<OME><Image><Pixels><Channel Name="Test" Color="255"/></Pixels></Image></OME>'
        result = extract_channel_info({"ome_xml": ome_xml}, "unknown_format")
        assert result is not None
        assert result[0].name == "Test"

    def test_empty_metadata(self):
        result = extract_channel_info({}, "czi")
        assert result is None

    def test_none_source_format(self):
        result = extract_channel_info({}, None)
        assert result is None


# ============================================================================
# Integration: build_omero_metadata with extracted channels
# ============================================================================

class TestOmeroIntegration:
    def test_ome_xml_to_omero(self):
        """End-to-end: OME-XML → extract → build → OMERO block."""
        ome_xml = '''<OME><Image><Pixels>
            <Channel Name="DAPI" Color="-16776961" EmissionWavelength="461"/>
            <Channel Name="GFP" Color="-16711936" EmissionWavelength="509"/>
            <Channel Name="RFP" Color="-65536" EmissionWavelength="590"/>
        </Pixels></Image></OME>'''

        channels = extract_channels_from_ome_xml(ome_xml)
        result = build_omero_metadata(
            channels, "uint16", (1, 3, 100, 512, 512), ["t", "c", "z", "y", "x"]
        )

        assert len(result["channels"]) == 3
        assert result["channels"][0]["label"] == "DAPI"
        assert result["channels"][0]["color"] == "FF0000"  # -16776961 RGBA → red
        assert result["channels"][1]["label"] == "GFP"
        assert result["channels"][1]["color"] == "FF00FF"  # -16711936 RGBA → magenta
        assert result["channels"][2]["label"] == "RFP"
        assert result["channels"][2]["color"] == "FFFF00"  # -65536 RGBA → yellow
        assert result["rdefs"]["model"] == "color"
        assert result["rdefs"]["defaultZ"] == 50

    def test_czi_xml_to_omero(self):
        """End-to-end: CZI XML → extract → build → OMERO block."""
        czi_xml = '''<ImageDocument><Metadata><Information><Image><Dimensions><Channels>
            <Channel Name="DAPI" Color="-16776961">
              <EmissionWavelength>461</EmissionWavelength>
            </Channel>
            <Channel Name="GFP" Color="-16711936">
              <EmissionWavelength>509</EmissionWavelength>
            </Channel>
        </Channels></Dimensions></Image></Information></Metadata></ImageDocument>'''

        channels = extract_channels_from_czi_xml(czi_xml)
        result = build_omero_metadata(
            channels, "uint8", (1, 2, 50, 256, 256), ["t", "c", "z", "y", "x"]
        )

        assert len(result["channels"]) == 2
        assert result["channels"][0]["label"] == "DAPI"
        assert result["channels"][0]["color"] == "FF0000"  # -16776961 RGBA → red
        assert result["channels"][0]["window"]["end"] == 255  # uint8
        assert result["channels"][1]["label"] == "GFP"
        assert result["channels"][1]["color"] == "FF00FF"  # -16711936 RGBA → magenta
        assert result["rdefs"]["model"] == "color"
        assert result["rdefs"]["defaultZ"] == 25
