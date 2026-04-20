"""
OMERO rendering metadata for OME-NGFF output.

Generates the ``omero`` block (channel names, colors, display windows,
rendering defaults) that visualization tools need to display images
correctly. The algorithm is ported from bioformats2raw's Converter.java
and Colors.java.

Public API:
    ChannelInfo              — format-agnostic channel descriptor
    build_omero_metadata     — main builder (channels + rdefs)
    extract_channel_info     — format-routing dispatcher
    extract_channels_from_*  — per-format extractors
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# ChannelInfo dataclass
# ============================================================================

@dataclass
class ChannelInfo:
    """Format-agnostic channel metadata used to build OMERO rendering hints.

    All fields are optional — the builder uses fallback cascades when fields
    are missing (matching bioformats2raw's priority logic).
    """
    name: Optional[str] = None
    color: Optional[str] = None  # hex RGB ("FF0000") or int RGBA
    emission_wavelength: Optional[float] = None   # nanometers
    excitation_wavelength: Optional[float] = None  # nanometers
    dye_name: Optional[str] = None


# ============================================================================
# Color helpers
# ============================================================================

# Wavelength thresholds (from bioformats2raw Colors.java)
_BLUE_TO_GREEN_NM = 500.0
_GREEN_TO_RED_NM = 560.0

# Default colors cycled by channel index (R, G, B, Magenta, Yellow, Cyan)
_DEFAULT_CHANNEL_COLORS = ["FF0000", "00FF00", "0000FF", "FF00FF", "FFFF00", "00FFFF"]


def _rgba_int_to_hex_rgb(color_int: int) -> str:
    """Convert a signed/unsigned 32-bit RGBA-packed integer to 6-digit hex RGB.

    OME-XML Color attributes are packed as RGBA (R in most-significant byte,
    A in least-significant byte). This matches bioformats2raw's extraction:
    ``(color.getValue() >> 8) & 0xffffff``.
    """
    if color_int < 0:
        color_int = color_int & 0xFFFFFFFF
    return f"{(color_int >> 8) & 0xFFFFFF:06X}"


def wavelength_to_color(wavelength_nm: float) -> str:
    """Map a wavelength in nm to a hex RGB color.

    Uses the same thresholds as bioformats2raw Colors.colorFromWavelength:
      <500 nm  → blue  (0000FF)
      500–560  → green (00FF00)
      >560 nm  → red   (FF0000)
    """
    if wavelength_nm < _BLUE_TO_GREEN_NM:
        return "0000FF"
    elif wavelength_nm < _GREEN_TO_RED_NM:
        return "00FF00"
    return "FF0000"


def resolve_channel_color(
    channel: ChannelInfo, index: int, num_channels: int
) -> str:
    """Determine channel display color using bioformats2raw fallback cascade.

    Priority: explicit color → single-channel grey → emission wavelength →
    excitation wavelength → RGB rotation by index.
    """
    # 1. Explicit color
    if channel.color is not None:
        c = channel.color
        if isinstance(c, int):
            return _rgba_int_to_hex_rgb(c)
        if isinstance(c, str):
            c = c.lstrip('#')
            if len(c) == 6:
                return c.upper()
            # Try parsing as integer string
            try:
                return _rgba_int_to_hex_rgb(int(c))
            except ValueError:
                pass

    # 2. Single channel → grey
    if num_channels == 1:
        return "808080"

    # 3. Emission wavelength
    if channel.emission_wavelength is not None:
        return wavelength_to_color(channel.emission_wavelength)

    # 4. Excitation wavelength
    if channel.excitation_wavelength is not None:
        return wavelength_to_color(channel.excitation_wavelength)

    # 5. Index-based rotation
    return _DEFAULT_CHANNEL_COLORS[index % len(_DEFAULT_CHANNEL_COLORS)]


def resolve_channel_label(channel: ChannelInfo, index: int) -> str:
    """Determine channel display name using bioformats2raw fallback cascade.

    Priority: explicit name → emission wavelength → excitation wavelength →
    dye name → 'Channel N'.
    """
    if channel.name:
        return channel.name

    if channel.emission_wavelength is not None:
        w = channel.emission_wavelength
        return str(int(w)) if w == int(w) else str(w)

    if channel.excitation_wavelength is not None:
        w = channel.excitation_wavelength
        return str(int(w)) if w == int(w) else str(w)

    if channel.dye_name:
        return channel.dye_name

    return f"Channel {index}"


# ============================================================================
# Window / dtype helpers
# ============================================================================

def get_dtype_window(dtype: str) -> dict:
    """Return display window bounds based on pixel data type.

    Uses bioformats2raw's ZarrTypes.getRange logic for integer types
    and sensible defaults for floating point.
    """
    dtype = dtype.lower().replace('numpy.', '').replace('np.', '')
    windows = {
        'uint8':   {"start": 0, "end": 255,   "min": 0, "max": 255},
        'uint16':  {"start": 0, "end": 65535,  "min": 0, "max": 65535},
        'int8':    {"start": -128, "end": 127,  "min": -128, "max": 127},
        'int16':   {"start": -32768, "end": 32767, "min": -32768, "max": 32767},
        'float32': {"start": 0, "end": 1, "min": 0, "max": 1},
        'float64': {"start": 0, "end": 1, "min": 0, "max": 1},
    }
    return windows.get(dtype, {"start": 0, "end": 65535, "min": 0, "max": 65535})


# ============================================================================
# Main OMERO builder
# ============================================================================

def build_omero_metadata(
    channels: Optional[List[ChannelInfo]],
    dtype: str,
    shape: tuple,
    axes: List[str],
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
) -> dict:
    """Build a complete OMERO rendering metadata block.

    This is the main entry point for OMERO metadata generation, porting
    the algorithm from bioformats2raw's Converter.java.

    Args:
        channels: Per-channel metadata (None to synthesize from shape).
        dtype: Pixel data type string (e.g. 'uint16').
        shape: Array shape tuple.
        axes: Axis name list matching shape (e.g. ['t','c','z','y','x']).
        channel_minmax: Optional per-channel (min, max) tuples computed from
            actual pixel data during conversion. Overrides dtype-based defaults
            when provided (matching bioformats2raw's MinMaxCalculator behavior).

    Returns:
        Dict with 'channels' and 'rdefs' keys ready for OME-NGFF attributes.
    """
    # Normalize axis names
    axes_lower = [a.lower() for a in axes]

    # Determine number of channels from shape/axes
    if 'c' in axes_lower:
        c_idx = axes_lower.index('c')
        num_channels = shape[c_idx]
    else:
        num_channels = 1

    # Synthesize channel list if none provided
    if not channels:
        channels = [ChannelInfo() for _ in range(num_channels)]
    elif len(channels) != num_channels:
        # Pad or truncate to match actual channel count
        if len(channels) < num_channels:
            channels = list(channels) + [ChannelInfo() for _ in range(num_channels - len(channels))]
        else:
            channels = list(channels)[:num_channels]

    # Determine Z size for defaultZ
    if 'z' in axes_lower:
        z_idx = axes_lower.index('z')
        size_z = shape[z_idx]
    else:
        size_z = 1

    # Build default window from dtype (fallback)
    default_window = get_dtype_window(dtype)

    # Build per-channel entries
    channel_entries = []
    for i, ch in enumerate(channels):
        # Use data-driven min/max if available, else dtype defaults
        if channel_minmax and i < len(channel_minmax):
            cmin, cmax = channel_minmax[i]
            window = {"start": cmin, "end": cmax, "min": cmin, "max": cmax}
        else:
            window = dict(default_window)

        channel_entries.append({
            "active": i < 3,
            "coefficient": 1,
            "color": resolve_channel_color(ch, i, num_channels),
            "family": "linear",
            "inverted": False,
            "label": resolve_channel_label(ch, i),
            "window": window,
        })

    # Rendering definitions
    color_render = 1 < num_channels < 8
    rdefs = {
        "defaultT": 0,
        "defaultZ": size_z // 2,
        "model": "color" if color_render else "greyscale",
    }

    return {"channels": channel_entries, "rdefs": rdefs}


# ============================================================================
# Format-specific channel extractors
# ============================================================================

def extract_channels_from_ome_xml(ome_xml: str) -> Optional[List[ChannelInfo]]:
    """Extract channel metadata from an OME-XML string into ChannelInfo objects.

    Parses <Channel> elements for Name, Color, EmissionWavelength,
    ExcitationWavelength, and Fluor attributes. Does NOT skip channels
    missing Name or Color — the builder's fallback cascades handle missing
    fields.

    Args:
        ome_xml: Raw OME-XML string.

    Returns:
        List of ChannelInfo, or None if no channels found.
    """
    if not ome_xml or not isinstance(ome_xml, str):
        return None

    channels = []
    # Match <Channel ...> or <OME:Channel ...> elements
    channel_pattern = r'<(?:OME:)?Channel\s+([^>]+?)/?>'

    for channel_match in re.finditer(channel_pattern, ome_xml):
        attrs = channel_match.group(1)

        name = None
        name_match = re.search(r'Name\s*=\s*"([^"]*)"', attrs)
        if name_match:
            name = name_match.group(1)

        color = None
        color_match = re.search(r'Color\s*=\s*"(-?\d+)"', attrs)
        if color_match:
            color = _rgba_int_to_hex_rgb(int(color_match.group(1)))

        emission = None
        em_match = re.search(r'EmissionWavelength\s*=\s*"([0-9.]+)"', attrs)
        if em_match:
            try:
                emission = float(em_match.group(1))
            except ValueError:
                pass

        excitation = None
        ex_match = re.search(r'ExcitationWavelength\s*=\s*"([0-9.]+)"', attrs)
        if ex_match:
            try:
                excitation = float(ex_match.group(1))
            except ValueError:
                pass

        fluor = None
        fluor_match = re.search(r'Fluor\s*=\s*"([^"]*)"', attrs)
        if fluor_match:
            fluor = fluor_match.group(1)

        channels.append(ChannelInfo(
            name=name,
            color=color,
            emission_wavelength=emission,
            excitation_wavelength=excitation,
            dye_name=fluor,
        ))

    return channels if channels else None


def extract_channels_from_czi_xml(czi_xml: str) -> Optional[List[ChannelInfo]]:
    """Extract channel metadata from CZI raw metadata XML.

    CZI stores channel info under:
      //Metadata/Information/Image/Dimensions/Channels/Channel

    Display colors come from //DisplaySetting//Channel/Color (set in Zen).

    Args:
        czi_xml: Raw CZI metadata XML string.

    Returns:
        List of ChannelInfo, or None if no channels found.
    """
    if not czi_xml or not isinstance(czi_xml, str):
        return None

    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(czi_xml)
    except ET.ParseError:
        return None

    # CZI XML structure: <ImageDocument><Metadata><Information><Image><Dimensions><Channels>
    # Try multiple paths since CZI XML structure can vary
    channel_elements = root.findall('.//Dimensions/Channels/Channel')
    if not channel_elements:
        channel_elements = root.findall('.//Information/Image/Dimensions/Channels/Channel')
    if not channel_elements:
        channel_elements = root.findall('.//Channels/Channel')
    if not channel_elements:
        return None

    # Build a name→color map from DisplaySetting (display colors set in Zen)
    display_colors = {}
    for ds_ch in root.findall('.//DisplaySetting//Channel'):
        ds_name = ds_ch.get('Name') or ds_ch.get('Id')
        color_elem = ds_ch.find('Color')
        if ds_name and color_elem is not None and color_elem.text:
            c = color_elem.text.strip()
            if c.startswith('#') and len(c) == 7:
                display_colors[ds_name] = c[1:].upper()  # #RRGGBB → RRGGBB

    channels = []
    for ch_elem in channel_elements:
        name = ch_elem.get('Name')

        # Priority: Dimensions/Channel@Color → DisplaySetting color
        color = None
        color_attr = ch_elem.get('Color')
        if color_attr:
            try:
                color = _rgba_int_to_hex_rgb(int(color_attr))
            except ValueError:
                # CZI may use #AARRGGBB hex strings
                if color_attr.startswith('#') and len(color_attr) == 9:
                    color = color_attr[3:9].upper()  # strip #AA, keep RRGGBB
        if not color and name and name in display_colors:
            color = display_colors[name]

        emission = None
        em_elem = ch_elem.find('EmissionWavelength')
        if em_elem is not None and em_elem.text:
            try:
                emission = float(em_elem.text)
            except ValueError:
                pass

        excitation = None
        ex_elem = ch_elem.find('ExcitationWavelength')
        if ex_elem is not None and ex_elem.text:
            try:
                excitation = float(ex_elem.text)
            except ValueError:
                pass

        dye = None
        dye_elem = ch_elem.find('DyeName')
        if dye_elem is not None and dye_elem.text:
            dye = dye_elem.text
        if not dye:
            fluor_elem = ch_elem.find('Fluor')
            if fluor_elem is not None and fluor_elem.text:
                dye = fluor_elem.text

        channels.append(ChannelInfo(
            name=name,
            color=color,
            emission_wavelength=emission,
            excitation_wavelength=excitation,
            dye_name=dye,
        ))

    return channels if channels else None


def extract_channels_from_nd2(nd2_path: str) -> Optional[List[ChannelInfo]]:
    """Extract channel metadata from an ND2 file using the nd2 Python API.

    Falls back to OME-XML extraction if the direct API is unavailable.

    Args:
        nd2_path: Path to .nd2 file.

    Returns:
        List of ChannelInfo, or None if extraction fails.
    """
    try:
        import nd2
        with nd2.ND2File(nd2_path) as f:
            # Try direct metadata API first
            try:
                md_channels = f.metadata.channels
                if md_channels:
                    channels = []
                    for ch in md_channels:
                        name = getattr(ch.channel, 'name', None) if hasattr(ch, 'channel') else None
                        # Some nd2 versions use ch.name directly
                        if not name:
                            name = getattr(ch, 'name', None)
                        channels.append(ChannelInfo(name=name))
                    if channels:
                        return channels
            except (AttributeError, TypeError):
                pass

            # Fallback to OME-XML
            try:
                ome_meta = f.ome_metadata()
                if ome_meta:
                    ome_xml = ome_meta.to_xml() if hasattr(ome_meta, 'to_xml') else str(ome_meta)
                    return extract_channels_from_ome_xml(ome_xml)
            except Exception:
                pass
    except Exception:
        pass

    return None


def extract_channels_from_ims(ims_path: str) -> Optional[List[ChannelInfo]]:
    """Extract channel metadata from an Imaris IMS (HDF5) file.

    IMS stores channels under DataSet/ResolutionLevel 0/TimePoint 0/Channel N.
    Channel names may be in DataSetInfo/Channel N attributes.

    Args:
        ims_path: Path to .ims file.

    Returns:
        List of ChannelInfo, or None if extraction fails.
    """
    try:
        import h5py
        with h5py.File(ims_path, 'r') as f:
            channels = []

            # Try to get channel names from DataSetInfo
            for i in range(100):  # reasonable upper bound
                key = f'DataSetInfo/Channel {i}'
                if key not in f:
                    break
                group = f[key]
                name = None
                for attr_name in ['Name', 'name', 'Description', 'description']:
                    if attr_name in group.attrs:
                        val = group.attrs[attr_name]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='replace')
                        name = str(val).strip()
                        break
                channels.append(ChannelInfo(name=name if name else None))

            if channels:
                return channels

            # Fallback: count channels from dataset structure
            tp_group = f.get('DataSet/ResolutionLevel 0/TimePoint 0')
            if tp_group:
                channel_keys = sorted(
                    k for k in tp_group.keys() if k.startswith('Channel')
                )
                if channel_keys:
                    return [ChannelInfo() for _ in channel_keys]
    except Exception:
        pass

    return None


# ============================================================================
# Format-routing dispatcher
# ============================================================================

def extract_channel_info(
    raw_metadata: dict, source_format: str, reader_path: Optional[str] = None
) -> Optional[List[ChannelInfo]]:
    """Route to the appropriate channel extractor based on source format.

    This is the main dispatcher called by the converter to extract channel
    metadata from whatever the reader provides.

    Args:
        raw_metadata: Dict from reader.get_metadata()
        source_format: Format identifier (e.g. 'czi', 'nd2', 'tiff', 'ims')
        reader_path: Path to source file (needed for nd2/ims direct API access)

    Returns:
        List of ChannelInfo, or None.
    """
    fmt = (source_format or '').lower()

    # CZI: use raw_xml with CZI-specific parser
    if fmt == 'czi':
        raw_xml = raw_metadata.get('raw_xml')
        if raw_xml:
            result = extract_channels_from_czi_xml(raw_xml)
            if result:
                return result

    # ND2: try direct API, then OME-XML fallback
    if fmt == 'nd2' and reader_path:
        result = extract_channels_from_nd2(reader_path)
        if result:
            return result

    # IMS: try HDF5 metadata
    if fmt in ('ims', 'imaris', 'hdf5') and reader_path:
        result = extract_channels_from_ims(reader_path)
        if result:
            return result

    # Generic fallback: try OME-XML from any format
    ome_xml = raw_metadata.get('ome_xml') or raw_metadata.get('raw_xml')
    if ome_xml:
        return extract_channels_from_ome_xml(ome_xml)

    return None
