<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:str="http://exslt.org/strings" extension-element-prefixes="str">


    <xsl:import href="Channels.xsl"/>
    <xsl:import href="Plane.xsl"/>

    <!-- SizeX -->
    <xsl:template match="SizeX">
        <xsl:attribute name="SizeX">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- SizeY -->
    <xsl:template match="SizeY">
        <xsl:attribute name="SizeY">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- SizeZ -->
    <xsl:template name="SizeZ">
        <xsl:param name="simg"/>
        <xsl:attribute name="SizeZ">
            <xsl:choose>
                <xsl:when test="$simg/SizeZ">
                    <xsl:value-of select="$simg/SizeZ"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:text>1</xsl:text>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

    <!-- SizeC -->
    <xsl:template name="SizeC">
        <xsl:param name="simg"/>
        <xsl:attribute name="SizeC">
            <xsl:choose>
                <xsl:when test="$simg/SizeC">
                    <xsl:value-of select="$simg/SizeC"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:text>1</xsl:text>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

    <!-- SizeT -->
    <xsl:template name="SizeT">
        <xsl:param name="simg"/>
        <xsl:attribute name="SizeT">
            <xsl:choose>
                <xsl:when test="$simg/SizeT">
                    <xsl:value-of select="$simg/SizeT"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:text>1</xsl:text>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

    <!-- /Metadata/Scaling/Items/Distance[@Id=X]/Value => /OME/Image/Pixels/@PhysicalSizeX -->
    <xsl:template match="Distance">
        <xsl:param name="dim"/>
        <xsl:if test="Value &gt; 0">
            <xsl:attribute name="PhysicalSize{$dim}">
                <!-- `Value` is always in meters. Convert it to µm. -->
                <xsl:value-of select="Value * 1000000"/>
            </xsl:attribute>
            <xsl:attribute name="PhysicalSize{$dim}Unit">
                <xsl:text>µm</xsl:text>
            </xsl:attribute>
        </xsl:if>
    </xsl:template>

    <xsl:template match="Items">
        <xsl:apply-templates select="Distance[@Id='X']">
            <xsl:with-param name="dim">X</xsl:with-param>
        </xsl:apply-templates>
        <xsl:apply-templates select="Distance[@Id='Y']">
            <xsl:with-param name="dim">Y</xsl:with-param>
        </xsl:apply-templates>
        <xsl:apply-templates select="Distance[@Id='Z']">
            <xsl:with-param name="dim">Z</xsl:with-param>
        </xsl:apply-templates>
    </xsl:template>




    <xsl:template name="Sizes">
        <xsl:param name="image"/>
        <!-- If there is a `ParameterCollection` containing `Binning` with a status of "SuperValid", we can use the `ImageFrame` from that `ParameterCollection`
             to get Size X and Y for a single tile. If no `ParameterCollection/Binning/@Status` is "SuperValid", we can use any other `ParameterCollection/ImageFrame`. -->
        <xsl:variable name="param_collection" select="/ImageDocument/Metadata/HardwareSetting/ParameterCollection[ImageFrame and Binning/@Status = 'SuperValid'] | /ImageDocument/Metadata/HardwareSetting/ParameterCollection[ImageFrame and not(/ImageDocument/Metadata/HardwareSetting/ParameterCollection[Binning/@Status = 'SuperValid'])]" />
        <xsl:choose>
            <!-- If we found a `ParameterCollection` meeting the above criteria, use it to calculate SizeX and SizeY. Otherwsie, use the
                 default SizeX and SizeY elements. -->
            <xsl:when test="$param_collection">
                <!-- To get accurate values for SizeX and SizeY, we need to get the values for a single tile. These are available on the `ImageFrame` element,
                     which will have a value of the form "x_offset,y_offset,x_pixels,y_pixels".
                     NOTE: While this is the most accurate way to extract SizeX and SizeY that we know of, it may not be entirely correct for mosaic images,
                     depending on if users expect these values to correspond to a single tile or a merged tile. We will likely need to revisit this in the future. -->
                <xsl:variable name="image_frame_vals" select="str:tokenize($param_collection/ImageFrame, ',')" />
                <xsl:attribute name="SizeX">
                    <xsl:value-of select="$image_frame_vals[3]"/>
                </xsl:attribute>
                <xsl:attribute name="SizeY">
                    <xsl:value-of select="$image_frame_vals[4]"/>
                </xsl:attribute>
            </xsl:when>
            <xsl:otherwise>
                <xsl:apply-templates select="$image/SizeX"/>
                <xsl:apply-templates select="$image/SizeY"/>
            </xsl:otherwise>
        </xsl:choose>
        <xsl:call-template name="SizeZ">
            <xsl:with-param name="simg" select="$image"/>
        </xsl:call-template>
        <xsl:call-template name="SizeC">
            <xsl:with-param name="simg" select="$image"/>
        </xsl:call-template>
        <xsl:call-template name="SizeT">
            <xsl:with-param name="simg" select="$image"/>
        </xsl:call-template>
        <xsl:apply-templates select="/ImageDocument/Metadata/Scaling/Items"/>
    </xsl:template>


    <!-- PixelType -->
    <!-- zisraw/CommonTypes.xsd: 224 -->
    <!-- ome/ome.xsd: 1689 -->
    <xsl:template match="PixelType">
        <xsl:variable name="pixel_data" select="."/>
        <xsl:attribute name="Type">
            <!-- Attempt to get to map the ZISRAW pixeltype to OME pixeltype -->
            <xsl:choose>
                <xsl:when test="$pixel_data='Gray8'">
                    <xsl:text>uint8</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Gray16'">
                    <xsl:text>uint16</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Gray32'">
                    <!-- Zeiss Docs: planned-->
                    <xsl:text>float</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Gray64'">
                    <!-- Zeiss Docs: planned-->
                    <xsl:text>double</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Bgr24'">
                    <xsl:text>uint8</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Bgr48'">
                    <xsl:text>uint16</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Gray32Float'">
                    <!-- float, specifically an IEEE 4 byte float-->
                    <xsl:text>float</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Bgr96Float'">
                    <!-- float, specifically an IEEE 4 byte float-->
                    <xsl:text>float</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Gray64ComplexFloat'">
                    <!-- 2 x float, specifically an IEEE 4 byte float-->
                    <xsl:text>complex</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Bgr192ComplexFloat'">
                    <!-- a BGR triplet of (2 x float), specifically an IEEE 4 byte float-->
                    <xsl:text>complex</xsl:text>
                </xsl:when>
                <xsl:when test="$pixel_data='Bgra32'">
                    <!-- Bgra32 = 3 uint8 followed by a 8 bit transparency value-->
                    <!-- From other sources (non-Zeiss) the a value is a uint8-->
                    <xsl:text>uint8</xsl:text>
                </xsl:when>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

    <xsl:template match="Image">
        <xsl:param name="idx" />
        <xsl:element name="ome:Pixels">
            <xsl:attribute name="ID">
                <xsl:value-of select="concat('Pixels:', position() - 1, '-', $idx)"/>
            </xsl:attribute>
            <xsl:attribute name="DimensionOrder">
                <!--  Hardcoded for AICSImageIO  -->
                <xsl:text>XYZCT</xsl:text>
            </xsl:attribute>
            <xsl:if test="ComponentBitCount and string-length(ComponentBitCount) &gt; 0">
                <xsl:attribute name="SignificantBits">
                    <xsl:value-of select="ComponentBitCount"/>
                </xsl:attribute>
            </xsl:if>
            <xsl:apply-templates select="PixelType"/>
            <!-- SizeX SizeY .... SizeT -->
            <xsl:call-template name="Sizes">
                <xsl:with-param name="image" select="."/>
            </xsl:call-template>
            <!-- Channel -->
            <xsl:apply-templates select="/ImageDocument/Metadata/Information/Image/Dimensions/Channels">
                <xsl:with-param name="idx" select="$idx"/>
            </xsl:apply-templates>
            <xsl:element name="ome:TiffData">
                <xsl:attribute name="IFD">
                    <xsl:value-of select="position()"/>
                </xsl:attribute>
            </xsl:element>
            <!-- Plane -->
            <xsl:apply-templates select="$subblocks" mode="plane">
                <xsl:with-param name="scene_index" select="$idx" />
            </xsl:apply-templates>
        </xsl:element>
    </xsl:template>

</xsl:stylesheet>
