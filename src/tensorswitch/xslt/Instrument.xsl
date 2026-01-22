<?xml version="1.0" encoding="UTF-8"?>
<!-- #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # # Comments in this template will generally be pointers from spec to spec # Example: # Get Instrument Info # zisraw/Instrument.xsd: 45 #
ome/ome.xsd: 979 # # This means that for more details on how this section of the template was created # view line 45 of the zisraw/Instrument.xsd file and view line 979 of the ome/ome.xsd file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

<xsl:stylesheet version="1.1" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <!-- Includes -->
    <xsl:include href="CommonTypes.xsl"/>

    <!-- ManufacturerSpec/Model -->
    <!-- zisraw/Instrument.xsd: 26 -->
    <!-- ome/ome.xsd: 6389 -->
    <xsl:template match="Model">
        <xsl:attribute name="Model">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/SerialNumber -->
    <!-- zisraw/Instrument.xsd: 31 -->
    <!-- ome/ome.xsd: 6395 -->
    <xsl:template match="SerialNumber">
        <xsl:attribute name="SerialNumber">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/LotNumber -->
    <!-- zisraw/Instrument.xsd: 36 -->
    <!-- ome/ome.xsd: 6401 -->
    <xsl:template match="LotNumber">
        <xsl:attribute name="LotNumber">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- ManufacturerSpec/SpecsFile -->
    <!-- zisraw: No valid zisraw spec -->
    <!-- ome/ome.xsd: 6407 -->
    <!-- Note: This is required by the OME-4DN spec but not provided by ZISRAW -->
    <xsl:template match="SpecsFile">
        <xsl:attribute name="SpecsFile">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- Manufacturer -->
    <!-- zisraw/Instrument.xsd: 11 -->
    <!-- ome/ome.xsd: 6378 -->
    <xsl:template name="Manufacturer" match="Manufacturer">
        <xsl:attribute name="Manufacturer">
            <xsl:value-of select="."/>
        </xsl:attribute>

        <xsl:apply-templates select="Model"/>
        <xsl:apply-templates select="SerialNumber"/>
        <xsl:apply-templates select="LotNumber"/>
        <xsl:apply-templates select="SpecsFile"/>
    </xsl:template>

    <!-- Type -->
    <!-- zisraw/Instrument.xsd: 166 -->
    <!-- ome/ome.xsd: 7996 -->
    <xsl:template match="Type">
        <xsl:attribute name="Type">
            <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>

    <!-- Microscope -->
    <!-- zisraw/Instrument.xsd: 50 -->
    <!-- ome/ome.xsd: 2039 -->
    <xsl:template match="Microscope">
        <xsl:element name="ome:Microscope">
            <!-- Not in OME but in BINA -->
            <!-- <xsl:apply-templates select="@Id"/> -->
            <!-- <xsl:apply-templates select="@Name"/> -->
            <!-- <xsl:apply-templates select="Type"/> -->

            <xsl:attribute name="Manufacturer">
                <xsl:text>Zeiss</xsl:text>
            </xsl:attribute>
            <xsl:attribute name="Model">
                <xsl:value-of select="Manufacturer/Model"/>
            </xsl:attribute>
            <xsl:attribute name="SerialNumber">
                <xsl:value-of select="Manufacturer/SerialNumber"/>
            </xsl:attribute>
            <xsl:apply-templates select="Manufacturer/LotNumber"/>
        </xsl:element>
    </xsl:template>

    <xsl:template match="LightSources">
        <xsl:for-each select="LightSource">
            <xsl:choose>
                <xsl:when test="LightSourceType/Laser">
                    <xsl:element name="ome:Laser">
                        <xsl:attribute name="ID">
                            <xsl:text>LightSource:</xsl:text>
                            <xsl:value-of select="@Id"/>
                        </xsl:attribute>
                        <!-- Add Wavelength only if it's not empty -->
                        <xsl:if test="string(LightSourceType/Laser/Wavelength) != ''">
                            <xsl:attribute name="Wavelength">
                                <xsl:value-of select="LightSourceType/Laser/Wavelength"/>
                            </xsl:attribute>
                        </xsl:if>
                    </xsl:element>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:element name="ome:Filament">
                        <xsl:attribute name="ID">
                            <xsl:value-of select="@Id"/>
                        </xsl:attribute>
                        <xsl:attribute name="Type">
                            <xsl:text>Other</xsl:text>
                        </xsl:attribute>
                    </xsl:element>
                </xsl:otherwise>
            </xsl:choose>

        </xsl:for-each>
    </xsl:template>

    <xsl:template match="Detectors">
        <xsl:for-each select="Detector">
            <xsl:element name="ome:Detector">
                <xsl:attribute name="ID">
                    <!-- Convert any whitespace to '_' -->
                    <xsl:value-of select="translate(normalize-space(@Id), ' ', '_')"/>
                </xsl:attribute>
                <xsl:attribute name="Model">
                    <xsl:value-of select="Manufacturer/Model"/>
                </xsl:attribute>
            </xsl:element>
        </xsl:for-each>
    </xsl:template>

    <xsl:template match="Objectives">
        <xsl:for-each select="Objective">
            <xsl:element name="ome:Objective">
                <xsl:if test="Manufacturer/Manufacturer">
                    <xsl:attribute name="Manufacturer">
                        <xsl:value-of select="Manufacturer/Manufacturer"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Manufacturer/Model">
                    <xsl:attribute name="Model">
                        <xsl:value-of select="Manufacturer/Model"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Manufacturer/SerialNumber">
                    <xsl:attribute name="SerialNumber">
                        <xsl:value-of select="Manufacturer/SerialNumber"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Manufacturer/LotNumber">
                    <xsl:attribute name="LotNumber">
                        <xsl:value-of select="Manufacturer/LotNumber"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="@Id">
                    <xsl:attribute name="ID">
                        <xsl:value-of select="@Id"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Correction">
                    <xsl:attribute name="Correction">
                        <xsl:value-of select="Correction"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Immersion">
                    <xsl:attribute name="Immersion">
                        <xsl:value-of select="Immersion"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="LensNA">
                    <xsl:attribute name="LensNA">
                        <xsl:value-of select="LensNA"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="NominalMagnification">
                    <xsl:attribute name="NominalMagnification">
                        <xsl:value-of select="NominalMagnification"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="CalibratedMagnification">
                    <xsl:attribute name="CalibratedMagnification">
                        <xsl:value-of select="CalibratedMagnification"/>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="WorkingDistance">
                    <xsl:attribute name="WorkingDistance">
                        <xsl:value-of select="WorkingDistance"/>
                    </xsl:attribute>
                    <xsl:attribute name="WorkingDistanceUnit">
                        <xsl:text>µm</xsl:text>
                    </xsl:attribute>
                </xsl:if>
                <xsl:if test="Iris">
                    <xsl:attribute name="Iris">
                        <xsl:value-of select="Iris"/>
                    </xsl:attribute>
                </xsl:if>
            </xsl:element>
        </xsl:for-each>
    </xsl:template>



    <!-- Instrument -->
    <!-- zisraw/Instrument.xsd: 45 -->
    <!-- ome/ome.xsd: 1235 -->
    <xsl:template match="Instrument">
        <xsl:element name="ome:Instrument">
            <xsl:attribute name="ID">
                <xsl:text>Instrument:0</xsl:text>
            </xsl:attribute>
            <xsl:apply-templates select="@Id"/>
            <xsl:apply-templates select="@Name"/>

            <!-- Plural pulled from ome/ome.xsd: 2042 -->
            <xsl:apply-templates select="Microscopes"/>
            <xsl:apply-templates select="LightSources"/>
            <xsl:apply-templates select="Detectors"/>
            <xsl:apply-templates select="Objectives"/>
        </xsl:element>
    </xsl:template>

</xsl:stylesheet>
