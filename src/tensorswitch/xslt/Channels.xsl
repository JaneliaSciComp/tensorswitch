<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.1"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <!-- /Metadata/DisplaySetting/Channels/Channel/IlluminationType => /OME/Image/Channel/IlluminationType -->
    <xsl:template match="IlluminationType">
        <xsl:if test="string-length(.) &gt; 0">
            <xsl:attribute name="IlluminationType">
                <xsl:value-of select="."/>
            </xsl:attribute>
        </xsl:if>
    </xsl:template>


    <xsl:template match="ExcitationWavelength">
        <xsl:if test="string-length(.) &gt; 0">
            <xsl:attribute name="ExcitationWavelength">
                <xsl:value-of select="."/>
            </xsl:attribute>
            <xsl:attribute name="ExcitationWavelengthUnit">
                <xsl:text>nm</xsl:text>
            </xsl:attribute>
        </xsl:if>
    </xsl:template>

    <xsl:template match="EmissionWavelength">
        <xsl:if test="string-length(.) &gt; 0">
            <xsl:attribute name="EmissionWavelength">
                <xsl:value-of select="."/>
            </xsl:attribute>
            <xsl:attribute name="EmissionWavelengthUnit">
                <xsl:text>nm</xsl:text>
            </xsl:attribute>
        </xsl:if>
    </xsl:template>

    <!-- TODO: The OME spec only allows for one light source per channel,
    but CZIs can have multiple. We have not yet determined how to handle
    this. -->
    <!--    <xsl:template match="LightSourceSettings">-->
    <!--        <xsl:element name="ome:LightSourceSettings">-->
    <!--            <xsl:attribute name="ID">-->
    <!--                <xsl:value-of select="LightSource/@Id"/>-->
    <!--            </xsl:attribute>-->
    <!--        </xsl:element>-->
    <!--    </xsl:template>-->


    <xsl:template match="DetectorSettings">
        <xsl:element name="ome:DetectorSettings">
            <xsl:attribute name="ID">
                <!-- Convert any whitespace to '_' -->
                <xsl:value-of select="translate(normalize-space(Detector/@Id), ' ', '_')"/>
            </xsl:attribute>
            <xsl:attribute name="Binning">
                <xsl:choose>
                    <!-- OME only supports the following values for binning -->
                    <xsl:when test="Binning = '1,1' or Binning = '2,2' or Binning = '4,4' or Binning = '8,8'">
                        <!-- Binning values appear as "2,2" in CZI, but should be formatted as "2x2" in OME -->
                        <xsl:value-of select="translate(Binning, ',', 'x')"/>
                    </xsl:when>
                    <xsl:otherwise>
                        <xsl:text>Other</xsl:text>
                    </xsl:otherwise>
                </xsl:choose>
            </xsl:attribute>
        </xsl:element>
    </xsl:template>

    <!-- /Metadata/DisplaySetting/Channels/Channel => /OME/Image/Channel -->
    <!-- /Metadata/DisplaySetting/Channels/Channel/DyeName => /OME/Image/Channel@Name -->
    <!-- /Metadata/DisplaySetting/Channels/Channel/DyeName => /OME/Image/Channel@Fluor (if IlluminationType = 'Fluorescence') -->
    <!-- /ImageDocument/Metadata/Information/Image/Dimensions/Channels/Channel/ExcitationWavelength => /Ome/Image/Channel@ExcitationWavelength  -->
    <!-- /ImageDocument/Metadata/Information/Image/Dimensions/Channels/Channel/EmissionWavelength => /Ome/Image/Channel@EmissionWavelength  -->
    <xsl:template match="Channel">
        <xsl:param name="idx"/>
        <xsl:element name="ome:Channel">
            <xsl:attribute name="ID">
                <!-- Prepend 'Channel:' if it isn't contained within the ID in order to satisfy OME spec -->
                <xsl:variable name="channel_id">
                    <xsl:value-of select="@Id"/>
                </xsl:variable>
                <xsl:if test="not(contains($channel_id,'Channel:'))">
                    <xsl:text>Channel:</xsl:text>
                </xsl:if>
                <xsl:value-of select="@Id"/>
                <xsl:text>-</xsl:text>
                <xsl:value-of select="$idx"/>
            </xsl:attribute>
            <xsl:attribute name="Name">
                <xsl:value-of select="@Name"/>
            </xsl:attribute>
            <xsl:if test="AcquisitionMode and string-length(AcquisitionMode) &gt; 0">
                <xsl:attribute name="AcquisitionMode">
                    <xsl:value-of select="AcquisitionMode"/>
                </xsl:attribute>
            </xsl:if>
            <xsl:apply-templates select="IlluminationType"/>
            <xsl:apply-templates select="ExcitationWavelength"/>
            <xsl:apply-templates select="EmissionWavelength"/>
            <xsl:if test="IlluminationType = 'Epifluorescence' and Fluor and string-length(Fluor) &gt; 0">
                <xsl:attribute name="Fluor">
                    <xsl:value-of select="Fluor"/>
                </xsl:attribute>
            </xsl:if>

            <!--<xsl:apply-templates select="LightSourcesSettings/LightSourceSettings"/>-->
            <xsl:apply-templates select="DetectorSettings"/>

        </xsl:element>
    </xsl:template>

    <xsl:template match="Channels">
        <xsl:param name="idx"/>
        <xsl:for-each select="Channel">
            <xsl:apply-templates select=".">
                <xsl:with-param name="idx" select="$idx"/>
            </xsl:apply-templates>
        </xsl:for-each>
    </xsl:template>



</xsl:stylesheet>
