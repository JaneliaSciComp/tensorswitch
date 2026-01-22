<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.1" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <!-- /Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/IncubationRecording/Components/MTBIncubationCO2Channel/Value
        =>
     /OME/Image/ImagingEnvironment@CO2Percent
    -->
    <xsl:template match="MTBIncubationCO2Channel">
        <xsl:attribute name="CO2Percent">
            <xsl:choose>
                <xsl:when test="Value > 1.0">
                    <xsl:value-of select="Value div 100.0"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="Value"/>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

    <!-- /Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/IncubationRecording/Components/MTBIncubationTemperatureChannel4/Value
        =>
     /OME/Image/ImagingEnvironment@UnitTemperature
    -->
    <xsl:template match="MTBIncubationTemperatureChannel4">
        <xsl:attribute name="Temperature">
            <xsl:value-of select="Value"/>
        </xsl:attribute>
        <xsl:attribute name="TemperatureUnit">
            <xsl:text>°C</xsl:text>
        </xsl:attribute>
    </xsl:template>

    <!-- /Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/IncubationRecording
        =>
     /OME/Image/ImagingEnvironment
    -->
    <xsl:template match="IncubationRecording">
        <xsl:if test="Components/MTBIncubationCO2Channel/IsEnabled = 'true'">
            <xsl:element name="ome:ImagingEnvironment">
                <xsl:apply-templates select="Components/MTBIncubationCO2Channel"/>
                <xsl:apply-templates select="Components/MTBIncubationTemperatureChannel4"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

</xsl:stylesheet>
