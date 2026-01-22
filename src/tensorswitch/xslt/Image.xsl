<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.1"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <xsl:import href="ImagingEnvironment.xsl"/>
    <xsl:import href="ObjectiveSettings.xsl"/>
    <xsl:import href="Pixels.xsl"/>

    <xsl:variable name="img" select="/ImageDocument/Metadata/Information/Image"/>

    <xsl:template name="common_image_contents">
        <xsl:param name="idx"/>
        <xsl:choose>
            <!-- /Subblocks/Subblock/METADATA/Tags/AcquisitionTime => /OME/Image/@AcquisitionDate   -->
            <xsl:when test="$subblocks/Subblock[@S=$idx]/METADATA/Tags/AcquisitionTime">
                <xsl:element name="ome:AcquisitionDate">
                    <xsl:value-of select="$subblocks/Subblock[@S=$idx]/METADATA/Tags/AcquisitionTime"/>
                </xsl:element>
            </xsl:when>
            <!-- /Metadata/Information/Image/AcquisitionDataAndTime => /OME/Image/@AcquisitionDate   -->
            <xsl:when test="$img/AcquisitionDateAndTime">
                <xsl:element name="ome:AcquisitionDate">
                    <xsl:value-of select="$img/AcquisitionDateAndTime"/>
                </xsl:element>
            </xsl:when>
        </xsl:choose>
        <!-- ObjectiveSettings -->
        <xsl:apply-templates select="$img/ObjectiveSettings"/>
        <!-- Imaging Environment -->
        <xsl:apply-templates select="/ImageDocument/Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/IncubationRecording"/>
    </xsl:template>

    <!-- For single scene images, manually create top level attributes of the /OME/Image/Image element -->
    <xsl:template name="single_scene_image">
        <xsl:element name="ome:Image">
            <xsl:attribute name="ID">Image:0</xsl:attribute>
            <xsl:call-template name="common_image_contents">
                <xsl:with-param name="idx">0</xsl:with-param>
            </xsl:call-template>
            <!--   Pixels  -->
            <xsl:apply-templates select="$img">
                <xsl:with-param name="idx">0</xsl:with-param>
            </xsl:apply-templates>
        </xsl:element>
    </xsl:template>

    <!-- /ImageDocument/Metadata/Information/Image/Dimensions/S/Scenes/Scene => /OME/Image/Image -->
    <xsl:template match="Scene">
        <xsl:variable name="idx" select="@Index"/>
        <xsl:element name="ome:Image">
            <xsl:attribute name="ID">
                <xsl:text>Image:</xsl:text>
                <xsl:value-of select="$idx"/>
            </xsl:attribute>
            <xsl:attribute name="Name">
                <xsl:value-of select="@Name"/>
            </xsl:attribute>
            <xsl:call-template name="common_image_contents">
                <xsl:with-param name="idx" select="$idx"/>
            </xsl:call-template>
            <!--   Pixels  -->
            <xsl:apply-templates select="$img">
                <xsl:with-param name="idx" select="$idx"/>
            </xsl:apply-templates>
        </xsl:element>
    </xsl:template>

</xsl:stylesheet>
