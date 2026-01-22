<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.1"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <xsl:include href="Instrument.xsl"/>

    <!-- /Metadata/Information/Document/UserName => /OME/Experimenter@UserName -->
    <xsl:template match="UserName">
        <xsl:attribute name="ID">
            <xsl:text>Experimenter:</xsl:text><xsl:value-of select="translate(.,' ','_')"/>
        </xsl:attribute>
        <xsl:attribute name="UserName">
            <xsl:value-of select="translate(.,' ','_')"/>
        </xsl:attribute>
    </xsl:template>

    <xsl:template match="Document">
        <xsl:element name="ome:Experimenter">
            <xsl:apply-templates select="UserName"/>
        </xsl:element>
    </xsl:template>


    <xsl:template match="Information">
        <xsl:apply-templates select="Document"/>
        <xsl:apply-templates select="Instrument"/>
    </xsl:template>

</xsl:stylesheet>
