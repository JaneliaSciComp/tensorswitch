<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.1"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06">

    <!-- Stores the Acquisition Times from subblocks in a TimestampAnnotation element -->
    <xsl:template match="Subblocks" mode="structured_annotations">
        <xsl:for-each select="Subblock">
            <xsl:element name="ome:TimestampAnnotation">
                <xsl:attribute name="ID">
                    <xsl:text>Annotation:SubblockAcquisition</xsl:text>
                    <!-- Add the name and value of each attribute to the ID of this element -->
                    <xsl:for-each select="@*">
                        <xsl:value-of select="name(.)"/>
                        <xsl:value-of select="."/>
                    </xsl:for-each>
                </xsl:attribute>
                <xsl:element name="ome:Description">
                    <xsl:text>Acquistion Time from subblock </xsl:text>
                    <xsl:for-each select="@*">
                        <xsl:value-of select="name(.)"/>
                        <xsl:value-of select="."/>
                    </xsl:for-each>
                </xsl:element>
                <xsl:element name="ome:Value">
                    <xsl:value-of select="METADATA/Tags/AcquisitionTime"/>
                </xsl:element>
            </xsl:element>
        </xsl:for-each>
    </xsl:template>

    <xsl:template match="Application">
        <xsl:element name="ome:StructuredAnnotations">
            <xsl:element name="ome:XMLAnnotation">
                <xsl:attribute name="ID">
                    <xsl:text>urn:lsid:allencell.org:Annotation:AcquisitionSoftware</xsl:text>
                </xsl:attribute>
                <xsl:element name="ome:Description">
                    <xsl:value-of select="Name"/>
                    <xsl:value-of select="Version"/>
                </xsl:element>
                <xsl:element name="ome:Value">
                </xsl:element>
            </xsl:element>
            <xsl:apply-templates select="$subblocks" mode="structured_annotations"/>
        </xsl:element>
    </xsl:template>

</xsl:stylesheet>
