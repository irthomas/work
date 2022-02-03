# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:02:31 2021

@author: iant

XML 
"""

import logging
import os.path
import re
import numpy as np
from lxml import etree
from datetime import datetime, timedelta

import os
#import re
import subprocess



from nomad_ops.core.psa.l1p0a_to_psa.config import \
    HDF5_TIME_FORMAT, ASCII_DATE_TIME_YMD_UTC, PSA_MODIFICATION_DATE, \
    PSA_VERSION, TITLE, INFORMATION_MODEL_VERSION, PSA_VERSION_DESCRIPTION




"""XML parameters"""
MODEL_HREF = ["\"https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1F00.sch\"", \
             "\"https://psa.esa.int/psa/v1/PDS4_PSA_1F00_1300.sch\"", \
             "\"https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1F00_1910.sch\"", \
             "\"https://psa.esa.int/psa/em16/v1/PDS4_EM16_1F00_1200.sch\"", \
             "\"https://psa.esa.int/psa/em16/tgo/nmd/v1/PDS4_EM16_TGO_NMD_1F00_1000.sch\""]
MODEL_SCHEMATRON = "\"http://purl.oclc.org/dsdl/schematron\""


SCHEMA_LOCATION = \
"http://pds.nasa.gov/pds4/pds/v1 \
https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1F00.xsd \
http://psa.esa.int/psa/v1 \
https://psa.esa.int/psa/v1/PDS4_PSA_1F00_1300.xsd \
http://pds.nasa.gov/pds4/geom/v1 \
https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1F00_1910.xsd \
http://psa.esa.int/psa/em16/tgo/nmd/v1 \
https://psa.esa.int/psa/em16/tgo/nmd/v1/PDS4_EM16_TGO_NMD_1F00_1000.xsd"

#http://psa.esa.int/psa/em16/v1 \ #no matching xmlns
#http://psa.esa.int/psa/em16/v1/PDS4_EM16_1F00_1200.xsd \ #no matching xmlns

XMLNS = "http://pds.nasa.gov/pds4/pds/v1"
XMLNS_GEOM = "http://pds.nasa.gov/pds4/geom/v1"
XMLNS_PSA = "http://psa.esa.int/psa/v1"
XMLNS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
XMLNS_EM16_TGO_NMD = "http://psa.esa.int/psa/em16/tgo/nmd/v1"


PSA_LOGICAL_IDENTIFIER_PREFIX = "urn:esa:psa:em16_tgo_nmd:data_calibrated:"
PSA_PAR_LOGICAL_IDENTIFIER_PREFIX = "urn:esa:psa:em16_tgo_nmd:data_partially_processed:"
BROWSE_LOGICAL_IDENTIFIER_PREFIX = "urn:esa:psa:em16_tgo_nmd:browse_calibrated:"
AOTF_FUNCTION_LOGICAL_IDENTIFIER_PREFIX = "urn:esa:psa:em16_tgo_nmd:calibration:"




def makeBrowseXmlLabel(paths):
    """return browse xml label string for writing to file elsewhere (e.g. for adding to zip file)"""
    
 
    element = etree.Element("{" + XMLNS + "}Product_Browse", attrib={"{" + XMLNS_XSI + "}schemaLocation" : SCHEMA_LOCATION}, \
            nsmap={None:XMLNS, "geom":XMLNS_GEOM, "psa":XMLNS_PSA, "xsi":XMLNS_XSI, "em16_tgo_nmd":XMLNS_EM16_TGO_NMD})
    psaCalDoc = etree.ElementTree(element)
    
    for MODEL in MODEL_HREF:
        psaCalDoc.getroot().addprevious(etree.ProcessingInstruction("xml-model", "href=%s schematypens=%s" %(MODEL,MODEL_SCHEMATRON)))

    subElementA = etree.SubElement(element, "Identification_Area")
    subElementA0 = etree.SubElement(subElementA, "logical_identifier")
    subElementA0.text = paths["brow_lid_full"]
    subElementA1 = etree.SubElement(subElementA, "version_id")
    subElementA1.text = PSA_VERSION
    subElementA2 = etree.SubElement(subElementA, "title")
    subElementA2.text = TITLE
    subElementA3 = etree.SubElement(subElementA, "information_model_version")
    subElementA3.text = INFORMATION_MODEL_VERSION
    subElementA4 = etree.SubElement(subElementA, "product_class")
    subElementA4.text = "Product_Browse"

    subElementA5 = etree.SubElement(subElementA, "Modification_History")
    subElementA5A = etree.SubElement(subElementA5, "Modification_Detail")
    subElementA5A0 = etree.SubElement(subElementA5A, "modification_date")
    subElementA5A0.text = PSA_MODIFICATION_DATE
    subElementA5A1 = etree.SubElement(subElementA5A, "version_id")
    subElementA5A1.text = PSA_VERSION
    subElementA5A2 = etree.SubElement(subElementA5A, "description")
    subElementA5A2.text = PSA_VERSION_DESCRIPTION
    

    subElementB = etree.SubElement(element, "Reference_List")
    subElementB0 = etree.SubElement(subElementB, "Internal_Reference")
    subElementB0A = etree.SubElement(subElementB0, "lid_reference")
    subElementB0A.text = paths["data_lid_full"]
    subElementB0B = etree.SubElement(subElementB0, "reference_type")
    subElementB0B.text = "browse_to_data"

    subElementC = etree.SubElement(element, "File_Area_Browse")
    subElementC0 = etree.SubElement(subElementC, "File")
    subElementC0A = etree.SubElement(subElementC0, "file_name")
    subElementC0A.text = paths["data_xml_filename"]
    subElementC0B = etree.SubElement(subElementC0, "local_identifier")
    subElementC0B.text = paths["data_xml_filename"] #must not start with a number - just use the filename here
    subElementC0C = etree.SubElement(subElementC0, "creation_date_time")
    subElementC0C.text = datetime.strftime(datetime.now(),ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z"

    subElementC1 = etree.SubElement(subElementC, "Encoded_Image")
    subElementC1A = etree.SubElement(subElementC1, "offset")
    subElementC1A.set("unit", "byte")
    subElementC1A.text = "0"
    subElementC1B = etree.SubElement(subElementC1, "encoding_standard_id")
    subElementC1B.text = "PNG"
        
    browseXmlFileOut = etree.tostring(psaCalDoc, xml_declaration=True, pretty_print=True, encoding="utf-8").decode()


    return browseXmlFileOut


