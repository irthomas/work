# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:40:08 2023

@author: iant

TEST ESA PSA EPN TAP

"""

import pyvo

service = pyvo.dal.TAPService("https://archives.esac.esa.int/psa-tap/tap/")

channel = "SO"
lid = "nmd_cal_sc_so_20180524t%%"
version_id = "2.0"
query = "SELECT TOP 100 * FROM psa.product_ui_em16_nomad WHERE bundle_distribution_path = 'em16_tgo_nmd' AND processing_level = 'Calibrated' \
    AND subinstrument_name = '%s' AND logical_identifier_short LIKE '%s' AND version_id = '%s'" % (
    channel, lid, version_id)
results = service.search(query)
results_dict = [dict(t.items()) for t in results]


result_dict = results_dict[0]

print("https://archives.esac.esa.int/psa/pdap/download?RESOURCE_CLASS=PRODUCT&ID=%s" %
      (result_dict["download_path"].replace("/distribution/getProduct?id=", "")))
print(result_dict["postcard_path"])

# gets lids
# lids = sorted([d["logical_identifier_short"] for d in results_dict])
# for lid in lids:
#     print(lid)

# compare results with same lid

# for key in results_dict[46].keys():
#     if results_dict[46][key] != results_dict[78][key]:
#         print(key, results_dict[46][key], results_dict[78][key])


# https://archives.esac.esa.int/psa/pdap/metadata?RETURN_TYPE=VOTABLE&RESOURCE_CLASS=DATA_SET


# SO files available
# https://archives.esac.esa.int/psa/pdap/download?RESOURCE_CLASS=PRODUCT&ID=urn:esa:psa:em16_tgo_nmd:data_calibrated:nmd_cal_sc_so_20180524t022525-20180524t024316-h-i-165::1.0
# https://archives.esac.esa.int/psa/pdap/fileaccess?ID=/repo/esa/psa/em16_tgo_nmd/browse_calibrated/2020-06-09/nmd_cal_sc_browse_20180524t022525-20180524t024316-h-i-165-so/1.0/nmd_cal_sc_browse_20180524T022525-20180524T024316-h-i-165-so.png

# LNO files not available
# https://archives.esac.esa.int/psa/pdap/download?RESOURCE_CLASS=PRODUCT&ID=urn:esa:psa:em16_tgo_nmd:data_calibrated:nmd_cal_sc_lno_20190213t120858-20190213t131138-d-190::3.0
# https://archives.esac.esa.int/psa/pdap/fileaccess?ID=/repo/esa/psa/em16_tgo_nmd/browse_calibrated/2022-10-11/nmd_cal_sc_browse_20190213t120858-20190213t131138-d-190-lno/3.0/nmd_cal_sc_browse_20190213T120858-20190213T131138-d-190-lno.png


# service = pyvo.dal.TAPService("http://vespa-ae.oma.be/tap")
# query = "SELECT * FROM nomad.epn_core"
# results = service.search(query)
# results_dict = [dict(t.items()) for t in results]
