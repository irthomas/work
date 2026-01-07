# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:52:46 2023

@author: iant
"""


# from psa_utils import download, tap, packager, geogen, internal

from psa_utils import tap
psa = tap.PsaTap(tap_url="https://archives.esac.esa.int/psa-tap/tap/")


# Now for some examples:
# nomad = psa.query("select top 1 * from epn_core where instrument_name='NOMAD' and granule_uid like '%cal%' order by time_min desc")
# print(nomad.iloc[0].granule_uid)


# date_string='20200417'
# # Raw v3.0
# counter = psa.query("select count(*) from epn_core where instrument_name='NOMAD' and granule_uid like '%%{:s}%%' and granule_uid like '%:3.0' and granule_uid like '%raw_hk_hk1_%'".format(date_string))
# print("Raw HK1, ",counter.COUNT_ALL[0])

# counter = psa.query("select count(*) from epn_core where instrument_name='NOMAD' and granule_uid like '%%{:s}%%' and granule_uid like '%:3.0' and granule_uid like '%raw_hk_hk2_%'".format(date_string))
# print("Raw HK2, ",counter.COUNT_ALL[0])

# counter = psa.query("select count(*) from epn_core where instrument_name='NOMAD' and granule_uid like '%%{:s}%%' and granule_uid like '%:3.0' and granule_uid like '%raw_sc_sinbad%'".format(date_string))
# print("Raw SINBAD,",counter.COUNT_ALL[0])


# count number of products available in the archive for a given channel and version number
total = 0
for channel in ["LNO", "SO", "UVIS"]:
    # for version_id in ["1.0", "2.0", "3.0", "3.1", "4.0"]:
    for version_id in ["4.0"]:
        query = "SELECT COUNT(*) FROM psa.product_ui_em16_nomad WHERE \
            bundle_distribution_path = 'em16_tgo_nmd' AND \
            processing_level = 'Calibrated' \
            AND subinstrument_name = '%s' AND version_id = '%s'" % (channel, version_id)
        counter = psa.query(query)
        print(channel, version_id, counter.COUNT_ALL[0])
        total += counter.COUNT_ALL[0]

print(total)

# channel = "SO"
# lid = "nmd_cal_sc_so_20180524t02%%-121"
# version_id = "2.0"
# query = "SELECT TOP 100 * FROM psa.v_product_ui_em16 WHERE bundle_distribution_path = 'em16_tgo_nmd' AND processing_level = 'Calibrated' AND subinstrument_name = '%s' AND logical_identifier_short LIKE '%s' AND version_id = '%s'" %(channel, lid, version_id)

# results_dict = psa.query(query).to_dict("list")

# # gets lids
# lids = sorted(results_dict["logical_identifier_short"])
# for lid in lids:
#     print(lid)
#
# download.download_file(tap_url+results_dict["download_path"][0])
