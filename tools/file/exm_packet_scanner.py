# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 20:52:30 2026

@author: iant

SCAN .EXM FILES FOR CHECKSUM ERRORS


SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM
integrationtime	modei	acquisitionmode	vstart	vend	hstart	hend	binningsize
75	SO	Hor_Bin	135	205	0	1047	7 = 71 rows in total, 131 columns x 15 rows per packet x 2.5 = 1965 * 2.5 = 4912.5

CORRUPTED CHECKSUM DUE TO GAPS IN DATA + FOLLOWING PACKET COUNT 955 NOT DELIVERED
E.G. GOOD DATA IN 5 LINES FOLLOWED BY UNPREDICTABLE GAPS

11001001 01011000 0000 11001001 01101100 0000
11001001 10101000 0000 11001001 10010000 0000
11000111 11011100 0000 11001001 00101100 0000
11001001 11011000 0000 11001001 01011000 0000
11001010 00010000 0000 11001000 10011000 0000
           111100 0000 11001001 01111000 0000
11001001 111   00 0000 11001000 00111100 0000
11001        0000 0000 11001001 00100000 0000
11001001 01000000 0000 1             000 0000
11001010 01001100 0000 11001010 0000

"""
from tools.general.progress_bar import progress
import glob
import os
import numpy as np
import struct
import matplotlib.pyplot as plt


def bytes_to_string(byte_string):
    return ''.join('\\x{:02x}'.format(letter) for letter in byte_string)


def scan_exm_packets(file_path):
    """read file, make list of packet dictionaries"""
    with open(file_path, "rb") as f:
        hex_in = f.read()

    tms = []
    i = 0
    errors = []
    while i < len(hex_in):
        tm = {}

        i_start = i

        il = 1
        tm["type"] = int.from_bytes(hex_in[i:i+il], "big")
        i += il

        il = 3
        tm["size"] = int.from_bytes(hex_in[i:i+il], "big")
        i += il

        if tm["type"] in [37, 60]:
            il = tm["size"] - 6
            tm["data"] = hex_in[i:i+il]
            i += il

        else:

            il = 8
            tm["ts"] = int.from_bytes(hex_in[i:i+il], "big")
            i += il

            il = 2
            tm["count"] = int.from_bytes(hex_in[i:i+il], "big")
            i += il

            if tm["type"] in [22, 25]:
                il = tm["size"] - 16
                tm["data"] = hex_in[i:i+il]
                i += il

            if tm["type"] == 27:
                il = 32
                tm["cops"] = hex_in[i:i+il]
                i += il
            elif tm["type"] == 28:
                il = 32
                tm["cops"] = hex_in[i:i+il]
                i += il

                il = 42
                tm["hsk"] = hex_in[i:i+il]
                i += il

                il = tm["size"] - 90
                tm["data"] = hex_in[i:i+il]
                i += il
            elif tm["type"] == 29:
                il = 42
                tm["hsk"] = hex_in[i:i+il]
                i += il

        tm["cs_calc"] = sum([x for x in hex_in[i_start:i]]) & 0xffff

        il = 2
        tm["cs"] = int.from_bytes(hex_in[i:i+il], "big")
        i += il

        if tm["type"] not in [22, 25, 27, 28, 29, 37, 60]:
            print("Error unknown TM type:", tm)
            errors.append("Error unknown TM type %s" % tm)

        if tm["cs_calc"] != tm["cs"]:
            print("Error: checksum wrong in packet", len(tms))
            errors.append(len(tms))

        tms.append(tm)
    return tms, errors


def reconstruct_frames(tms, filename):
    """code to analyse bytes and reconstruct detector frames"""

    # TODO: calculate these from the data itself
    if filename == "SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM":
        BYTES_PER_VALUE = 2.5
        N_COLS = 131
        N_ROWS = 71
        packet_range = range(1867, 1872)

    if filename == "SCI__DNMD__030D5601_2018-241T12-30-52__00001.EXM":
        BYTES_PER_VALUE = 2
        N_COLS = 1048
        N_ROWS = 15*12+14
        packet_range = range(1849, 1860)

    big_arr = np.zeros((N_ROWS, N_COLS), dtype=int) + np.nan
    for i in packet_range:
        if tms[i]["type"] == 28:
            print(i, tms[i]["type"], tms[i]["size"], tms[i]["count"], len(tms[i]["data"]), tms[i]["cs_calc"] == tms[i]["cs"])

    #         # struct.unpack(tms[i]["data"])

            byte_string = tms[i]["data"]
            nbytes = len(byte_string)  # eg 4913 bytes
            nvalues = int(np.floor(nbytes / BYTES_PER_VALUE))  # e.g. 1965 values (odd) = 131 x 15

            odd_number = np.mod(nvalues, 2) == 1

            if odd_number:
                byte_string += b'\x00\x00'  # pad with zeros e.g. 4915 bytes = 1966 values (last zero)

            arr = np.zeros(nvalues, dtype=int)

            niters = int(np.ceil(nvalues / 2))

            # extract 2 values at a time
            for ib in range(niters):
                # get 5 bytes each time

                if BYTES_PER_VALUE == 2.5:
                    bs = ib * 5
                    b5 = int.from_bytes(struct.unpack('>5s', byte_string[bs:bs+5])[0], "big")
                    # if i == 1869 and ib > 15 and ib < 25:
                    # print(f'{b5:<040b}'[0:8], f'{b5:<040b}'[8:16], f'{b5:<040b}'[16:20], f'{b5:<040b}'[20:28], f'{b5:<040b}'[28:36], f'{b5:<040b}'[36:40])
                    # print(ib, bs, bs+5, bytes_to_string(struct.unpack('>5s', byte_string[bs:bs+5])[0]))

                    # extract 2.5 bytes from first half
                    arr[ib*2] = (b5 >> 20) & 0xfffff

                    # extract 2.5 bytes from second half if not very last value (padded with zeros)
                    if ib == int(np.ceil(nvalues / 2)) - 1 and odd_number:
                        continue
                    else:
                        arr[ib*2+1] = b5 & 0xfffff

                elif BYTES_PER_VALUE == 2.0:
                    bs = ib * 4
                    b5 = int.from_bytes(struct.unpack('>4s', byte_string[bs:bs+4])[0], "big")
                    # if i == 1869 and ib > 15 and ib < 25:
                    # print(f'{b5:<040b}'[0:8], f'{b5:<040b}'[8:16], f'{b5:<040b}'[16:20], f'{b5:<040b}'[20:28], f'{b5:<040b}'[28:36], f'{b5:<040b}'[36:40])
                    # print(ib, bs, bs+5, bytes_to_string(struct.unpack('>5s', byte_string[bs:bs+5])[0]))

                    # extract 2.5 bytes from first half
                    arr[ib*2] = (b5 >> 16) & 0xffff

                    # extract 2.5 bytes from second half if not very last value (padded with zeros)
                    if ib == int(np.ceil(nvalues / 2)) - 1 and odd_number:
                        continue
                    else:
                        arr[ib*2+1] = b5 & 0xffff
            # if
            print(len(arr))
            if np.mod(len(arr), N_COLS) > 0:
                arr = np.concatenate((arr, np.zeros(N_COLS - np.mod(len(arr), N_COLS), dtype=int)))
            arr2 = np.reshape(arr, (-1, N_COLS))

            # for checksum error in SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM
            if filename == "SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM":
                # previous good frame
                # if i == 1852:
                #     big_arr[0:15, :] = arr2
                # if i == 1853:
                #     big_arr[15:30, :] = arr2
                # if i == 1854:
                #     big_arr[30:45, :] = arr2
                # if i == 1855:
                #     big_arr[45:60, :] = arr2
                # if i == 1856:
                #     big_arr[60:72, :] = arr2

                # bad frame
                if i == 1867:
                    big_arr[0:15, :] = arr2
                if i == 1868:
                    big_arr[15:30, :] = arr2
                if i == 1869:
                    big_arr[30:45, :] = arr2
                if i == 1870:
                    big_arr[45:60, :] = arr2
                if i == 1871:
                    big_arr[60:72, :] = arr2

            if filename == "SCI__DNMD__030D5601_2018-241T12-30-52__00001.EXM":
                # bad frame
                if i == 1849:
                    big_arr[0:15, :] = arr2
                if i == 1850:
                    big_arr[15:30, :] = arr2
                if i == 1851:
                    big_arr[30:45, :] = arr2
                if i == 1852:
                    big_arr[45:60, :] = arr2
                if i == 1853:
                    big_arr[60:75, :] = arr2
                if i == 1854:
                    big_arr[75:90, :] = arr2
                if i == 1855:
                    big_arr[90:105, :] = arr2
                if i == 1856:
                    big_arr[105:120, :] = arr2
                if i == 1857:
                    big_arr[120:135, :] = arr2
                if i == 1858:
                    big_arr[135:150, :] = arr2
                if i == 1859:
                    big_arr[150:158, :] = arr2

    plt.figure()
    plt.title("Reconstructed UVIS frame")
    plt.xlabel("Detector column (spectral direction)")
    plt.ylabel("Detector row")
    plt.imshow(big_arr, aspect="auto", vmin=12000, vmax=13500)


def scan_exms(data_path, stop_on_error=True, error_file=""):
    """Scan all .EXM files in a directory and subdirectories
    Optional: stop and return packet list dict if an error is found
    Optional: write list of error files to file"""
    exm_paths = glob.glob(data_path + os.sep + "**" + os.sep + "*.EXM", recursive=True)

    print("Scanning %i files in %s" % (len(exm_paths), data_path))
    for exm_path in progress(exm_paths):
        tms, errors = scan_exm_packets(exm_path)
        if len(errors) > 0:
            exm = os.path.basename(exm_path)
            print(exm)
            if error_file != "":
                with open(error_file, "a") as f:
                    f.write("%s %s\n" % (exm, errors))
            if stop_on_error:
                return tms

    return []


# examples

# analyse one file

# file_path = r"W:/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/spacewire/2020/05/29/SCI__DNMD__032BD701_2020-150T18-09-48__00001.EXM"
# # file_path = r"C:/Users/iant/Documents/DATA/temp/SCI__DNMD__03870F01_2025-235T20-15-39__00002.EXM"  # missing packet 20250823_201540_0p3b_UVIS_D
# file_path = r"C:/Users/iant/Documents/DATA/temp/SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM"  # checksum error 20250822_175340_0p1a_UVIS
# file_path = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/spacewire/2025/08/22/SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM"  # checksum error 20250822_175340_0p1a_UVIS
# file_path = r"W:/projects/NOMAD/Operations/Temp/for_Ian/SCI__DNMD__03870201_2025-234T17-53-39__00001.EXM"  # EDDS checksum error 20250822_175340_0p1a_UVIS
# file_path = r"W:/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/spacewire/2018/08/29/SCI__DNMD__030D5601_2018-241T12-30-52__00001.EXM"

# tms, _ = scan_exm_packets(file_path)
# reconstruct_frames(tms, os.path.basename(file_path))


# scan exms in single directory
# data_path = r"W:\projects\NOMAD\Operations\Temp\for_Ian"
# tms = scan_exms(data_path, stop_on_error=False, error_file="checksum_errors.txt")

# scan all exms in datastore
# months = [str(i).zfill(2) for i in range(1, 13)]
# years = [str(i) for i in range(2018, 2030)]
# for year in years:
#     for month in months:
#         data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\spacewire\%s\%s" % (year, month)
#         data_path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/spacewire/%s/%s" % (year, month)
#         if os.path.exists(data_path):
#             tms = scan_exms(data_path, stop_on_error=False, error_file="checksum_errors.txt")
