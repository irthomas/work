# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:11:39 2023

@author: iant


OLD - SOLAR LINE POSITIONS NOW FOUND AUTOMATICALLY

PROVIDE A DICTIONARY WHICH MANUALLY DEFINES THE POSITION AND EXTENT OF ALL SOLAR LINES TO BE ANALYSED IN EACH FILE

"""


# solar line dict
# arr_region_rows = rows beyond the absorption, to avoid converting too much data (normally 0 to end)
# arr_region_cols = columns beyond the absorption, to avoid converting too much data
# abs_region_rows = rows containing absorption
# abs_region_cols = columns containing absorption
solar_line_dict = {
    # "LNO-20200201-001633-194-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1500, 2500], "abs_region_rows":[300, -1], "abs_region_cols":[1859, 1944], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"blaze_rows":[]}
    # ],
    # "LNO-20181021-064022-125-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[500, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[700, 810], "cutoffs":[0.93, 0.93], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2900, 3199], "abs_region_rows":[0, -1], "abs_region_cols":[2950, 3090], "cutoffs":[0.93, 0.93], "smfac":25},
    #     {"blaze_rows":[580, 970, 1360]}
    # ],
    # "LNO-20181021-064022-140-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[400, 700], "abs_region_rows":[0, -1], "abs_region_cols":[450, 580], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[800, 900], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2100, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2230, 2360], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2390, 2490], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"blaze_rows":[950, 1350, 1710]}
    #     ],


    # "LNO-20181021-071529-167-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[850, 1300], "abs_region_rows":[0, 1700], "abs_region_cols":[990, 1250], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1400, 1700], "abs_region_rows":[0, 1700], "abs_region_cols":[1520, 1600], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2700], "abs_region_rows":[0, -1], "abs_region_cols":[2420, 2540], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[850]}
    #     ],

    # "LNO-20181021-071529-194-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1800, 2100], "abs_region_rows":[0, -1], "abs_region_cols":[1910, 2000], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[]}
    #     ],

    # "LNO-20181106-192332-146-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[150, 350], "abs_region_rows":[0, -1], "abs_region_cols":[220, 280], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1050, 1350], "abs_region_rows":[0, -1], "abs_region_cols":[1110, 1200], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2410, 2500], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2800, 3100], "abs_region_rows":[0, -1], "abs_region_cols":[2910, 2980], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[150, 1310]}
    #     ],

    # "LNO-20181106-192332-161-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[100, 400], "abs_region_rows":[0, 1600], "abs_region_cols":[210, 280], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[300, 550], "abs_region_rows":[0, -1], "abs_region_cols":[380, 470], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[840, 930], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2000, 2200], "abs_region_rows":[0, -1], "abs_region_cols":[2040, 2140], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2200, 2400], "abs_region_rows":[0, -1], "abs_region_cols":[2260, 2340], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2340, 2420], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2520, 2850], "abs_region_rows":[0, 1300], "abs_region_cols":[2530, 2610], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"blaze_rows":[1250]}
    #     ],


    # "LNO-20181106-195839-170-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181106-195839-197-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-172841-122-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-172841-137-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-180348-158-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-180348-191-4": [
    #     {"arr_region_rows": [0, -1], "arr_region_cols":[2000, 2300], "abs_region_rows":[0, -1],
    #         "abs_region_cols":[2120, 2250], "cutoffs":[0.97, 0.97], "smfac":25},
    # ],
    # "LNO-20190212-142357-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-142357-140-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-145904-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-145904-194-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190408-040458-194-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190609-015021-140-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190621-234127-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190622-001634-140-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190728-012702-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190728-012702-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190809-101441-146-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190809-101441-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20191002-000902-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20191002-000902-182-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],

    # "LNO-20200201-001633-200-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200207-212317-113-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200207-212317-116-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200624-113644-146-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200624-113644-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200628-135310-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200628-135310-164-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200812-135659-170-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200812-135659-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200827-133646-182-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200827-133646-194-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20201018-141050-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20201018-141050-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210208-022300-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210208-022300-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210306-015642-185-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210306-015642-188-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210402-123335-143-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210402-123335-173-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210606-021551-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210606-021551-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220323-231521-116-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220323-231521-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220325-123611-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220325-123611-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220417-151406-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220417-151406-164-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220418-030236-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220418-030236-170-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220619-140101-164-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220619-140101-176-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220624-020349-179-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220624-020349-182-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221030-012837-128-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221030-012837-149-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221031-125107-152-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221031-125107-155-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],

    # "SO-20180716-000706-178-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[670, 770], "cutoffs":[0.97, 0.97], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1210, 1290], "cutoffs":[0.97, 0.97], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1310, 1390], "cutoffs":[0.97, 0.97], "smfac":25, "good":True},
    #         # {"blaze_rows":[1100, 1500]}
    #         ],

    # "SO-20181010-084333-184-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1330, 1430], "cutoffs":[0.97, 0.97], "smfac":25, "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1440, 1520], "cutoffs":[0.97, 0.97], "smfac":25, "good":True},
    #         # {"blaze_rows":[1500]}
    #         ],

    # "SO-20181114-084251-186-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[940, 1020], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1330, 1410], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1440, 1530], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2530, 2630], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2680, 2780], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2940, 3040], "cutoffs":[0.97, 0.97], "smfac":25, "good":True},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20181206-171850-181-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[160, 270], "cutoffs":[0.92, 0.92], "smfac":25, "good":True},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2220, 2350], "cutoffs":[0.95, 0.95], "smfac":25, "good":True},
    #         # {"blaze_rows":[390]}
    #         ],

    # "SO-20190416-024455-184-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1240, 1340], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1360, 1430], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1910, 2020], "cutoffs":[0.97, 0.97], "smfac":25},
    #         # {"blaze_rows":[1500]}
    #         ],

    # "SO-20210226-085144-178-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[550, 650], "cutoffs":[0.95, 0.95], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1100, 1190], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2130, 2230], "cutoffs":[0.95, 0.95], "smfac":25},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20211105-155547-178-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[570, 670], "cutoffs":[0.95, 0.95], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1110, 1190], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2150, 2250], "cutoffs":[0.95, 0.95], "smfac":25},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20211105-155547-181-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[90, 190], "cutoffs":[0.92, 0.92], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1380, 1450], "cutoffs":[0.96, 0.96], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2160, 2270], "cutoffs":[0.95, 0.95], "smfac":25},
    #         # {"blaze_rows":[390]}
    #         ],

    # "SO-20230112-084925-178-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[505, 605], "cutoffs":[0.95, 0.95], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1035, 1135], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2080, 2180], "cutoffs":[0.95, 0.95], "smfac":25},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20230112-084925-181-4":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[40, 120], "cutoffs":[0.9, 0.9], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1320, 1390], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2080, 2220], "cutoffs":[0.95, 0.95], "smfac":25},
    # #         # {"blaze_rows":[360]}
    #         ],

    # "SO-20181129-002850-184-2":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1240, 1340], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1360, 1430], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1910, 2020], "cutoffs":[0.98, 0.98], "smfac":25},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20210912-022732-178-2":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[670, 770], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1210, 1290], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1310, 1390], "cutoffs":[0.97, 0.97], "smfac":25},
    #         ],

    # "SO-20190107-015635-184-8":[
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[540, 640], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[680, 780], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[870, 970], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[920, 1020], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1310, 1410], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1420, 1520], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1780, 1880], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2520, 2620], "cutoffs":[0.96, 0.96]},
    #         {"abs_region_rows":[0, 1800], "abs_region_cols":[2670, 2770], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2850, 2950], "cutoffs":[0.95, 0.95], "good":False},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20190307-011600-184-8":[
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[500, 600], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[640, 740], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[830, 930], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[880, 980], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1270, 1370], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1380, 1480], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1740, 1840], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2480, 2580], "cutoffs":[0.96, 0.96]},
    #         {"abs_region_rows":[0, 1800], "abs_region_cols":[2630, 2730], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2810, 2910], "cutoffs":[0.95, 0.95], "good":False},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20201010-113533-178-8":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[85, 185], "cutoffs":[0.92, 0.92]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[580, 680], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[870, 970], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1110, 1210], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1210, 1310], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1930, 2030], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2160, 2260], "cutoffs":[0.95, 0.95]},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20210201-111011-178-8":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[45, 145], "cutoffs":[0.92, 0.92]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[540, 640], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[830, 930], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1070, 1170], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1170, 1270], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1890, 1990], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2120, 2220], "cutoffs":[0.95, 0.95]},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20210523-001053-178-8":[
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[45, 145], "cutoffs":[0.92, 0.92]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[540, 640], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[830, 930], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1070, 1170], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1170, 1270], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1890, 1990], "cutoffs":[0.97, 0.97], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2120, 2220], "cutoffs":[0.95, 0.95]},
    #         # {"blaze_rows":[]}
    #         ],

    # "SO-20221011-132104-178-8":[
    # {"abs_region_rows":[0, -1], "abs_region_cols":[45, 145], "cutoffs":[0.92, 0.92]},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[540, 640], "cutoffs":[0.95, 0.95]},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[830, 930], "cutoffs":[0.95, 0.95], "good":False},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[1070, 1170], "cutoffs":[0.97, 0.97]},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[1170, 1270], "cutoffs":[0.97, 0.97], "good":False},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[1890, 1990], "cutoffs":[0.97, 0.97], "good":False},
    # {"abs_region_rows":[0, -1], "abs_region_cols":[2120, 2220], "cutoffs":[0.95, 0.95]},
    #         # {"blaze_rows":[]}
    #         ],
    # "SO-20221011-132104-184-8":[
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[500, 600], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[640, 740], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[830, 930], "cutoffs":[0.95, 0.95], "good":False},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[880, 980], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1270, 1370], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[500, -1], "abs_region_cols":[1380, 1480], "cutoffs":[0.95, 0.95]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[1740, 1840], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2480, 2580], "cutoffs":[0.96, 0.96]},
    #         {"abs_region_rows":[0, 1800], "abs_region_cols":[2630, 2730], "cutoffs":[0.97, 0.97]},
    #         {"abs_region_rows":[0, -1], "abs_region_cols":[2810, 2910], "cutoffs":[0.95, 0.95], "good":False},
    #         # {"blaze_rows":[]}
    #         ],


}


# # code to make the dictionary above
# channel = "so"
# # channel = "lno"
# for f in [f for f in os.listdir(os.path.join(MINISCAN_PATH, channel)) if "-8.h5" in f]:
#     s = '''"%s":[
#         {"abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
#         {"blaze_rows":[]}
#         ],''' %(os.path.splitext(f)[0])

#     print(s)
# # stop()
