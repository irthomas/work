    

import os
import struct
import sqlite3
import logging
import itertools
from datetime import datetime, timezone
from tools.file.paths import paths, SYSTEM

if SYSTEM == "Linux":
    from nomad_ops.core.storage.edds import EDDS_Trees
    import nomad_ops.core.edds.edds_files as edds_files
    from nomad_ops.core.raw.bira_raw_file import BIRARawPacket
    import nomad_ops.core.instrument.calibration as calib
    from nomad_ops.config import PATH_EXPORT_HEATERS_TEMP


else:
    PATH_EXPORT_HEATERS_TEMP = os.path.join(paths['LOCAL_DIRECTORY'], "db")


DB_PATH = os.path.join(PATH_EXPORT_HEATERS_TEMP, "heaters_temp.db")

DELTA_LIMIT = 120

DT_FORMAT = "%Y-%m-%dT%H:%M:%S"


logger = logging.getLogger( __name__ )

class HeatersDB(object):
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            if not os.path.exists(PATH_EXPORT_HEATERS_TEMP):
                os.makedirs(PATH_EXPORT_HEATERS_TEMP, exist_ok=True)

            open(db_path, 'w').close()
            self.create_heaters_db(db_path)
        self.__db_path = db_path
        self._conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    @classmethod
    def create_heaters_db(cls, db_path):
        cache_db = cls(db_path)
        cache_db._conn.execute( """CREATE TABLE heaters_temp (
                                ts timestamp primary key,
                                so_nominal text,
                                lno_nominal text,
                                so_redundant text,
                                lno_redundant text,
                                uvis_nominal text
                                )""" )
        cache_db._conn.commit()
        return cache_db

    def close(self):
        self._conn.close()

    def delete(self, start_dt, end_dt):
        cur = self._conn.cursor()
        cur.execute("DELETE FROM heaters_temp WHERE ts >= :start AND ts <= :end", {"start" : start_dt, "end" : end_dt})
        logger.info("%d rows affected." % cur.rowcount)
        self._conn.commit()
        cur.close()

    def insert_row(self, values):
        self._conn.execute("INSERT OR IGNORE INTO heaters_temp VALUES (?,?,?,?,?,?)", values)
        self._conn.commit()

    def clear_table(self):
        self._conn.execute("DROP TABLE heaters_temp")
        self._conn.execute( """CREATE TABLE heaters_temp (
                                ts timestamp primary key,
                                so_nominal text,
                                lno_nominal text,
                                so_redundant text,
                                lno_redundant text,
                                uvis_nominal text
                                )""" )
        self._conn.commit()

    def get_nearest(self, dt):
        query = """SELECT *, ABS(strftime('%s',v) - strftime('%s',:dt)) AS delta FROM (
        SELECT *, MIN(ts) AS v FROM heaters_temp WHERE ts >= :dt
        UNION SELECT *, MAX(ts) AS v FROM heaters_temp WHERE ts <= :dt )
        ORDER BY delta LIMIT 1"""
        res = self._conn.execute(query, {"dt": dt})
        return res.fetchone()

    def get_range(self, start_dt, end_dt):
        res = self._conn.execute("SELECT * FROM heaters_temp WHERE ts >= :start AND ts <= :end", {"start" : start_dt, "end" : end_dt})
        return res.fetchall()


def scan_tm1553_372(beg_dt, end_dt):
    edds_trees = EDDS_Trees()
    beg_ts = beg_dt.replace(tzinfo=timezone.utc).timestamp()
    end_ts = end_dt.replace(tzinfo=timezone.utc).timestamp()
    heaters_pkts = []
    files = edds_trees.tm1553_372.files_for_period(beg_dt, end_dt)
    files.sort(key=lambda x: x.beg_dtime)

    for file_info in files:
        edds_file = edds_files.EDDS_1553_Tm_File(file_info.path)
        logger.info("Processing %s" % file_info.path)
        for pkt in edds_file:
            ts = pkt.timestamp
            if ts >= beg_ts and ts <= end_ts:
                if pkt.src == BIRARawPacket.PACKET_SRC_SURVIVAL_TEMP_EDDS:
                    heaters_pkts.append(pkt)

#SO baseplate (nominal sensor)
#3				LNO baseplate (nominal sensor)
#4				SO baseplate (redundant sensor)
#5				LNO baseplate (redundant sensor)
#6				UVIS baseplate

    heaters_pkts.sort()
    # Remove adjacent duplicates
    heaters_pkts = [k for k,_ in itertools.groupby(heaters_pkts)]
#    headers = ["datetime", "so_nominal", "lno_nominal", "so_redundant", "lno_redundant", "uvis_nominal"]
    temp_readings = []

    for row_ix, pkt in enumerate(heaters_pkts):
        temps = struct.unpack(">ffffff", pkt.buf)
        temps = [calib.SurvivalHeatersCorrection.convert(temps[i]) for i in (0,4,1,3,5)]
        dt = datetime.utcfromtimestamp(pkt.timestamp)
        temp_readings.append((dt,)+tuple(temps))

    return temp_readings

def update_heaters_db(args):
    db_obj = HeatersDB(DB_PATH)
    if args.regen:
        logger.info("Clearing the heaters db temperature table")
        db_obj.clear_table()
    if not args.beg or not args.end:
        logger.info("No begin or end time specified, skip scanning TM1553_372 files.")
        return
    logger.info("Removing existing rows between %s and %s" % (args.beg.strftime(DT_FORMAT), args.end.strftime(DT_FORMAT)))
    db_obj.delete(args.beg, args.end)
    logger.info("Scanning the TM1553_372 files for heater temperatures")
    for i in scan_tm1553_372(args.beg, args.end): db_obj.insert_row(i)
    db_obj.close()

def get_temperature(dt):
    db_obj = HeatersDB(DB_PATH)
    try:
        temps = db_obj.get_nearest(dt)
        if temps:
            if temps[7] > DELTA_LIMIT:
                logger.error("No data found for %s: delta limit exceeded", dt)
            else:
                return temps
        else:
            logger.error("No data found: check database")

    except Exception as e:
        logger.error(e.args[0])
    finally:
        db_obj.close()

def get_temperature_range(beg_dt, end_dt):
#    logger.info("Getting temperature range from %s", DB_PATH)
    db_obj = HeatersDB(DB_PATH)
    try:
        temps = db_obj.get_range(beg_dt, end_dt)
        if temps:
            return temps
        else:
            logger.error("No data found: check database")
            return temps

    except Exception as e:
        logger.error(e.args[0])
    finally:
        db_obj.close()
