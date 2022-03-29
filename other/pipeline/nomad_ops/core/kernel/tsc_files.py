# -*- coding: utf-8 -*-

from datetime import datetime
import logging
import os.path
import re
import urllib.request
import urllib.error

from nomad_ops.config import PATH_LOCAL_SPICE_KERNELS

__project__   = "NOMAD"
__author__    = "Roland Clairquin"
__contact__   = "roland.clairquin@oma.be"

logger = logging.getLogger( __name__ )

SCLK_KERNEL_URL = "ftp://spiftp.esac.esa.int/data/SPICE/ExoMars2016/kernels/sclk/em16_tgo_step_20170201.tsc"
SCLK_KERNEL_URL = "ftp://spiftp.esac.esa.int/data/SPICE/ExoMars2016/kernels/sclk/em16_tgo_step_20171024.tsc"
LEAPS_KERNEL_URL = "ftp://spiftp.esac.esa.int/data/SPICE/ExoMars2016/kernels/lsk/naif0012.tls"

KERNELS_LOCAL_PATH = "local"

class TSCFile(object):
    FIELDS_RE = re.compile("\s*([\w/]+)\s*=\s*(?:(?:\(([^\(]*)\))|(?:([^\n]*)\n))\s*")
    cache = {}

    @classmethod
    def get_file(cls, url):
        if url not in cls.cache:
            cls.cache[url] = cls(url)
        return cls.cache[url]

    def __init__(self, url):
        logger.info("Creating kernel file (%s): %s", type(self), url)
        self._field_dict = {}
        data_buf = ""
        try:
            fd = urllib.request.urlopen(url, timeout=1.0)
        except urllib.error.URLError:
            local_path = os.path.join(PATH_LOCAL_SPICE_KERNELS, url.split("ExoMars2016/kernels/")[-1])
            #local_path = os.path.join(KERNELS_LOCAL_PATH, url.split("/")[-1])
            logger.error("Unable to retrieve : %s\nUsing local copy %s", url, local_path)
            fd = open(local_path, "rb")

        parse_flag = False
        for line in fd:
            line = line.decode("ascii")
            if parse_flag:
                if line.strip().startswith("\\begintext"):
                    break
                else:
                    data_buf += line
            if line.strip().startswith("\\begindata"):
                parse_flag = True
        fd.close()

        if data_buf:
            for field_name, f_str1, f_str2 in self.FIELDS_RE.findall(data_buf):
                field_str = f_str1 or f_str2
                self._field_dict[field_name.strip()] = field_str.strip()

class LEAPS_TSCFile(TSCFile):
    def __init__(self, url):
        super(LEAPS_TSCFile, self).__init__(url)

        self._deltat_coeffs = []
        coeff_iter = iter(self._field_dict["DELTET/DELTA_AT"].split())
        while True:
            try:
                c0 = int(next(coeff_iter).strip(" ,"))
            except StopIteration:
                break
            c1 = datetime.strptime(next(coeff_iter).strip(), "@%Y-%b-%d")
            self._deltat_coeffs.append((c0,c1))

    def leap_seconds_at(self, ts):
        dtime = datetime.utcfromtimestamp(ts)
        coeffs = self._deltat_coeffs[0]
        for next_coeffs in self._deltat_coeffs:
            if next_coeffs[1] > dtime:
                break
            coeffs = next_coeffs
        return coeffs[0]

class SCLK_TSCFile(TSCFile):
    def __init__(self, url):
        super(SCLK_TSCFile, self).__init__(url)
        self._sclk1_coeffs = []
        coeff_iter = iter(self._field_dict["SCLK01_COEFFICIENTS_143"].split())
        J2000_TIME_OFFSET = (datetime(2000, 1, 1, 11, 58, 55, 816000) -
                             datetime.utcfromtimestamp(0)).total_seconds()
        leaps_file = LEAPS_TSCFile.get_file(LEAPS_KERNEL_URL)
        while True:
            try:
                c0 = float(next(coeff_iter).strip()) / 65536
            except StopIteration:
                break
            c1 = float(next(coeff_iter).strip()) + J2000_TIME_OFFSET
            c1 -= leaps_file.leap_seconds_at(c1) - 32 # 32 == leaps counter @ 2000-1-1
            c2 = float(next(coeff_iter).strip())
            self._sclk1_coeffs.append((c0,c1,c2))
            #print c0, c1,"(",datetime.utcfromtimestamp(c1),")", c2

    @property
    def sclk1_coeffs(self):
        return self._sclk1_coeffs

    def obt_to_datetime(self, obt):
        coeffs = self._sclk1_coeffs[0]
        for next_coeffs in self._sclk1_coeffs[1:]:
            if next_coeffs[0] >  obt:
                break
            coeffs = next_coeffs

        #print coeffs[1], "+ (",obt, "-", coeffs[0],") *", coeffs[2], "->", dt
        return coeffs[1] + (obt - coeffs[0]) * coeffs[2]

