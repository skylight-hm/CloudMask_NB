import os
import datetime
import traceback

import h5py
import numpy as np

from .ProductionBase import ProductionBase
from ..utils.cspp import infer_airmass


class FY4AAGRIL1GEODISK4KM(ProductionBase):
    def __init__(self, fname: str = None, **kwargs):
        super(FY4AAGRIL1GEODISK4KM, self).__init__()
        try:
            self.fname = fname
            self.fdir = os.path.dirname(fname)
            self.fbname = os.path.basename(fname)
            sdt_str = self.fbname.split('_')[9]
            edt_str = self.fbname.split('_')[10]
            self.start_time_stamp = datetime.datetime.strptime(
                sdt_str, '%Y%m%d%H%M%S')
            self.end_time_stamp = datetime.datetime.strptime(
                edt_str, '%Y%m%d%H%M%S')
        except Exception as e:
            print(e)

    @property
    def file_name_pattern(self):
        return "FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_[0-9._]{29}_4000M_V0001.HDF"

    def get_sun_zenith(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f['NOMSunZenith'])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_sun_glint(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f['NOMSunGlintAngle'])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_sun_azimuth(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f['NOMSunAzimuth'])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_satellite_zenith(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f['NOMSatelliteZenith'])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_satellite_azimuth(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f['NOMSatelliteAzimuth'])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def infer_air_mass(self):
        son_zen = self.get_satellite_zenith()
        sol_zen = self.get_sun_zenith()
        air_mass = infer_airmass(son_zen, sol_zen)
        return air_mass
