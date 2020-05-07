from .DISK import FY4AAGRIL1GEODISKProduction

import traceback

import h5py
import numpy as np


class FY4AAGRIL1GEODISK4KM(FY4AAGRIL1GEODISKProduction):

    def __init__(self, fname: str = None, **kwargs):
        super(FY4AAGRIL1GEODISK4KM, self).__init__()
        self.fname = fname

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
        dtor = np.pi / 180
        cos_zen = np.cos(dtor * son_zen)
        cos_solar_zen = np.cos(dtor * sol_zen)
        air_mass = np.where(np.logical_and(cos_zen != 0, cos_solar_zen != 0),
                            1.0 / cos_solar_zen + 1.0 / cos_zen,
                            np.ones(cos_solar_zen.shape))
        return air_mass
