import os

import numpy as np
import xarray as xr
import h5py

from metesatpy.production import FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM
from metesatpy.production.FY4A import FY4NavFile
from metesatpy.utils.conv import cal_nxn_indices


class Ref063Min3x3Day(object):
    lut_ds: xr.Dataset

    def __init__(self, **kwargs):
        super(Ref063Min3x3Day, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', None)
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, fy4_l1: FY4AAGRIL1FDIDISK4KM):
        ref_065 = fy4_l1.get_band_by_channel('ref_065')
        ref_065_min = cal_nxn_indices(ref_065, func=np.min)  # 2748 x 2748
        feature = ref_065 - ref_065_min
        return feature

    def prepare_valid_mask(self,
                           fy4_l1: FY4AAGRIL1FDIDISK4KM,
                           fy4_l1_geo: FY4AAGRIL1GEODISK4KM,
                           fy4_nav: FY4NavFile):
        # obs mask
        ref_065 = fy4_l1.get_band_by_channel('ref_065')
        obs_mask = ~ref_065.mask
        # day mask
        sun_zen = fy4_l1_geo.get_sun_zenith()
        day_mask = sun_zen <= 85.0
        # dem mask
        dem = fy4_nav.get_dem()
        sft = fy4_nav.prepare_surface_type_to_cspp()
        dem_mask = np.logical_and(dem > 2000, sft != 6)
        # coastal mask
        coastal = fy4_nav.get_coastal()
        coastal_mask = coastal > 0
        # accumulate
        valid_mask = np.logical_and(obs_mask, day_mask)
        return valid_mask

    def infer(self,
              fy4_l1: FY4AAGRIL1FDIDISK4KM,
              fy4_l1_geo: FY4AAGRIL1GEODISK4KM,
              fy4_nav: FY4NavFile):
        sft = fy4_nav.prepare_surface_type_to_cspp()
        valid_mask = self.prepare_valid_mask(fy4_l1, fy4_l1_geo, fy4_nav)
        x = self.prepare_feature(fy4_l1)
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1, 4]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1, 4]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, 4, bin_idx_i - 1]  # sft, bin_idx start from 1
        r = np.ma.masked_array(np.zeros(x.shape), valid_mask)
        r[valid_mask] = r_v
        return r
