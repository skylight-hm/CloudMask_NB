import os

import numpy as np
import xarray as xr
import h5py

from metesatpy.production import FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM
from metesatpy.production.FY4A import FY4NavFile
from metesatpy.utils.conv import cal_nxn_indices


class Ref063Min3x3Day(object):
    lut_ds: xr.Dataset
    short_name: str = 'Ref063Min3x3Day'

    def __init__(self, **kwargs):
        super(Ref063Min3x3Day, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', None)
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, fy4_l1: FY4AAGRIL1FDIDISK4KM):
        ref_065 = fy4_l1.get_band_by_channel('ref_065') * 100
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
        # day mask (covnert to bool)
        sun_zen = fy4_l1_geo.get_sun_zenith()
        day_mask = sun_zen <= 85.0
        day_mask = day_mask.data
        # dem mask
        dem = fy4_nav.get_dem()
        sft = fy4_nav.prepare_surface_type_to_cspp()
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        # coastal mask
        coastal = fy4_nav.get_coastal()
        coastal_mask = coastal > 0

        # space mask
        space_mask = fy4_nav.get_space_mask(b=True)

        # day with obs
        valid_mask1 = np.logical_and(obs_mask, day_mask)

        # not mount and coastal
        valid_mask2 = np.logical_and(~mount_mask, ~coastal_mask)

        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)

        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        return valid_mask

    def infer(self,
              fy4_l1: FY4AAGRIL1FDIDISK4KM,
              fy4_l1_geo: FY4AAGRIL1GEODISK4KM,
              fy4_nav: FY4NavFile,
              prob=False):
        sft = fy4_nav.prepare_surface_type_to_cspp()
        space_mask = fy4_nav.get_space_mask(True)
        valid_mask = self.prepare_valid_mask(fy4_l1, fy4_l1_geo, fy4_nav)
        x = self.prepare_feature(fy4_l1)
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return p
        else:
            return r
