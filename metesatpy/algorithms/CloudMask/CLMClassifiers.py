import os

import numpy as np
import xarray as xr

from metesatpy.utils.conv import cal_nxn_indices

lut_root_dir = os.path.join(os.getenv('METEPY_DATA_PATH', 'data'), 'LUT')


class Ref063Min3x3Day(object):
    lut_ds: xr.Dataset
    short_name: str = 'Ref063Min3x3Day'
    lut_file_name: str = 'Ref_063_Min_3x3_Day.nc'

    def __init__(self, **kwargs):
        super(Ref063Min3x3Day, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_063: np.ma.masked_array):
        ref_063_min = cal_nxn_indices(ref_063, func=np.min)  # 2748 x 2748
        feature = ref_063 - ref_063_min
        return feature

    def prepare_valid_mask(self,
                           obs: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           sun_zen: np.ndarray,
                           coastal_mask: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs_mask
        obs_mask = ~obs.mask
        # day mask
        day_mask = sun_zen <= 85.0
        day_mask = day_mask.data
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        # day with obs
        valid_mask1 = np.logical_and(obs_mask, day_mask)
        # not mount and coastal
        valid_mask2 = np.logical_and(~mount_mask, ~coastal_mask)
        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class TStd(object):
    lut_ds: xr.Dataset
    short_name: str = 'TStd'
    lut_file_name: str = 'T_Std.nc'

    def __init__(self, **kwargs):
        super(TStd, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, bt_1080: np.ma.masked_array):
        bt_1080_std = cal_nxn_indices(bt_1080, func=np.std)  # 2748 x 2748
        feature = bt_1080_std
        return feature

    def prepare_valid_mask(self,
                           obs: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           coastal_mask: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs mask
        obs_mask = ~obs.mask
        # obs value mask
        obs_v_mask = obs >= 0
        # mount mask
        mount_mask = np.logical_and(dem > 2000, sft != 6)

        # with obs
        valid_mask1 = np.logical_and(obs_mask, obs_v_mask)

        # not mount and coastal
        valid_mask2 = np.logical_and(~mount_mask, ~coastal_mask)

        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)

        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class Bt1185(object):
    lut_ds: xr.Dataset
    short_name: str = 'Btd_11_85'
    lut_file_name: str = 'Btd_11_85.nc'

    def __init__(self, **kwargs):
        super(Bt1185, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, bt_1080: np.ma.masked_array, bt_850: np.ma.masked_array):
        feature = bt_1080 - bt_850
        return feature

    def prepare_valid_mask(self,
                           bt_1080: np.ma.masked_array,
                           bt_850: np.ma.masked_array,
                           sft: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs mask
        bt_1080_mask = ~bt_1080.mask
        bt_850_mask = ~bt_850.mask
        obs_mask = np.logical_and(bt_1080_mask, bt_850_mask)
        # cold mask
        cloud_mask = bt_1080.data < 220
        # accumulate
        valid_mask = np.logical_and(obs_mask, ~cloud_mask)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class RefRatioDay(object):
    lut_ds: xr.Dataset
    short_name: str = 'RefRatioDay'
    lut_file_name: str = 'Ref_Ratio_Day.nc'

    def __init__(self, **kwargs):
        super(RefRatioDay, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_063: np.ma.masked_array, ref_086: np.ma.masked_array):
        feature = ref_086 / ref_063
        return feature

    def prepare_valid_mask(self,
                           ref_063: np.ma.masked_array,
                           ref_086: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           sun_zen: np.ndarray,
                           sun_glint: np.ndarray,
                           space_mask: np.ndarray = None,
                           bt_1080: np.ma.masked_array = None):
        # obs_mask
        ref_063_mask = ~ref_063.mask
        ref_086_mask = ~ref_086.mask
        obs_mask = np.logical_and(ref_063_mask, ref_086_mask)

        # day mask
        day_mask = sun_zen <= 85.0
        day_mask = day_mask.data
        # mount mask
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        #  glint mask glint threshold 40
        glint = np.zeros(sun_glint.shape, np.uint8)
        glint[sun_glint < 40] = 1
        glint[bt_1080 < 273.15] = 0
        bt_1080_std_3x3 = cal_nxn_indices(bt_1080, 1, np.std)
        glint[bt_1080_std_3x3 > 1.0] = 0
        ref_063_std_3x3 = cal_nxn_indices(ref_063, 1, np.std)
        glint[ref_063_std_3x3 > 2.0] = 0
        glint[ref_063 < 5.0] = 0
        glint_mask = glint.astype(np.bool)
        # day with obs
        valid_mask1 = np.logical_and(obs_mask, day_mask)
        # not mount and glint
        valid_mask2 = np.logical_and(~mount_mask, ~glint_mask)
        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class Ref138Day(object):
    lut_ds: xr.Dataset
    short_name: str = 'Ref138Day'
    lut_file_name: str = 'Ref_138_Day.nc'

    def __init__(self, **kwargs):
        super(Ref138Day, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_137: np.ma.masked_array):
        feature = ref_137
        return feature

    def prepare_valid_mask(self,
                           obs: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           sun_zen: np.ndarray,
                           scat_ang: np.ma.masked_array,
                           air_mass: np.ma.masked_array,
                           space_mask: np.ndarray = None):
        # obs mask
        obs_mask = ~obs.mask
        # obs value mask
        obs_v_mask = obs > 0
        # day mask
        day_mask = sun_zen > 80.0
        day_mask = day_mask.data
        # air mass mask
        air_mass_mask = air_mass > 5
        # forward mask
        scat_ang = scat_ang.data
        forward_mask = np.logical_and(scat_ang < 80, sun_zen < 95.0)
        # air mass
        # mount mask
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        # with obs
        valid_mask1 = np.logical_and(obs_mask, obs_v_mask)
        # day with air mass
        valid_mask2 = ~np.logical_or(day_mask, air_mass_mask)
        # not mount and forward
        # accumulate
        valid_mask3 = np.logical_and(~mount_mask, ~forward_mask)
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        valid_mask = np.logical_and(valid_mask, valid_mask3)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class NdsiDay(object):
    lut_ds: xr.Dataset
    short_name: str = 'NdsiDay'
    lut_file_name: str = 'Ndsi_Day.nc'

    def __init__(self, **kwargs):
        super(NdsiDay, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_063: np.ma.masked_array, ref_160: np.ma.masked_array):
        feature = (ref_063 - ref_160) / (ref_063 + ref_160)
        return feature

    def prepare_valid_mask(self,
                           ref_063: np.ma.masked_array,
                           ref_160: np.ma.masked_array,
                           sft: np.ndarray,
                           sun_glint: np.ndarray,
                           sun_zen: np.ndarray,
                           scat_ang: np.ma.masked_array,
                           air_mass: np.ma.masked_array,
                           space_mask: np.ndarray = None,
                           bt_1080: np.ma.masked_array = None):
        # obs mask
        ref_063_mask = ~ref_063.mask
        ref_160_mask = ~ref_160.mask
        obs_mask = np.logical_and(ref_063_mask, ref_160_mask)
        # day mask
        day_mask = sun_zen > 80.0
        day_mask = day_mask.data
        # air mass mask
        air_mass_mask = air_mass > 5
        # obs value mask
        obs_v_mask = np.logical_and(ref_063_mask >= 0, ref_160_mask >= 0)
        #  glint mask glint threshold 40
        glint = np.zeros(sun_glint.shape, np.uint8)
        glint[sun_glint < 40] = 1
        glint[bt_1080 < 273.15] = 0
        bt_1080_std_3x3 = cal_nxn_indices(bt_1080, 1, np.std)
        glint[bt_1080_std_3x3 > 1.0] = 0
        ref_063_std_3x3 = cal_nxn_indices(ref_063, 1, np.std)
        glint[ref_063_std_3x3 > 2.0] = 0
        glint[ref_063 < 5.0] = 0
        glint_mask = glint.astype(np.bool)
        # forward mask
        scat_ang = scat_ang.data
        forward_mask = np.logical_and(scat_ang < 80, sun_zen < 95.0)
        # with obs
        valid_mask1 = np.logical_and(obs_mask, obs_v_mask)
        # day with air mass
        valid_mask2 = ~np.logical_or(day_mask, air_mass_mask)
        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        # not mount and not forward mask
        valid_mask3 = np.logical_and(~glint_mask, ~forward_mask)
        valid_mask = np.logical_and(valid_mask, valid_mask3)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class Ref063Day(object):
    lut_ds: xr.Dataset
    short_name: str = 'Ref063Day'
    lut_file_name: str = 'Ref_063_Day.nc'

    def __init__(self, **kwargs):
        super(Ref063Day, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_065: np.ma.masked_array, ref_065_clear_c: np.ma.masked_array):
        feature = ref_065 - ref_065_clear_c
        return feature

    def prepare_valid_mask(self,
                           ref_063: np.ma.masked_array,
                           ref_063_clear: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           sun_glint: np.ndarray,
                           sun_zen: np.ndarray,
                           scat_ang: np.ma.masked_array,
                           air_mass: np.ma.masked_array,
                           snow_mask: np.ndarray,
                           space_mask: np.ndarray = None,
                           bt_1080: np.ma.masked_array = None
                           ):
        # obs mask
        ref_063_mask = ~ref_063.mask
        ref_063_clear_mask = ~ref_063_clear.mask
        obs_mask = np.logical_and(ref_063_mask, ref_063_clear_mask)
        # day mask
        day_mask = sun_zen > 80.0
        day_mask = day_mask.data
        # air mass mask
        air_mass_mask = air_mass > 5
        # mount mask
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        #  glint mask glint threshold 40
        glint = np.zeros(sun_glint.shape, np.uint8)
        glint[sun_glint < 40] = 1
        glint[bt_1080 < 273.15] = 0
        bt_1080_std_3x3 = cal_nxn_indices(bt_1080, 1, np.std)
        glint[bt_1080_std_3x3 > 1.0] = 0
        ref_063_std_3x3 = cal_nxn_indices(ref_063, 1, np.std)
        glint[ref_063_std_3x3 > 2.0] = 0
        glint[ref_063 < 5.0] = 0
        glint_mask = glint.astype(np.bool)
        # forward mask
        scat_ang = scat_ang.data
        forward_mask = np.logical_and(scat_ang < 80, sun_zen < 95.0)
        # with obs
        # not mount and not forward mask
        # not mount and coastal
        valid_mask1 = np.logical_and(~mount_mask, ~glint_mask)
        # day with air mass
        valid_mask2 = ~np.logical_or(day_mask, air_mass_mask)
        # accumulate
        # not snow and forward mask
        valid_mask3 = np.logical_and(~snow_mask, ~forward_mask)
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        valid_mask = np.logical_and(valid_mask, valid_mask3)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, obs_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)

        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class T11(object):
    lut_ds: xr.Dataset
    short_name: str = 'T_11'
    lut_file_name: str = 'T_11.nc'

    def __init__(self, **kwargs):
        super(T11, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, bt_1080: np.ma.masked_array):
        feature = bt_1080
        return feature

    def prepare_valid_mask(self,
                           bt_1080: np.ma.masked_array,
                           sft: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs mask
        bt_1080_mask = ~bt_1080.mask
        # accumulate
        valid_mask = bt_1080_mask
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class TmaxT(object):
    lut_ds: xr.Dataset
    short_name: str = 'Tmax_T'
    lut_file_name: str = 'Tmax_T.nc'

    def __init__(self, **kwargs):
        super(TmaxT, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, bt_1080: np.ma.masked_array):
        bt_1080_max = cal_nxn_indices(bt_1080, func=np.max)
        feature = bt_1080_max - bt_1080
        return feature

    def prepare_valid_mask(self,
                           bt_1080: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           coastal_mask: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs mask
        bt_1080_mask = ~bt_1080.mask
        # mount
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        # accumulate
        # space mask
        valid_mask1 = np.logical_and(bt_1080_mask, ~space_mask)
        valid_mask2 = np.logical_and(~mount_mask, ~coastal_mask)
        valid_mask = np.logical_and(valid_mask1, valid_mask2)
        # not mount and coastal
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class Btd37511Night(object):
    lut_ds: xr.Dataset
    short_name: str = 'Btd_375_11_Night'
    lut_file_name: str = 'Btd_375_11_Night.nc'

    def __init__(self, **kwargs):
        super(Btd37511Night, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, bt_375: np.ma.masked_array, bt_1080: np.ma.masked_array):
        feature = bt_375 - bt_1080
        return feature

    def prepare_valid_mask(self,
                           bt_375: np.ma.masked_array,
                           bt_1080: np.ma.masked_array,
                           sft: np.ndarray,
                           sun_zen: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs mask
        bt_1080_mask = ~bt_1080.mask
        warm_mask = bt_375 >= 240
        # solar contaminate
        sun_con_mask = sun_zen > 90
        # accumulate
        # space mask
        valid_mask1 = np.logical_and(bt_1080_mask, warm_mask)
        valid_mask = np.logical_and(valid_mask1, sun_con_mask)
        # not mount and coastal
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r


class RefStd(object):
    lut_ds: xr.Dataset
    short_name: str = 'RefStdDay'
    lut_file_name: str = 'Ref_Std.nc'

    def __init__(self, **kwargs):
        super(RefStd, self).__init__()
        lut_file_path = kwargs.get('lut_file_path', os.path.join(lut_root_dir, self.lut_file_name))
        self._load_lut(lut_file_path)

    def _load_lut(self, lut_file_path: str = None):
        if lut_file_path:
            self.lut_ds = xr.open_dataset(lut_file_path)

    def prepare_feature(self, ref_063: np.ma.masked_array):
        feature = cal_nxn_indices(ref_063, func=np.std)
        return feature

    def prepare_valid_mask(self,
                           obs: np.ma.masked_array,
                           dem: np.ndarray,
                           sft: np.ndarray,
                           sun_zen: np.ndarray,
                           coastal_mask: np.ndarray,
                           space_mask: np.ndarray = None):
        # obs_mask
        obs_mask = ~obs.mask
        # day mask
        day_mask = sun_zen <= 85.0
        day_mask = day_mask.data
        mount_mask = np.logical_and(dem > 2000, sft != 6)
        # day with obs
        valid_mask1 = np.logical_and(obs_mask, day_mask)
        # not mount and coastal
        valid_mask2 = np.logical_and(~mount_mask, ~coastal_mask)
        # accumulate
        valid_mask = np.logical_and(valid_mask2, valid_mask1)
        # space mask
        valid_mask = np.logical_and(valid_mask, ~space_mask)
        valid_mask = np.logical_and(valid_mask, sft > 0)
        return valid_mask

    def infer(self,
              x: np.ma.masked_array,
              sft: np.ndarray,
              valid_mask: np.ndarray,
              space_mask: np.ndarray = None,
              prob=False):
        bin_start = self.lut_ds['bin_start'].data[sft[valid_mask] - 1]  # sft start from 1
        delta_bin = self.lut_ds['delta_bin'].data[sft[valid_mask] - 1]  # sft start from 1
        bin_idx = (x[valid_mask] - bin_start) / delta_bin
        bin_idx_i = bin_idx.astype(np.int)
        bin_idx_i = np.clip(bin_idx_i, 1, 100)
        r_da = self.lut_ds['class_cond_ratio_reg']
        r_v = r_da.data[sft[valid_mask] - 1, bin_idx_i - 1]  # sft, bin_idx start from 1
        if space_mask is None:
            space_mask = x.mask
        r = np.ma.masked_array(np.ones(x.shape), space_mask)
        r[valid_mask] = r_v
        if prob:
            prior_yes = self.lut_ds['prior_yes'].data[sft[valid_mask] - 1]  # sft start from 1
            p = np.ma.masked_array(np.zeros(x.shape), space_mask)
            p[valid_mask] = 1.0 / (1.0 + r[valid_mask] / prior_yes - r[valid_mask])
            return r, p
        else:
            return r
