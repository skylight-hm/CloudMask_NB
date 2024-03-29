import numpy as np
import os
import copy

from scipy.interpolate import Rbf

import h5py
import re
import tqdm
import pandas as pd
import dask as da

from metesatpy.production.FY4A import FY4NavFile, FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM, FY4AAGRICLM4KM
from metesatpy.algorithms.CloudMask import Ref063Min3x3Day
from metesatpy.utils.cspp import infer_airmass, infer_scat_angle_short

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')

fy4_nav_file_name = 'fygatNAV.FengYun-4A.xxxxxxx.4km.hdf'
fy4_nav_file_path = os.path.join(data_root_dir, fy4_nav_file_name)
fy4_nav = FY4NavFile(fy4_nav_file_path)

snow_mask = fy4_nav.get_snow_mask()
snow_mask = snow_mask == 3
pix_lat = fy4_nav.get_latitude()
pix_lon = fy4_nav.get_longitude()
dem = fy4_nav.get_dem()
sft = fy4_nav.prepare_surface_type_to_cspp()
coastal = fy4_nav.get_coastal()
coastal_mask = coastal > 0
space_mask = fy4_nav.get_space_mask(b=True)

sat_lat = 0
sat_lon = 104.7

workspace = r"D:\WorkSpace\20200429\project\data"

df = pd.read_csv('train.csv')

ref_063_min_day = Ref063Min3x3Day()
ref_063_min_day_lut_ds_x = copy.deepcopy(ref_063_min_day.lut_ds)

for idx, csft in enumerate(ref_063_min_day_lut_ds_x.cspp_sft.data.tolist()):
    sft_cursor = idx + 1
    sft_mask = sft == sft_cursor
    start_idx = 40
    for i in tqdm.trange(start_idx, 100, desc='%s' % csft):
        agri_l1_file_path = df.loc[df.index[i], ('l1')]
        agri_geo_file_path = df.loc[df.index[i], ('geo')]
        agri_clm_file_path = df.loc[df.index[i], ('clm')]
        l1 = FY4AAGRIL1FDIDISK4KM(agri_l1_file_path)
        clm = FY4AAGRICLM4KM(agri_clm_file_path)
        geo = FY4AAGRIL1GEODISK4KM(agri_geo_file_path)
        ref_065 = l1.get_band_by_channel('ref_065')

        sun_zen = geo.get_sun_zenith()
        sat_zen = geo.get_satellite_zenith()
        sun_glint = geo.get_sun_glint()
        scat_ang = infer_scat_angle_short(pix_lat, pix_lon, sat_lat, sat_lon, sun_zen, sat_zen)
        air_mass = infer_airmass(sat_zen, sun_zen)

        ref_063_min_day = Ref063Min3x3Day()
        x = ref_063_min_day.prepare_feature(ref_065) * 100

        valid_mask = ref_063_min_day.prepare_valid_mask(ref_065, dem, sft, sun_zen, coastal_mask, space_mask)

        sft_valide_mask = np.logical_and(valid_mask, sft_mask)

        clm_array = clm.get_clm()
        label = clm_array[sft_valide_mask]
        fea = x[sft_valide_mask]

        cloudy_f = fea[label == 0]
        if cloudy_f.shape[0] > 0:
            cloudy_da_fea = da.array.from_array(cloudy_f, chunks='auto')  # cloudy
        if i == start_idx:
            cloudy_da_fea_total = cloudy_da_fea
        else:
            cloudy_da_fea_total = da.array.concatenate((cloudy_da_fea_total, cloudy_da_fea))

        clear_f = fea[label == 3]
        if clear_f.shape[0] > 0:
            clear_da_fea = da.array.from_array(clear_f, chunks='auto')  # clear
        if i == start_idx:
            clear_da_fea_total = clear_da_fea
        else:
            clear_da_fea_total = da.array.concatenate((clear_da_fea_total, clear_da_fea))

    v_min = min(cloudy_da_fea_total.min().compute(), clear_da_fea_total.min().compute())
    v_max = max(cloudy_da_fea_total.max().compute(), clear_da_fea_total.max().compute())

    cloudy_h, bins1 = da.array.histogram(cloudy_da_fea_total, bins=100, range=[v_min, v_max])
    clear_h, bins2 = da.array.histogram(clear_da_fea_total, bins=100, range=[v_min, v_max])
    cloudy_counts = cloudy_h.compute()
    clear_counts = clear_h.compute()
    prior_yes = cloudy_counts.sum() / (cloudy_counts.sum() + clear_counts.sum())
    condition_no = clear_counts / clear_counts.sum()
    condition_yea = cloudy_counts / cloudy_counts.sum()
    ratio = condition_no / (condition_yea + 1e-5)

    ref_063_min_day_lut_ds_x.prior_yes.data[idx] = prior_yes
    ref_063_min_day_lut_ds_x.bin_start.data[idx] = bins1[0]
    ref_063_min_day_lut_ds_x.bin_end.data[idx] = bins1[-1]
    ref_063_min_day_lut_ds_x.delta_bin.data[idx] = bins1[1] - bins1[0]
    ref_063_min_day_lut_ds_x.bins.data[idx, :] = bins1[1:]
    ref_063_min_day_lut_ds_x.class_cond_ratio_reg.data[idx, :] = ratio

ref_063_min_day_lut_ds_x.to_netcdf(r'D:\WorkSpace\20200429\project\data\LUT\Ref_063_Min_3x3_Day_60.nc')
