"""Console script for CloudMask_NB."""
import os
import sys
import argparse

import numpy as np
import tifffile as tiff

from CloudMask_NB.FY4A.NavFKM import FY4NavFile, get_nav_path
from CloudMask_NB.FY4A.GEOFKM import FY4AAGRIL1GEODISK4KM
from CloudMask_NB.FY4A.FDIFKM import FY4AAGRIL1FDIDISK4KM
from CloudMask_NB.FY4A.CLMFKM import FY4AAGRICLM4KM

from CloudMask_NB.FY4A.NavieBayes import T11, TStd, Bt1185, T11, Btd37511Night, \
    TmaxT, Emiss375Day, Emiss375Night, GeoColorRGB


def detect_cloud_mask(agri_l1_file_path, agri_geo_file_path,
                      agri_clm_tif_path):
    fy4_l1 = FY4AAGRIL1FDIDISK4KM(agri_l1_file_path)
    fy4_geo = FY4AAGRIL1GEODISK4KM(agri_geo_file_path)
    month = fy4_l1.start_time_stamp.month
    fy4_nav_file_path = get_nav_path(month)
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

    sun_zen = fy4_geo.get_sun_zenith()

    bt_372 = fy4_l1.get_band_by_channel('bt_372_low')
    ems_372 = fy4_l1.get_band_by_channel('ems_372')
    bt_850 = fy4_l1.get_band_by_channel('bt_850')
    bt_1080 = fy4_l1.get_band_by_channel('bt_1080')

    r = []
    # 1 TStd
    
    tstd = TStd(lut_file_path=TStd.get_lut_path(month))
    x = tstd.prepare_feature(bt_1080)
    valid_mask = tstd.prepare_valid_mask(bt_1080, dem, sft, coastal_mask,
                                         space_mask)
    ratio, prob = tstd.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 2 Bt1185
    bt1185 = Bt1185(lut_file_path=Bt1185.get_lut_path(month))
    x = bt1185.prepare_feature(bt_1080, bt_850)
    valid_mask = bt1185.prepare_valid_mask(bt_1080, bt_850, sft, space_mask)
    ratio, prob = bt1185.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 3 T11
    t11 = T11(lut_file_path=T11.get_lut_path(month))
    x = t11.prepare_feature(bt_1080)
    valid_mask = t11.prepare_valid_mask(bt_1080, sft, space_mask)
    ratio, prob = t11.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 4 Btd37511Night
    btd37511 = Btd37511Night(lut_file_path=Btd37511Night.get_lut_path(month))
    x = btd37511.prepare_feature(bt_372, bt_1080)
    valid_mask = btd37511.prepare_valid_mask(bt_372, bt_1080, sft, sun_zen,
                                             space_mask)
    ratio, prob = btd37511.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 5  Tmax-T
    tmax_t = TmaxT(lut_file_path=TmaxT.get_lut_path(month))
    x = tmax_t.prepare_feature(bt_1080)
    valid_mask = tmax_t.prepare_valid_mask(bt_1080, dem, sft, coastal_mask,
                                           space_mask)
    ratio, prob = tmax_t.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 6  GeoColor
    geo_color_tif_path = fy4_l1.fname.replace('FDI', 'CLR').replace(
        'V0001.HDF', 'GeoColor.tif')
    geo_color = GeoColorRGB(lut_file_path=GeoColorRGB.get_lut_path(month))
    x = geo_color.prepare_feature(geo_color_tif_path)
    valid_mask = geo_color.prepare_valid_mask(x, sft, space_mask)
    ratio, prob = geo_color.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 7  Emiss4Day
    emiss4day = Emiss375Day(lut_file_path=Emiss375Day.get_lut_path(month))
    x = emiss4day.prepare_feature(ems_372)
    valid_mask = emiss4day.prepare_valid_mask(ems_372, sft, sun_zen,
                                              space_mask)
    ratio, prob = emiss4day.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 8  Emiss4Night
    emiss4night = Emiss375Night(lut_file_path=Emiss375Night.get_lut_path(month))
    x = emiss4night.prepare_feature(ems_372)
    valid_mask = emiss4night.prepare_valid_mask(ems_372, sft, sun_zen,
                                                space_mask)
    ratio, prob = emiss4night.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)

    # 融合
    r_s = np.stack(r)
    r_s_p = np.prod(r_s, 0)
    prior_yes = t11.lut_ds['prior_yes'].data[sft[sft > 0] - 1]
    p = np.ma.masked_array(np.zeros(x.shape), space_mask)
    p[sft > 0] = 1.0 / (1.0 + r_s_p[sft > 0] / prior_yes - r_s_p[sft > 0])

    space_mask_i = space_mask.astype(np.int)
    dig_p = np.ones(space_mask.shape, np.uint8) * 4
    dig_p[p >= 0.9] = 0
    dig_p[np.logical_and(p >= 0.5, p < 0.9)] = 1
    dig_p[np.logical_and(p > 0.1, p < 0.5)] = 2
    dig_p[p <= 0.1] = 3
    dig_p[space_mask_i == 1] = 126
    tiff.imwrite(agri_clm_tif_path, dig_p)
    return 0


def main():
    """Console script for metesatpy."""
    parser = argparse.ArgumentParser()
    parser.add_argument('agri_l1_file_path',
                        type=str,
                        help="input FY4A L1 HDF file.")
    parser.add_argument('agri_geo_file_path',
                        type=str,
                        help="input FY4A L1 HDF file.")
    parser.add_argument('agri_clm_tif_path',
                        type=str,
                        help="output FY4A CLM HDF file.")
    args = parser.parse_args()
    detect_cloud_mask(args.agri_l1_file_path, 
                      args.agri_geo_file_path,
                      args.agri_clm_tif_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
