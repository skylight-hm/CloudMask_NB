import os
import argparse
import tifffile as tiff
import numpy as np

import sys
sys.path.append('/FY4COMM/NBCLM/metesatpy')

from metesatpy.production.FY4A import FY4NavFile, FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM, FY4AAGRICLM4KM
from metesatpy.algorithms.CloudMask import TStd, Bt1185, T11, Btd37511Night, TmaxT, Emiss375Day, Emiss375Night, GeoColorRGB

def detect_cloud_mask(agri_l1_file_path, agri_geo_file_path, agri_clm_tif_path):
    
    fy4_l1 = FY4AAGRIL1FDIDISK4KM(agri_l1_file_path)
    fy4_geo = FY4AAGRIL1GEODISK4KM(agri_geo_file_path)
    month = fy4_l1.start_time_stamp.month

    fy4_nav_file_path = os.path.join('/FY4COMM/NBCLM/data', 'Assist', 'fygatNAV.FengYun-4A.xxxxxxx.4km_M%.2d.h5' % month)

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
    tstd = TStd(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/T_Std_M%.2d_handfix.nc" % month)
    x = tstd.prepare_feature(bt_1080)
    valid_mask = tstd.prepare_valid_mask(bt_1080, dem, sft, coastal_mask, space_mask)
    ratio, prob = tstd.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 2 Bt1185
    bt1185 = Bt1185(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/Btd_11_85_M%.2d_handfix.nc" % month)
    x = bt1185.prepare_feature(bt_1080, bt_850)
    valid_mask = bt1185.prepare_valid_mask(bt_1080, bt_850, sft, space_mask)
    ratio, prob = bt1185.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 3 T11
    t11 = T11(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/T_11_M%.2d_handfix.nc" % month)
    x = t11.prepare_feature(bt_1080)
    valid_mask = t11.prepare_valid_mask(bt_1080, sft, space_mask)
    ratio, prob = t11.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 4 Btd37511Night
    btd37511 = Btd37511Night(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/Btd_375_11_Night_M%.2d_handfix.nc" % month)
    x = btd37511.prepare_feature(bt_372, bt_1080)
    valid_mask = btd37511.prepare_valid_mask(bt_372, bt_1080, sft, sun_zen, space_mask)
    ratio, prob = btd37511.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 5  Tmax-T
    tmax_t = TmaxT(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/Tmax_T_M%.2d_handfix.nc" % month)
    x = tmax_t.prepare_feature(bt_1080)
    valid_mask = tmax_t.prepare_valid_mask(bt_1080, dem, sft, coastal_mask, space_mask)
    ratio, prob = tmax_t.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 6  GeoColor
    geo_color_tif_path = fy4_l1.fname.replace('FDI', 'CLR').replace('V0001.HDF', 'GeoColor.tif')
    geo_color = GeoColorRGB(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/GeoColorRGB_M%.2d_handifx.nc" % month)
    x = geo_color.prepare_feature(geo_color_tif_path)
    valid_mask = geo_color.prepare_valid_mask(x, sft, space_mask)
    ratio, prob = geo_color.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 7  Emiss4Day
    emiss4day = Emiss375Day(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/Emiss_375_Day_M%.2d_handifx.nc" % month)
    x = emiss4day.prepare_feature(ems_372)
    valid_mask = emiss4day.prepare_valid_mask(ems_372, sft, sun_zen, space_mask)
    ratio, prob = emiss4day.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    # 8  Emiss4Night
    emiss4night = Emiss375Night(lut_file_path=r"/FY4COMM/NBCLM/data/LUT/Emiss_375_Night_M%.2d_handifx.nc" % month)
    x = emiss4night.prepare_feature(ems_372)
    valid_mask = emiss4night.prepare_valid_mask(ems_372, sft, sun_zen, space_mask)
    ratio, prob = emiss4night.infer(x, sft, valid_mask, space_mask, prob=True)
    r.append(ratio)
    
    # 融合
    r_s = np.stack(r)
    r_s_p = np.prod(r_s, 0)
    prior_yes = t11.lut_ds['prior_yes'].data[sft[sft>0]-1]
    p = np.ma.masked_array(np.zeros(x.shape), space_mask)
    p[sft>0] = 1.0 / (1.0 + r_s_p[sft>0] / prior_yes - r_s_p[sft>0])
    
    space_mask_i = space_mask.astype(np.int)
    dig_p = np.ones(space_mask.shape, np.uint8) * 4
    dig_p[p>=0.9] = 0
    dig_p[np.logical_and(p>=0.5 ,p<0.9)] = 1
    dig_p[np.logical_and(p>0.1 ,p<0.5)] = 2
    dig_p[p<=0.1] = 3
    dig_p[space_mask_i==1] = 126
    tiff.imwrite(agri_clm_tif_path, dig_p)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navie Bayes Cloud Mask Alghrithum for fy4a agri.')
    parser.add_argument('agri_l1_file_path', type=str,
                        help='fy4a agri l1 file path.')
    parser.add_argument('agri_geo_file_path', type=str,
                        help='fy4a agri geo file path.')
    parser.add_argument('agri_clm_tif_path', type=str,
                        help='fy4a agri clm tif path.')

    args = parser.parse_args()
    detect_cloud_mask(args.agri_l1_file_path, args.agri_geo_file_path, args.agri_clm_tif_path)  # pragma: no cover