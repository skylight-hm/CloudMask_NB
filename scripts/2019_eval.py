import os
import argparse
import sys
import datetime
import numpy as np
import pandas as pd
import tqdm
import logging
import tifffile as tiff

sys.path.append('/FY4COMM/NBCLM/metesatpy')

from metesatpy.production.FY4A import FY4NavFile, FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM, FY4AAGRICLM4KM

from .auto_eval import confusion, save_fy4_clm_to_tiff, data_frame_cm_metric, appendDFToCSV_void


def main():
    parser = argparse.ArgumentParser(description='Naive Bayes Cloud Mask Algorithm for FY4A AGRI.')
    parser.add_argument('valid_csv', type=str,
                        help='valid csv with parameters')
    parser.add_argument('--sft_str', '-s', type=str,
                        help='surface type string',
                        choices=['Space', 'DeepOcean', 'ShallowOcean', 'UnfrozenLand', 'SnowLand',
                                 'Arctic', 'Antarctic', 'Desert'])
    parser.add_argument('--dn_str', '-d', type=str,
                        help='day night flag',
                        choices=['Day', ' Night', 'ALL'])
    parser.add_argument('--out', '-o', type=str,
                        help='output directory')
    args = parser.parse_args()
    logger = logging.getLogger('fy4 nb clm log')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    df = pd.read_csv(args.valid_csv)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    ddf = df.sample(frac=0.2, replace=True, random_state=29)
    ddf.reset_index(drop=True, inplace=True)
    ddf['p_flag'] = None
    for i in tqdm.trange(len(ddf)):
        t1_str = ddf.loc[i, ('dt')]
        l1_fpath = ddf.loc[i, ('l1')]
        geo_fpath = ddf.loc[i, ('geo')]
        clm_fpath = ddf.loc[i, ('clm')]
        clm_tif = clm_fpath.replace('.NC', '.tif')
        nb_clm_tif = clm_fpath.replace('_CLM', '_NBCLM').replace('.NC', '.tif')
        save_fy4_clm_to_tiff(clm_tif, clm_fpath)
        logger.info('convert clm format!')
        c4 = os.system("/home/fy4/miniconda3/bin/python /FY4COMM/NBCLM/geo_color_fy4/__main__.py {}".format(l1_fpath))
        if c4:
            c5 = os.system(
                "/home/fy4/miniconda3/bin/python /FY4COMM/NBCLM/metesatpy/scripts/entry.py {} {} {}".format(l1_fpath,
                                                                                                            geo_fpath,
                                                                                                            nb_clm_tif))
            if c5 == 0:
                logger.info('nb clm infer complete!')
                logger.info('evaluating...')
                fy4_l1 = FY4AAGRIL1FDIDISK4KM(l1_fpath)
                fy4_geo = FY4AAGRIL1GEODISK4KM(geo_fpath)
                month = fy4_l1.start_time_stamp.month
                fy4_nav_file_path = os.path.join('/FY4COMM/NBCLM/data', 'Assist',
                                                 'fygatNAV.FengYun-4A.xxxxxxx.4km_M%.2d.h5' % month)
                fy4_nav = FY4NavFile(fy4_nav_file_path)
                sft = fy4_nav.prepare_surface_type_to_cspp()
                ref_clm = tiff.imread(clm_tif)
                obj_clm = tiff.imread(nb_clm_tif)

                sft_cursor = fy4_nav.ET_SFT_class.get(args.sft_str, None)
                if sft_cursor == 0:
                    sft_mask = sft > 0
                else:
                    sft_mask = sft == sft_cursor
                dn_flag = args.dn_str
                if dn_flag == 'Day':
                    sun_zen = fy4_geo.get_sun_zenith()
                    time_mask = np.logical_and(sft > 0, sun_zen <= 85.0)
                elif dn_flag == 'Night':
                    sun_zen = fy4_geo.get_sun_zenith()
                    time_mask = np.logical_and(sft > 0, sun_zen > 85.0)
                else:
                    time_mask = sft_mask
                final_mask = np.logical_and(sft_mask, time_mask)

                a, b, c, d = confusion(ref_clm[final_mask], obj_clm[final_mask])
                data = {'dt': [], 'a': [], 'b': [], 'c': [], 'd': []}

                data['dt'].append(t1_str)
                data['a'].append(int(a))
                data['b'].append(int(b))
                data['c'].append(int(c))
                data['d'].append(int(d))

                df = pd.DataFrame(data)
                df = data_frame_cm_metric(df)
                month_csv = os.path.join(args.out,
                                         fy4_l1.start_time_stamp.strftime(
                                             '%Y%m_{}_{}.csv'.format(args.sft_str, args.dn_str)))
                appendDFToCSV_void(df, month_csv)
                logger.info('evaluate complete!')


if __name__ == '__main__':
    main()
