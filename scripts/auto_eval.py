import os
import argparse
import datetime
import logging

import pandas as pd
import numpy as np
import tifffile as tiff

import sys
sys.path.append('/FY4COMM/NBCLM/metesatpy')

from metesatpy.production.FY4A import FY4NavFile, FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM, FY4AAGRICLM4KM

def main(dt, save_space):
    logger = logging.getLogger('fy4 nb clm log')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    download_command = "wget -c --ftp-user=guoxuexingNRT --ftp-password=za4158 -P {save_space} {url}"
    t1_str = dt.strftime('%Y%m%d%H%M%S')
    t2 = dt + datetime.timedelta(minutes=14, seconds=59)
    t2_str = t2.strftime('%Y%m%d%H%M%S')
    
    l1_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_{t1_str}_{t2_str}_4000M_V0001.HDF'.format(t1_str=t1_str,
                                                                                                          t2_str=t2_str)
    geo_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_{t1_str}_{t2_str}_4000M_V0001.HDF'.format(t1_str=t1_str,
                                                                                                          t2_str=t2_str)
    clm_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_{t1_str}_{t2_str}_4000M_V0001.NC'.format(t1_str=t1_str,
                                                                                                          t2_str=t2_str)
    
    l1_url_prefix = dt.strftime('ftp://ftp.nsmc.org.cn/FY4A/AGRI/L1/FDI/DISK/4000M/%Y/%Y%m%d/')
    geo_url_prefix = dt.strftime('ftp://ftp.nsmc.org.cn/FY4A/AGRI/L1/FDI/DISK/GEO/%Y/%Y%m%d/')
    clm_url_prefix = dt.strftime('ftp://ftp.nsmc.org.cn/FY4A/AGRI/L2/CLM/DISK/NOM/%Y/%Y%m%d/')
    
    logger.info('time cursor: %s -- %s' % (t1_str, t2_str))
    logger.info('l1 downloading!')
    c1 = os.system(download_command.format(save_space=save_space, url=l1_url_prefix + l1_file_name))
    if c1 == 0:
        logger.info('l1 download complete!') 
    else:
        logger.error('l1 download failed!')
        return -1
    logger.info('geo downloading!')
    c2 = os.system(download_command.format(save_space=save_space, url=geo_url_prefix + geo_file_name))
    if c2 == 0:
        logger.info('geo download complete!') 
    else:
        logger.error('geo download failed!')
        return -1
    logger.info('clm downloading!')
    c3 = os.system(download_command.format(save_space=save_space, url=clm_url_prefix + clm_file_name))
    if c3 == 0:
        logger.info('clm download complete!') 
    else:
        logger.error('clm download failed!')
        return -1
    
    l1_fpath = os.path.join(save_space, l1_file_name)
    logger.info('ploting geo color!')
    c4 = os.system("/home/fy4/miniconda3/bin/python /FY4COMM/NBCLM/geo_color_fy4/__main__.py {}".format(l1_fpath))
    if c4 == 0 :
        logger.info('geo color complete!')
        geo_fpath = os.path.join(save_space, geo_file_name)
        clm_fpath = os.path.join(save_space, clm_file_name)
        clm_tif = clm_fpath.replace('.NC', '.tif')
        save_fy4_clm_to_tiff(clm_tif, clm_fpath)
        nb_clm_tif = clm_fpath.replace('_CLM', '_NBCLM').replace('.NC', '.tif')
        c5 = os.system("/home/fy4/miniconda3/bin/python /FY4COMM/NBCLM/metesatpy/scripts/entry.py {} {} {}".format(l1_fpath, geo_fpath ,nb_clm_tif))
        if c5 == 0:
            logger.info('nb clm infer complete!')
            logger.info('evaluating...')
            fy4_l1 = FY4AAGRIL1FDIDISK4KM(l1_fpath)
            month = fy4_l1.start_time_stamp.month
            fy4_nav_file_path = os.path.join('/FY4COMM/NBCLM/data', 'Assist', 'fygatNAV.FengYun-4A.xxxxxxx.4km_M%.2d.h5'  % month)
            fy4_nav = FY4NavFile(fy4_nav_file_path)
            sft = fy4_nav.prepare_surface_type_to_cspp()
            sft_mask = sft > 0

            ref_clm = tiff.imread(clm_tif)
            obj_clm =  tiff.imread(nb_clm_tif)
            
            a, b, c, d = confusion(ref_clm[sft_mask], obj_clm[sft_mask])
            data = {'dt':[], 'a':[], 'b':[], 'c':[], 'd':[]}
            
            data['dt'].append(t1_str)
            data['a'].append(int(a))
            data['b'].append(int(b))
            data['c'].append(int(c))
            data['d'].append(int(d))
            
            df = pd.DataFrame(data)
            df = data_frame_cm_metric(df)
            logger.debug(str(df))
            
            month_csv = os.path.join(save_space, dt.strftime('%Y%m.csv'))
            appendDFToCSV_void(df, month_csv)
            logger.info('evaluate complete!')
        os.remove(l1_fpath)
        os.remove(geo_fpath)
        os.remove(clm_fpath)
        os.remove(l1_fpath.replace('_FDI','_CLR').replace('V0001.HDF', 'GeoColor.tif'))
        os.remove(clm_tif)
        os.remove(nb_clm_tif)

def confusion(y_true: np.ma.masked_array, y_pred: np.ma.masked_array):
    a = np.logical_and(y_true == 0, y_pred == 0).sum()
    b = np.logical_and(y_true == 3, y_pred == 0).sum()
    c = np.logical_and(y_true == 0, y_pred == 3).sum()
    d = np.logical_and(y_true == 3, y_pred == 3).sum()
    return a, b, c, d

def data_frame_cm_metric(df, eps=1e-5):
    df['pod_cloudy'] = df['a'] / (df['a'] + df['b'] + eps)
    df['pod_clear'] = df['d'] / (df['c'] + df['d'] + eps)

    df['far_cloudy'] = df['c'] / (df['a'] + df['c'] + eps)
    df['far_clear'] = df['b'] / (df['b'] + df['d'] + eps)

    df['hr'] = (df['a'] + df['d']) / (df['a'] + df['b'] + df['c'] + df['d'] + eps)
    df['kss'] = (df['a'] * df['d'] - df['c'] * df['b']) / ((df['a'] + df['b']) * (df['c'] + df['d']) + eps)
    return df

def appendDFToCSV_void(df, csvFilePath, sep=","):
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def save_fy4_clm_to_tiff(tif_path, agri_clm_file_path):
    fy4_clm = FY4AAGRICLM4KM(agri_clm_file_path)
    clm_data = fy4_clm.get_clm()
    tiff.imwrite(tif_path, clm_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navie Bayes Cloud Mask Alghrithum for fy4a agri.')
    parser.add_argument('dt', type=str,
                        help='datetime string in \'YYYYmmdd_HHMM\' format.')
    parser.add_argument('--out', '-o', type=str,
                        default='/FY4COMM/NBCLM/PROC',
                        help='process space.')
    args = parser.parse_args()
    dt = datetime.datetime.strptime(args.dt, '%Y%m%d_%H%M')
    main(dt, args.out)