import tqdm
import os
import re
import glob
import pysftp
import datetime


def download_f4a_agri_l1_4km(start_time, server, out_dir='.'):
    """
    # remote_file_dir = "/FY4/FY4A/AGRI/L1/FDI/DISK/2019/20191010"
    # remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20191010144500_20191010145959_4000M_V0001.HDF'
    """
    end_time = start_time + datetime.timedelta(minutes=14, seconds=59)
    st = start_time.strftime('%Y%m%d%H%M%S')
    et = end_time.strftime('%Y%m%d%H%M%S')
    remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_{st}_{et}_4000M_V0001.HDF'.format(st=st, et=et)
    remote_file_dir = "/FY4/FY4A/AGRI/L1/FDI/DISK/{year}/{yearmonthday}".format(year=start_time.strftime('%Y'),
                                                                                yearmonthday=start_time.strftime(
                                                                                    '%Y%m%d')
                                                                                )
    local_file_dir = out_dir
    local_file_path = os.path.join(local_file_dir, remote_file_name)
    if not os.path.exists(local_file_path):
        sftp = pysftp.Connection(server['host'], username=server['username'], password=server['password'])
        with sftp.cd(remote_file_dir):
            sftp.get(remote_file_name, local_file_path)
        sftp.close()
    return local_file_path


def download_f4a_agri_geo_4km(start_time, server, out_dir='.'):
    """
    # remote_file_dir = "/FY4/FY4A/AGRI/L1/FDI/DISK/2019/20191010"
    # remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20191010144500_20191010145959_4000M_V0001.HDF'
    """
    end_time = start_time + datetime.timedelta(minutes=14, seconds=59)
    st = start_time.strftime('%Y%m%d%H%M%S')
    et = end_time.strftime('%Y%m%d%H%M%S')
    remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_{st}_{et}_4000M_V0001.HDF'.format(st=st, et=et)
    remote_file_dir = "/FY4/FY4A/AGRI/L1/FDI/DISK/{year}/{yearmonthday}".format(year=start_time.strftime('%Y'),
                                                                                yearmonthday=start_time.strftime(
                                                                                    '%Y%m%d')
                                                                                )
    local_file_dir = out_dir
    local_file_path = os.path.join(local_file_dir, remote_file_name)
    if not os.path.exists(local_file_path):
        sftp = pysftp.Connection(server['host'], username=server['username'], password=server['password'])
        with sftp.cd(remote_file_dir):
            sftp.get(remote_file_name, local_file_path)
        sftp.close()
    return local_file_path


def download_f4a_agri_clm_4km(start_time, server, out_dir='.'):
    """
    # remote_file_dir = "/FY4/FY4A/AGRI/L2/FDI/DISK/2019/20191010"
    # remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_20200101010000_20200101011459_4000M_V0001.NC'
    """
    end_time = start_time + datetime.timedelta(minutes=14, seconds=59)
    st = start_time.strftime('%Y%m%d%H%M%S')
    et = end_time.strftime('%Y%m%d%H%M%S')
    remote_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_{st}_{et}_4000M_V0001.NC'.format(st=st, et=et)
    remote_file_dir = "/FY4/FY4A/AGRI/L2/CLM/DISK/NOM/{year}/{yearmonthday}".format(year=start_time.strftime('%Y'),
                                                                                    yearmonthday=start_time.strftime(
                                                                                        '%Y%m%d')
                                                                                    )
    local_file_dir = out_dir
    local_file_path = os.path.join(local_file_dir, remote_file_name)
    if not os.path.exists(local_file_path):
        sftp = pysftp.Connection(server['host'], username=server['username'], password=server['password'])
        with sftp.cd(remote_file_dir):
            sftp.get(remote_file_name, local_file_path)
        sftp.close()
    return local_file_path


def main():
    server = {
        'host': '10.24.175.99',
        'username': 'haiyang',
        'password': 'haiyang'
    }
    fy4_l1_re = "FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_[0-9._]{29}_4000M_V0001.HDF"
    workspace = r"D:\WorkSpace\20200429\project\data\20200101"
    out_space = r"D:\WorkSpace\20200429\project\data\%Y%m%d"
    for dir_path, _, file_names in os.walk(workspace):
        for filename in file_names:
            ma = re.match(fy4_l1_re, filename)
            if ma:
                sdt_str = filename.split('_')[9]
                start_time_stamp = datetime.datetime.strptime(sdt_str, '%Y%m%d%H%M%S')
                for i in tqdm.trange(15, 25):
                    dt_c = start_time_stamp + datetime.timedelta(days=i)
                    try:
                        out_dir = dt_c.strftime(out_space)
                        os.makedirs(out_dir, exist_ok=True)
                        download_f4a_agri_l1_4km(dt_c, server, out_dir)
                        download_f4a_agri_geo_4km(dt_c, server, out_dir)
                        download_f4a_agri_clm_4km(dt_c, server, out_dir)
                    except Exception as e:
                        print(dt_c, e)


if __name__ == '__main__':
    main()
