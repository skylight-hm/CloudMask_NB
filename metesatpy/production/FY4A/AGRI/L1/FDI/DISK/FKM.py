from .DISK import FY4AAGRIL1FDIDISKProduction

import os
import datetime
import traceback
from pprint import pprint
from dataclasses import dataclass

import h5py
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FY4AAGRIL1FDIDISKChannel(object):
    short_name: str
    center_wave_length: str
    data_ds_name: str
    cal_ds_name: str


class FY4AAGRIL1FDIDISK4KM(FY4AAGRIL1FDIDISKProduction):
    channel_table: dict = {
        'ref_047': FY4AAGRIL1FDIDISKChannel(short_name='ref_047',
                                            center_wave_length='0.47um',
                                            data_ds_name='NOMChannel01',
                                            cal_ds_name='CALChannel01'),
        'ref_065': FY4AAGRIL1FDIDISKChannel(short_name='ref_065',
                                            center_wave_length='0.65um',
                                            data_ds_name='NOMChannel02',
                                            cal_ds_name='CALChannel02'),
        'ref_083': FY4AAGRIL1FDIDISKChannel(short_name='ref_083',
                                            center_wave_length='0.83um',
                                            data_ds_name='NOMChannel03',
                                            cal_ds_name='CALChannel03'),
        'ref_137': FY4AAGRIL1FDIDISKChannel(short_name='ref_137',
                                            center_wave_length='1.37um',
                                            data_ds_name='NOMChannel04',
                                            cal_ds_name='CALChannel04'),
        'ref_161': FY4AAGRIL1FDIDISKChannel(short_name='ref_161',
                                            center_wave_length='1.61um',
                                            data_ds_name='NOMChannel05',
                                            cal_ds_name='CALChannel05'),
        'ref_222': FY4AAGRIL1FDIDISKChannel(short_name='ref_222',
                                            center_wave_length='2.22um',
                                            data_ds_name='NOMChannel06',
                                            cal_ds_name='CALChannel06'),
        'bt_372_low': FY4AAGRIL1FDIDISKChannel(short_name='bt_372_low',
                                               center_wave_length='3.72um',
                                               data_ds_name='NOMChannel07',
                                               cal_ds_name='CALChannel07'),
        'bt_372_high': FY4AAGRIL1FDIDISKChannel(short_name='bt_372_high',
                                                center_wave_length='3.72um',
                                                data_ds_name='NOMChannel08',
                                                cal_ds_name='CALChannel08'),
        'bt_625': FY4AAGRIL1FDIDISKChannel(short_name='bt_625',
                                           center_wave_length='6.25um',
                                           data_ds_name='NOMChannel09',
                                           cal_ds_name='CALChannel09'),
        'bt_710': FY4AAGRIL1FDIDISKChannel(short_name='bt_710',
                                           center_wave_length='7.10um',
                                           data_ds_name='NOMChannel10',
                                           cal_ds_name='CALChannel10'),
        'bt_850': FY4AAGRIL1FDIDISKChannel(short_name='bt_850',
                                           center_wave_length='8.50um',
                                           data_ds_name='NOMChannel11',
                                           cal_ds_name='CALChannel11'),
        'bt_1080': FY4AAGRIL1FDIDISKChannel(short_name='bt_1080',
                                            center_wave_length='10.80um',
                                            data_ds_name='NOMChannel12',
                                            cal_ds_name='CALChannel12'),
        'bt_1200': FY4AAGRIL1FDIDISKChannel(short_name='bt_1200',
                                            center_wave_length='12.00um',
                                            data_ds_name='NOMChannel13',
                                            cal_ds_name='CALChannel13'),
        'bt_1350': FY4AAGRIL1FDIDISKChannel(short_name='bt_1350',
                                            center_wave_length='13.50um',
                                            data_ds_name='NOMChannel14',
                                            cal_ds_name='CALChannel14')
    }

    def __init__(self, fname: str = None, **kwargs):
        super(FY4AAGRIL1FDIDISK4KM, self).__init__()
        try:
            self.fname = fname
            self.fdir = os.path.dirname(fname)
            self.fbname = os.path.basename(fname)
            sdt_str = self.fbname.split('_')[9]
            edt_str = self.fbname.split('_')[10]
            self.start_time_stamp = datetime.datetime.strptime(sdt_str, '%Y%m%d%H%M%S')
            self.end_time_stamp = datetime.datetime.strptime(edt_str, '%Y%m%d%H%M%S')
        except Exception as e:
            print(e)

    @property
    def file_name_pattern(self):
        return "FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_[0-9._]{29}_4000M_V0001.HDF"

    def get_channel(self, name: str, **kwargs) -> FY4AAGRIL1FDIDISKChannel:
        return self.channel_table[name]

    def print_available_channels(self):
        pprint(self.channel_table)

    def get_data_by_name(self, name: str, **kwargs) -> np.ma.masked_array:
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f[name])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def set_band(self, name: str, **kwargs) -> int:
        pass

    def get_band_by_channel(self, name: str, **kwargs) -> np.ma.masked_array:
        try:
            f = h5py.File(self.fname, 'r')
            channel_cursor = self.channel_table[name]
            # idx data set name
            idx_ds_name = channel_cursor.data_ds_name
            idx_data = self._decorate_ds_data(f[idx_ds_name])
            # cal data set name
            cal_ds_name = channel_cursor.cal_ds_name
            cal_data = self._decorate_ds_data(f[cal_ds_name])
            idx_data[~idx_data.mask] = cal_data[idx_data[~idx_data.mask].astype(np.int)]
            f.close()
            return idx_data
        except Exception as e:
            print(e, self.fname)
            traceback.print_exc()

    def get_band(self, name: str) -> np.ndarray:
        pass

    def export(self, fname: str) -> int:
        return 0

    def plot(self, **kwargs):
        plot_type = kwargs.get('flag', 'vis')
        if plot_type == 'vis':
            r = self.get_band_by_channel('ref_047')
            g = self.get_band_by_channel('ref_065')
            b = self.get_band_by_channel('ref_083')
            img = np.dstack((r, g, b))
            title = kwargs.get('title', 'visual color\n' + os.path.basename(self.fname))
            fig, ax = plt.subplots(1, 1)
            pos = ax.imshow(img, 'gray')
            ax.set_title(title)
            fig.colorbar(pos, ax=ax)
        elif plot_type == 'ir':
            dn = self.get_band_by_channel('bt_625')
            img = dn
            title = kwargs.get('title', 'bt dn\n' + os.path.basename(self.fname))
            fig, ax = plt.subplots(1, 1)
            pos = ax.imshow(img, 'gray')
            ax.set_title(title)
            fig.colorbar(pos, ax=ax)
        else:
            img = None
            return None
        return fig
