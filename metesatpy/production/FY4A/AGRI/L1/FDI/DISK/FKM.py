from .DISK import FY4AAGRIL1FDIDISKProduction

import traceback
import h5py
import numpy as np


class FY4AAGRIL1FDIDISKChannel(object):
    short_name = None
    center_wave_length = None
    data_ds_name = None
    cal_ds_name = None

    def __init__(self, short_name: str, center_wave_length: str, data_ds_name: str, cal_ds_name: str):
        super(FY4AAGRIL1FDIDISKChannel, self).__init__()
        self.short_name = short_name
        self.center_wave_length = center_wave_length
        self.data_ds_name = data_ds_name
        self.cal_ds_name = cal_ds_name

    def __repr__(self):
        return {
            'short_name': self.short_name,
            'center_wave_length': self.center_wave_length,
            'data_ds_name': self.data_ds_name,
            'cal_ds_name': self.cal_ds_name
        }

    def __eq__(self, other):
        return other.short_name == self.short_name


class FY4AAGRIL1FDIDISK4KM(FY4AAGRIL1FDIDISKProduction):
    filepath = None

    def __init__(self, fname: str, **kwargs):
        super(FY4AAGRIL1FDIDISK4KM, self).__init__()
        self.fname = fname

        self.channel_table = {
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
        }

    def get_channel(self, name: str, **kwargs) -> FY4AAGRIL1FDIDISKChannel:
        return self.channel_table[name]

    def print_available_channels(self):
        print(self.channel_table)

    def get_data_by_name(self, name: str, **kwargs):
        try:
            f = h5py.File(self.fname, 'r')
            data = self._decorate_ds_data(f[name])
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def _decorate_ds_data(self, ds):
        slope = ds.attrs.get('Slope', 0)
        inter = ds.attrs.get('Intercept', 0)
        fill = ds.attrs.get('FillValue', 65535)
        array = ds[...]
        array = slope * array + inter
        array = np.ma.masked_values(array, fill)
        return array

    def set_band(self, name: str, **kwargs) -> np.ndarray:
        pass

    def get_band_by_channel(self, name: str, **kwargs) -> np.ndarray:
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
            print(e)
            traceback.print_exc()

    def get_band(self, name: str) -> np.ndarray:
        pass

    def export(self, fname: str) -> int:
        return 0
