from ..CLM import FY4AAGRICLM

import os
import datetime
import traceback

import h5py
import numpy as np
import matplotlib.pyplot as plt

from metesatpy.utils.statistic import pod, far, kss, hr


class FY4AAGRICLM4KM(FY4AAGRICLM):

    def __init__(self, fname: str = None, **kwargs):
        super(FY4AAGRICLM4KM, self).__init__()
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
        return "FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_[0-9._]{29}_4000M_V0001.NC"

    def get_clm(self):
        try:
            f = h5py.File(self.fname, 'r')
            data = f['CLM'][...]
            data = np.ma.masked_values(data, 126)
            data[data == 127] = 4
            f.close()
            return data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def plot(self, cmap=None, color_bar_axes: bool = True, return_axes: bool = False, title: str = None):
        fig = plt.figure(figsize=(10, 10))
        axs = []
        main_axes = fig.add_subplot(1, 1, 1)
        array = self.get_clm()
        params = {}
        if cmap is None:
            from metesatpy.painters.colormaps.L2 import CouldMaskPure
            cmc = CouldMaskPure()
            params = cmc.param_dict
        main_axes.imshow(array, **params)
        # main_axes.set_axis_off()
        main_axes.set_title(title, fontsize=16)
        axs.append(main_axes)
        if color_bar_axes:
            from metesatpy.painters.colormaps.L2 import CouldMaskPure
            color_ax = fig.add_axes([0.1, 0.03, 0.83, 0.05])
            cmc = CouldMaskPure()
            color_ax = cmc.plot(color_ax)
            axs.append(color_ax)

        return_content = fig
        if return_axes:
            return_content = (fig, axs)
        return return_content

    def delta_q(self, other_clm_array: np.ma.masked_array):
        clm_array = self.get_clm()
        d_flag = clm_array <= 3
        cm_delta_q_v = np.zeros(d_flag.shape, dtype=np.uint8)  # invalid
        cm_delta_q = clm_array - other_clm_array

        cloudy_shift_idx = np.logical_and(cm_delta_q <= -2.0, d_flag)
        cm_delta_q_v[cloudy_shift_idx] = 1  # cloudy shift

        small_cloudy_shift_idx = np.logical_and(cm_delta_q == -1, d_flag)
        cm_delta_q_v[small_cloudy_shift_idx] = 2  # small cloudy shift

        no_shift_idx = np.logical_and(cm_delta_q == 0, d_flag)
        cm_delta_q_v[no_shift_idx] = 3  # no shift

        small_clear_shift_idx = np.logical_and(cm_delta_q == 1, d_flag)
        cm_delta_q_v[small_clear_shift_idx] = 4  # small cloudy shift

        clear_shift_idx = np.logical_and(cm_delta_q >= 2.0, d_flag)
        cm_delta_q_v[clear_shift_idx] = 5  # clear shift
        cm_delta_q_v_m = np.ma.masked_array(cm_delta_q_v, ~d_flag)
        return cm_delta_q_v_m

    @classmethod
    def confusion(cls, y_true: np.ma.masked_array, y_pred: np.ma.masked_array):
        a = np.logical_and(y_true == 0, y_pred == 0).sum()
        b = np.logical_and(y_true == 3, y_pred == 0).sum()
        c = np.logical_and(y_true == 3, y_pred == 0).sum()
        d = np.logical_and(y_true == 3, y_pred == 3).sum()
        return a, b, c, d

    def compare_metric(self, other_clm_array: np.ma.masked_array):
        clm_array = self.get_clm()
        a, b, c, d = self.confusion(clm_array, other_clm_array)
        pod_cloudy, pod_clear = pod(a, b, c, d)
        far_cloudy, far_clear = far(a, b, c, d)
        hr_m = hr(a, b, c, d)
        kss_m = kss(a, b, c, d)
        result = {'a': int(a), 'b': int(b),
                  'c': int(c), 'd': int(d),
                  'pod_cloudy': pod_cloudy,
                  'pod_clear': pod_clear,
                  'far_cloudy': far_cloudy,
                  'far_clear': far_clear,
                  'hr': hr_m,
                  'kss': kss_m
                  }
        return result
