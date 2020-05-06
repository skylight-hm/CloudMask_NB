import unittest

import os

from production import FY4AAGRIL1FDIDISK4KM

import matplotlib.pyplot as plt
from metesatpy.algorithms.CloudMask import NaiveBayes

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')


class TestNaiveBayes(unittest.TestCase):
    def test_something(self):
        def setUp(self) -> None:
            agri_l1_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_' \
                                'NOM_20200101121500_20200101122959_4000M_V0001.HDF'
            self.agri_l1_file_path = os.path.join(data_root_dir, agri_l1_file_name)
            lut_file_name = 'modis_default_nb_cloud_mask_lut.nc'
            self.lut_file_path = os.path.join(data_root_dir, 'LUT', lut_file_name)

    def test_surface_prepare(self):
        nav_file_name = 'fygatNAV.FengYun-4A.xxxxxxx.4km.hdf'
        nav_file_path = os.path.join(data_root_dir, nav_file_name)
        nb = NaiveBayes()
        sft_nb = nb.prepare_surface_type_from_fy4a_nav(nav_file_path)
        plt.imshow(sft_nb)
        plt.show()

    def test_obs_prepare(self):
        pass


if __name__ == '__main__':
    unittest.main()
