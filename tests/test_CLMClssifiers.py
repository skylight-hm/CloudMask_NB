import unittest

import os

from metesatpy.production import FY4AAGRIL1FDIDISK4KM, FY4AAGRIL1GEODISK4KM
from metesatpy.production.FY4A import FY4NavFile
from metesatpy.algorithms.CloudMask import Ref063Min3x3Day

import matplotlib.pyplot as plt

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')


class TestCLMClassifiers(unittest.TestCase):
    def test_Ref063Min3x3Day(self):
        agri_l1_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_' \
                            'NOM_20200101121500_20200101122959_4000M_V0001.HDF'
        agri_l1_file_path = os.path.join(data_root_dir, agri_l1_file_name)

        agri_geo_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_' \
                             'NOM_20200101121500_20200101122959_4000M_V0001.HDF'
        agri_geo_file_path = os.path.join(data_root_dir, agri_geo_file_name)

        fy4_nav_file_name = 'fygatNAV.FengYun-4A.xxxxxxx.4km.hdf'
        fy4_nav_file_path = os.path.join(data_root_dir, fy4_nav_file_name)

        lut_nc_path = ''
        lut_file_name = 'modis_default_nb_cloud_mask_lut.nc'
        lut_file_path = os.path.join(data_root_dir, 'LUT', lut_file_name)

        l1 = FY4AAGRIL1FDIDISK4KM(agri_l1_file_path)
        geo = FY4AAGRIL1GEODISK4KM(agri_geo_file_path)
        nav = FY4NavFile(fy4_nav_file_path)

        ref_036 = Ref063Min3x3Day(lut_nc_path=lut_file_path)
        ratio = ref_036.infer(l1, geo, nav)
        plt.imshow(ratio)
        plt.show()


if __name__ == '__main__':
    unittest.main()
