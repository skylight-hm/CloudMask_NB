import unittest

import os

import matplotlib.pyplot as plt

from metesatpy.production import FY4AAGRIL1FDIDISK4KM
from metesatpy.production.FY4A import FY4NavFile

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')


class TestFY4AAGRI(unittest.TestCase):

    def setUp(self) -> None:
        agri_l1_file_name = 'FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_' \
                            'NOM_20200101121500_20200101122959_4000M_V0001.HDF'
        self.agri_l1_file_path = os.path.join(data_root_dir, agri_l1_file_name)

    def test_get_band_by_channel(self) -> None:
        l1 = FY4AAGRIL1FDIDISK4KM(self.agri_l1_file_path)
        array = l1.get_band_by_channel('ref_065')
        plt.imshow(array)
        plt.show()

    def test_print_available_channels(self) -> None:
        l1 = FY4AAGRIL1FDIDISK4KM(self.agri_l1_file_path)
        l1.print_available_channels()

    def test_plot(self) -> None:
        l1 = FY4AAGRIL1FDIDISK4KM(self.agri_l1_file_path)
        l1.plot()
        plt.show()

    def test_plot_ir(self) -> None:
        l1 = FY4AAGRIL1FDIDISK4KM(self.agri_l1_file_path)
        l1.plot(flag='ir')
        plt.show()

    def tearDown(self) -> None:
        pass


class TestFY4ANav(unittest.TestCase):

    def test_surface_prepare(self):
        nav_file_name = 'fygatNAV.FengYun-4A.xxxxxxx.4km.hdf'
        nav_file_path = os.path.join(data_root_dir, nav_file_name)
        fy4_nav = FY4NavFile(nav_file_path)
        sft_nb = fy4_nav.prepare_surface_type_to_cspp()
        plt.imshow(sft_nb)
        plt.show()


if __name__ == '__main__':
    unittest.main()
