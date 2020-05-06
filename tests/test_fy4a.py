import unittest

import os

from metesatpy.production import FY4AAGRIL1FDIDISK4KM

import matplotlib.pyplot as plt

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

    def test_(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
