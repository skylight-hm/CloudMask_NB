import unittest
import os

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')

from metesatpy.utils.cspp import extract_from_cspp_nc


class TestUtils(unittest.TestCase):
    def test_cspp_lut_convert(self):
        cspp_lut_nc = os.path.join(data_root_dir, 'LUT', 'modis_default_nb_cloud_mask_lut.nc')
        cspp_lut_ds = extract_from_cspp_nc(cspp_lut_nc)



if __name__ == '__main__':
    unittest.main()
