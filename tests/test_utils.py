import unittest
import os

import xarray as xr

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')

from metesatpy.utils.cspp import extract_from_cspp_nc


class TestUtils(unittest.TestCase):
    def test_cspp_lut_convert(self):
        cspp_lut_nc = os.path.join(data_root_dir, 'LUT', 'modis_default_nb_cloud_mask_lut.nc')
        cspp_lut_ds = extract_from_cspp_nc(cspp_lut_nc)
        for i in cspp_lut_ds.cspp_cls.data.tolist():
            py = cspp_lut_ds.prior_yes
            bs = cspp_lut_ds.bin_start.sel(cspp_cls=i)
            be = cspp_lut_ds.bin_end.sel(cspp_cls=i)
            db = cspp_lut_ds.delta_bin.sel(cspp_cls=i)
            bins = cspp_lut_ds.bins.sel(cspp_cls=i)
            ccrr = cspp_lut_ds.class_cond_ratio_reg.sel(cspp_cls=i)
            ds_c = xr.Dataset({
                'prior_yes': py,
                'bin_start': bs,
                'bin_end': be,
                'delta_bin': db,
                'bins': bins,
                'class_cond_ratio_reg': ccrr
            })
            ds_c.to_netcdf(r"D:\WorkSpace\20200429\project\data\LUT\%s.nc" % i)


if __name__ == '__main__':
    unittest.main()
