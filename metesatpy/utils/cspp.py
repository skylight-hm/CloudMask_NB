import xarray as xr
import numpy as np


def extract_from_cspp_nc(cspp_nc_path):
    cspp_ds = xr.open_dataset(cspp_nc_path)
    # coordinate
    cspp_cls = np.char.strip(cspp_ds['classifier_names'].data.astype(np.str))
    cspp_sft = np.char.strip(cspp_ds['sfc_type_names'].data.astype(np.str))
    cspp_bds = cspp_ds['n_bounds'].data
    # data
    cspp_bs = cspp_ds['bin_start'].data
    cspp_be = cspp_ds['bin_end'].data
    cspp_db = cspp_ds['delta_bin'].data

    cspp_ccrr = cspp_ds['class_cond_ratio_reg'].data
    # bin
    bin_start = xr.DataArray(cspp_bs, coords=[cspp_sft, cspp_cls],
                             dims=['cspp_sft', 'cspp_cls'])
    bin_end = xr.DataArray(cspp_be, coords=[cspp_sft, cspp_cls],
                           dims=['cspp_sft', 'cspp_cls'])
    delta_bin = xr.DataArray(cspp_db, coords=[cspp_sft, cspp_cls],
                             dims=['cspp_sft', 'cspp_cls'])
    # key var
    class_cond_ratio_reg = xr.DataArray(cspp_ccrr,
                                        coords=[cspp_sft, cspp_cls, cspp_bds],
                                        dims=['cspp_sft', 'cspp_cls', 'cspp_bds'])
    cspp_bins = np.linspace(bin_start,
                            bin_end,
                            len(cspp_bds))
    cspp_bins = np.transpose(cspp_bins, (1, 2, 0))
    bins = xr.DataArray(cspp_bins,
                        coords=[cspp_sft, cspp_cls, cspp_bds],
                        dims=['cspp_sft', 'cspp_cls', 'cspp_bds'])
    cspp_lut = xr.Dataset({
        'bin_start': bin_start,
        'bin_end': bin_end,
        'delta_bin': delta_bin,
        'bins': bins,
        'class_cond_ratio_reg': class_cond_ratio_reg
    })
    return cspp_lut
