import xarray as xr
import numpy as np

dtor = np.pi / 180


def extract_from_cspp_nc(cspp_nc_path):
    cspp_ds = xr.open_dataset(cspp_nc_path)
    # coordinate
    cspp_cls = np.char.strip(cspp_ds['classifier_names'].data.astype(np.str))
    cspp_sft = np.char.strip(cspp_ds['sfc_type_names'].data.astype(np.str))
    cspp_bds = cspp_ds['n_bounds'].data
    # data
    cspp_py = cspp_ds['prior_yes'].data
    cspp_bs = cspp_ds['bin_start'].data
    cspp_be = cspp_ds['bin_end'].data
    cspp_db = cspp_ds['delta_bin'].data

    cspp_ccrr = cspp_ds['class_cond_ratio_reg'].data
    # bin
    prior_yes = xr.DataArray(cspp_py, coords=[cspp_sft],
                             dims=['cspp_sft'])
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
        'prior_yes': prior_yes,
        'bin_start': bin_start,
        'bin_end': bin_end,
        'delta_bin': delta_bin,
        'bins': bins,
        'class_cond_ratio_reg': class_cond_ratio_reg
    })
    return cspp_lut


def infer_airmass(sat_zen, sol_zen):
    cos_zen = np.cos(dtor * sat_zen)
    cos_solar_zen = np.cos(dtor * sol_zen)
    air_mass = 1.0 / cos_solar_zen + 1.0 / cos_zen
    return air_mass


def infer_great_circle(lat, lon, sat_lat=None, sat_lon=None):
    sat_lat = lat[lat.shape[0] // 2, lat.shape[1] // 2] if sat_lat is None else sat_lat
    sat_lon = lon[lon.shape[0] // 2, lon.shape[1] // 2] if sat_lon is None else sat_lon

    cos_geo = np.cos(lat * dtor) * np.cos(sat_lat * dtor) * np.cos((lon - sat_lon) * dtor) + np.sin(
        lat * dtor) * np.sin(sat_lat * dtor)
    cos_geo = np.clip(cos_geo, -1, 1)
    geo_x = np.arccos(cos_geo) / dtor
    return geo_x


def infer_relative_azimuth(geo_x, sol_zen, sat_sol_zen=None):
    sat_sol_zen = sol_zen[sol_zen.shape[0] // 2, sol_zen.shape[1] // 2] if sat_sol_zen is None else sat_sol_zen
    cos_geo = np.cos(geo_x * dtor)
    cos_geo = np.clip(cos_geo, -1, 1)
    cossolzen_pix = np.cos(sol_zen * dtor)
    numor = cos_geo * cossolzen_pix - np.cos(sat_sol_zen * dtor)
    denom = np.sin(geo_x * dtor) * np.sqrt(1.0 - cossolzen_pix ** 2)
    psix = np.where(denom == 0.0, 1, numor / denom)
    psix = np.clip(psix, -1, 1)
    rel_az = np.arccos(psix) / dtor
    return rel_az


def infer_scat_angle(sol_zen, sen_zen, rel_az):
    scattering_angle = -1.0 * np.cos(sol_zen * dtor) * np.cos(sen_zen * dtor) - \
                       np.sin(sol_zen * dtor) * np.sin(sen_zen * dtor) * np.cos(rel_az * dtor)
    scattering_angle = np.clip(scattering_angle, -1, 1)
    scattering_angle = np.arccos(scattering_angle) / dtor
    return scattering_angle
