import numpy as np
import h5py


class ProductionBase(object):
    fname: str = None

    def __init__(self):
        super(ProductionBase, self).__init__()

    def _decorate_ds_data(self, ds: h5py.Dataset, masked=True) -> np.ma.masked_array:
        slope = ds.attrs.get('Slope', 0)
        inter = ds.attrs.get('Intercept', 0)
        fill = ds.attrs.get('FillValue', 65535)
        array = ds[...]
        array = slope * array + inter
        if masked:
            array = np.ma.masked_values(array, fill)
        else:
            array[array == fill] = np.nan
        return array
