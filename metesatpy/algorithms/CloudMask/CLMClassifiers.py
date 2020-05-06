import numpy as np

from metesatpy.production import FY4AAGRIL1FDIDISK4KM


class Ref063Min3x3Day(object):

    def __init__(self):
        super(Ref063Min3x3Day, self).__init__()

    def prepare_input(self, fy4_l1: FY4AAGRIL1FDIDISK4KM):
        ref_065 = fy4_l1.get_band_by_channel('ref_065')
