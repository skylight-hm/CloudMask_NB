import numexpr as ne

import h5py
import numpy as np


class NaiveBayes(object):

    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.classifier_list = [

        ]
        self.ET_Land_class = {
            'SHALLOW_OCEAN': 0,
            'LAND': 1,
            'COASTLINE': 2,
            'SHALLOW_INLAND_WATER': 3,
            'EPHEMERAL_WATER': 4,
            'DEEP_INLAND_WATER': 5,
            'MODERATE_OCEAN': 6,
            'DEEP_OCEAN': 7,
        }
        self.ET_Snow_class = {
            'NO_SNOW': 1,
            'SEA_ICE': 2,
            'SNOW': 3,
        }
        self.ET_Desert_class = {
            'NO_DESERT': 0,
            'NIR_DESERT': 1,
            'BRIGHT_DESERT': 2,
        }
        self.ET_SFT_class = {
            'Space': 0,
            'DeepOcean': 1,
            'ShallowOcean': 2,
            'UnfrozenLand': 3,
            'SnowLand': 4,
            'Arctic': 5,
            'Antarctic': 6,
            'Desert': 7,
        }

    def prepare_surface_type_from_fy4a_nav(self, fy4_nav_file_path: str):
        nav_f = h5py.File(fy4_nav_file_path)
        lat = nav_f['pixel_latitude'][...]
        lon = nav_f['pixel_longitude'][...]
        land_mask = nav_f['pixel_land_mask'][...]
        snow_mask = nav_f['pixel_snow_mask'][...]
        desert_mask = nav_f['pixel_desert_mask'][...]  # two way :1. CSPP(emiss) 2. fy4
        nb_sft = np.zeros(land_mask.shape, np.uint8)

        # DeepOcean
        nb_sft[land_mask == self.ET_Land_class['DEEP_OCEAN']] = self.ET_SFT_class['DeepOcean']
        # ShallowOcean
        nb_sft[land_mask == self.ET_Land_class['MODERATE_OCEAN']] = self.ET_SFT_class['ShallowOcean']
        nb_sft[land_mask == self.ET_Land_class['DEEP_INLAND_WATER']] = self.ET_SFT_class['ShallowOcean']
        nb_sft[land_mask == self.ET_Land_class['SHALLOW_INLAND_WATER']] = self.ET_SFT_class['ShallowOcean']
        nb_sft[land_mask == self.ET_Land_class['SHALLOW_OCEAN']] = self.ET_SFT_class['ShallowOcean']
        # UnfrozenLand
        nb_sft[land_mask == self.ET_Land_class['LAND']] = self.ET_SFT_class['UnfrozenLand']
        nb_sft[land_mask == self.ET_Land_class['COASTLINE']] = self.ET_SFT_class['UnfrozenLand']
        nb_sft[land_mask == self.ET_Land_class['EPHEMERAL_WATER']] = self.ET_SFT_class['UnfrozenLand']
        # SnowLand
        sl_idx = np.logical_and(lat > -60, snow_mask == self.ET_Snow_class['SNOW'])
        nb_sft[sl_idx] = self.ET_SFT_class['SnowLand']
        # Arctic
        arc_idx = np.logical_and(lat >= 0, snow_mask == self.ET_Snow_class['SEA_ICE'])
        nb_sft[arc_idx] = self.ET_SFT_class['Arctic']
        # Antarctic # no greenland
        ant_idx1 = np.logical_and(lat <= -60, snow_mask == self.ET_Snow_class['SNOW'])
        ant_idx2 = np.logical_and(lat <= -60, snow_mask == self.ET_Snow_class['SEA_ICE'])
        nb_sft[ant_idx1] = self.ET_SFT_class['Antarctic']
        nb_sft[ant_idx2] = self.ET_SFT_class['Antarctic']
        # Desert # fy4 way
        desert_idx = np.logical_or(desert_mask == self.ET_Desert_class['NIR_DESERT'],
                                   desert_mask == self.ET_Desert_class['BRIGHT_DESERT'])
        nb_sft[desert_idx] = self.ET_SFT_class['Desert']
        return nb_sft

    def prepare_geos(self, lut_nc):
        lut_f = h5py.File(lut_nc, 'r')

    def prepare_obs(self):
        pass

    def band_statistic(self):
        pass


