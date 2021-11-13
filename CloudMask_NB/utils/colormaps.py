from collections import namedtuple

import numpy as np
from matplotlib import colors, colorbar, pyplot as plt

ColorValue = namedtuple('ColorValue', ['name', 'plt_name', 'value'])


class ColorMap(object):
    def __init__(self):
        self._init_color()

    def _init_color(self):
        raise NotImplementedError

    def plot(self, ax=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.cs)

    @property
    def param_dict(self):
        raise NotImplementedError


class CouldMaskPure(ColorMap):
    def _init_color(self):
        c1 = ColorValue('Cloudy', 'white', 0)
        c2 = ColorValue('ProbCloudy', 'darkgray', 1)
        c3 = ColorValue('ProbClear', 'lightsteelblue', 2)
        c4 = ColorValue('Clear', 'blue', 3)
        c5 = ColorValue('Invalid', 'black', 4)
        self.cs = [c1, c2, c3, c4, c5]
        self.cs.sort(key=lambda x: x.value)
        self.color_map = colors.ListedColormap([i.plt_name for i in self.cs])
        self.bounds = [i.value + 0.5 for i in self.cs]
        self.bounds.insert(0, self.bounds[0] - 1)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 1))
            fig.subplots_adjust(bottom=0.5)
        norm = colors.BoundaryNorm(self.bounds, self.color_map.N)
        cb = colorbar.ColorbarBase(
            ax,
            cmap=self.color_map,
            norm=norm,
            extend='neither',
            orientation='horizontal')

        cb.set_ticks(np.linspace(self.cs[0].value, self.cs[-1].value, len(self)))  # remove the color tick
        cb.ax.tick_params(labelsize=8)
        cb.set_ticklabels([i.name for i in self.cs])
        return ax

    @property
    def vmax(self):
        return self.cs[-1].value

    @property
    def vmin(self):
        return self.cs[0].value

    @property
    def cmap(self):
        return self.color_map

    @property
    def param_dict(self):
        return {'cmap': self.cmap, 'vmax': self.vmax, 'vmin': self.vmin}
