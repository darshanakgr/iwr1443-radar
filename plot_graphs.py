import numpy as np
import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
from lib.plot import *

log_file = open("data.json", "r")

records = log_file.readlines()

heat_mode, heat_choice = ('rel', 'abs'), 0
comp_mode, comp_choice = ('lin', 'log'), 0

log2_10 = 20 * np.log10(2)

doppler_bins = 64
res_doppler = 0.0390

range_bins = 64
res_range = 0.125

comp_lin = 0.5
comp_log = 20 * np.log10(comp_lin)

log_lin = 0.0026041666666666665
range_bias = 0.08

fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)

fig.canvas.set_window_title("IWR1443")

ax1.set_title('Doppler-Range FFT Heatmap [{};{}]'.format(range_bins, doppler_bins), fontsize=10)
ax1.set_xlabel('Longitudinal distance [m]')
ax1.set_ylabel('Radial velocity [m/s]')

ax1.grid(color='white', linestyle=':', linewidth=0.5)

scale = max(doppler_bins, range_bins)
ratio = range_bins / doppler_bins

range_offset = res_range / 2
range_min = 0 - range_offset
range_max = range_min + scale * res_range

doppler_scale = scale // 2 * res_doppler
doppler_offset = 0  # (res_doppler * ratio) / 2
doppler_min = (-doppler_scale + doppler_offset) / ratio
doppler_max = (+doppler_scale + doppler_offset) / ratio

# fig.tight_layout(pad=2)

init_map = np.reshape([0, ] * range_bins * (doppler_bins - 1), (range_bins, doppler_bins - 1))
extent = [range_min - range_bias, range_max - range_bias, doppler_min, doppler_max]
aspect = (res_range / res_doppler) * ratio
im = ax1.imshow(init_map, cmap=plt.cm.jet, interpolation='quadric', aspect=aspect, extent=extent, alpha=.95)
ax1.plot([0 - range_bias, range_max - range_bias], [0, 0], color='white', linestyle=':', linewidth=0.5, zorder=1)

# plot for range profile
series = []

ax2 = plt.subplot(1, 2, 2)  # rows, cols, idx

ax2.set_title('Range Profile'.format(), fontsize=10)

ax2.set_xlabel('Distance [m]')
ax2.set_ylabel('Relative power [dB]')

ax2.set_xlim([0, range_max])

if int(range_max * 100) in (2250, 4093):  # high acc lab
    ax2.set_ylim([-5, 105])
    ax2.set_yticks(range(0, 100 + 1, 10))

elif int(range_max * 100) > 10000:  # 4k ffk
    ax2.set_ylim([0, 160])
    ax2.set_yticks(range(0, 160 + 1, 10))
    ax2.set_xticks(range(0, int(range_max) + 5, 10))

else:  # standard
    ax2.set_ylim([0, 120])
    ax2.set_yticks(range(0, 120 + 1, 10))

fig.tight_layout(pad=2)

ax2.plot([], [])
ax2.grid(linestyle=':')

# Plotting data


for record in records:
    data = json.loads(record)

    if 'range_doppler' not in data or len(data['range_doppler']) != range_bins * doppler_bins:
        continue

    a = np.array(data['range_doppler'])

    if comp_mode[comp_choice] == 'lin':
        a = comp_lin * 2 ** (a * log_lin)

    elif comp_mode[comp_choice] == 'log':
        a = a * log2_10 * log_lin + comp_log

    b = np.reshape(a, (range_bins, doppler_bins))
    c = np.fft.fftshift(b, axes=(1,))  # put left to center, put center to right

    im.set_array(c[:, 1:].T)  # rotate 90 degrees, cut off first doppler bin

    if heat_mode[heat_choice] == 'rel':
        im.autoscale()  # reset colormap

    elif heat_mode[heat_choice] == 'abs':
        im.set_clim(0, 1024 ** 2)  # reset colormap

    # range profile
    ax2.lines.clear()

    mpl.colors._colors_full_map.cache.clear()

    # for child in ax.get_children():
    #   if isinstance(child, mpl.collections.PathCollection):
    #        child.remove()

    if len(series) > 5:
        if series[0] is not None:
            series[0].remove()
        series.pop(0)

    x = None

    if 'range_profile' in data:
        y = data['range_profile']
        bin = range_max / len(y)
        x = [i * bin for i in range(len(y))]
        x = [v - range_bias for v in x]
        ax2.plot(x, y, color='blue', linewidth=0.75)

        if 'detected_points' in data:
            a = {}
            for p in data['detected_points']:
                ri, _ = (int(v) for v in p.split(','))
                if ri not in a: a[ri] = {'x': x[ri], 'y': y[ri], 's': 2}
                a[ri]['s'] += 1
            xo = [a[k]['x'] for k in a]
            yo = [a[k]['y'] for k in a]
            so = [a[k]['s'] for k in a]
            path = ax2.scatter(xo, yo, c='red', s=so, alpha=0.5)
            series.append(path)
        else:
            series.append(None)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.show(block=True)
