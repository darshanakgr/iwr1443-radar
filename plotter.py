import matplotlib.widgets as wgt
from lib.plot import *
from itertools import product, combinations  # a small cube (origin)
import matplotlib.patches as pat
import scipy.interpolate as spi

# init parameters

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

scale = max(doppler_bins, range_bins)
ratio = range_bins / doppler_bins

range_offset = res_range / 2
range_min = 0 - range_offset
range_max = range_min + scale * res_range

doppler_scale = scale // 2 * res_doppler
doppler_offset = 0  # (res_doppler * ratio) / 2
doppler_min = (-doppler_scale + doppler_offset) / ratio
doppler_max = (+doppler_scale + doppler_offset) / ratio

tx_azimuth_antennas = 2
rx_antennas = 4
angle_bins = 64

t = np.array(range(-angle_bins // 2 + 1, angle_bins // 2)) * (2 / angle_bins)
t = np.arcsin(t)  # t * ((1 + np.sqrt(5)) / 2)
r = np.array(range(range_bins)) * res_range

range_depth = range_bins * res_range
range_width, grid_res = range_depth / 2, 400

xi = np.linspace(-range_width, range_width, grid_res)
yi = np.linspace(0, range_depth, grid_res)
xi, yi = np.meshgrid(xi, yi)

x_azimuth = np.array([r]).T * np.sin(t)
y_azimuth = np.array([r]).T * np.cos(t)
y_azimuth = y_azimuth - range_bias

# array to save the data from range profile
series = []

# init figure
fig = plt.figure(figsize=(12, 12))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3, projection='3d')
ax = plt.subplot(2, 2, 4)  # rows, cols, idx

fig.canvas.set_window_title("IWR1443")

ax1.set_title('Doppler-Range FFT Heatmap [{};{}]'.format(range_bins, doppler_bins), fontsize=10)
ax1.set_xlabel('Longitudinal distance [m]')
ax1.set_ylabel('Radial velocity [m/s]')
ax1.grid(color='white', linestyle=':', linewidth=0.5)

init_map = np.reshape([0, ] * range_bins * (doppler_bins - 1), (range_bins, doppler_bins - 1))
extent = [range_min - range_bias, range_max - range_bias, doppler_min, doppler_max]
aspect = (res_range / res_doppler) * ratio
im = ax1.imshow(init_map, cmap=plt.cm.jet, interpolation='quadric', aspect=aspect, extent=extent, alpha=.95)
ax1.plot([0 - range_bias, range_max - range_bias], [0, 0], color='white', linestyle=':', linewidth=0.5, zorder=1)

# Range profile plot
ax2.set_title('Range Profile'.format(), fontsize=10)
ax2.set_xlabel('Distance [m]')
ax2.set_ylabel('Relative power [dB]')
ax2.set_xlim([0, range_max])
ax2.plot([], [])
ax2.grid(linestyle=':')

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

# detected objects plot
ax3.view_init(azim=-45, elev=15)
ax3.set_title('CFAR Detection'.format(), fontsize=10)

ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')
ax3.set_zlabel('z [m]')

ax3.set_xlim3d((-range_max / 2, +range_max / 2))
ax3.set_ylim3d((0, range_max))
ax3.set_zlim3d((-range_max / 2, +range_max / 2))

ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

ax3.xaxis._axinfo['grid']['linestyle'] = ':'
ax3.yaxis._axinfo['grid']['linestyle'] = ':'
ax3.zaxis._axinfo['grid']['linestyle'] = ':'

ax3.scatter(xs=[], ys=[], zs=[], marker='.', cmap='jet')

for child in ax3.get_children():
    if isinstance(child, art3d.Path3DCollection):
        child.remove()

r = [-0.075, +0.075]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax3.plot3D(*zip(s, e), color="black", linewidth=0.5)

set_aspect_equal_3d(ax3)

# range azimuth heat map
cm = ax.imshow(((0,) * grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)

ax.set_title('Azimuth-Range FFT Heatmap [{};{}]'.format(angle_bins, range_bins), fontsize=10)
ax.set_xlabel('Lateral distance along [m]')
ax.set_ylabel('Longitudinal distance along [m]')

ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)

ax.set_ylim([0, +range_depth])
ax.set_xlim([-range_width, +range_width])

for i in range(1, int(range_depth) + 1):
    ax.add_patch(pat.Arc((0, 0), width=i * 2, height=i * 2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))

# show plot
fig.tight_layout(pad=2)
plt.show(block=False)


def update_plot(record):
    data = json.loads(record)

    if 'range_doppler' not in data or len(data['range_doppler']) != range_bins * doppler_bins:
        return

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

    # update range-profile plot
    ax2.lines.clear()
    mpl.colors._colors_full_map.cache.clear()

    if len(series) > 5:
        series.pop(0)

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

            # detected points plot
            xm, ym, zm = ax3.get_xlim3d(), ax3.get_ylim3d(), ax3.get_zlim3d()

            for _, p in data['detected_points'].items():

                x, y, z, d = p['x'], p['y'], p['z'], p['v']

                if xm[0] <= x <= xm[1] and ym[0] <= y <= ym[1] and zm[0] <= z <= zm[1]:

                    val = min(1.0, d / 65536)
                    val = 0.1 * np.log(val + 0.0001) + 1
                    val = max(0, min(val, 1))

                    pt = Point((x, y, z), color=(val, 0.0, 1 - val), size=3, marker='.')
                    ax3.add_artist(pt)

                    az, el = ax3.azim, ax3.elev

                    if abs(az) > 90:
                        x_ = max(xm)
                    else:
                        x_ = min(xm)

                    if az < 0:
                        y_ = max(ym)
                    else:
                        y_ = min(ym)

                    if el < 0:
                        z_ = max(zm)
                    else:
                        z_ = min(zm)

                    xz = Point((x, y_, z), color=(0.67, 0.67, 0.67), size=1, marker='.')
                    ax3.add_artist(xz)

                    yz = Point((x_, y, z), color=(0.67, 0.67, 0.67), size=1, marker='.')
                    ax3.add_artist(yz)

                    xy = Point((x, y, z_), color=(0.67, 0.67, 0.67), size=1, marker='.')
                    ax3.add_artist(xy)

    # range azimuth plot update
    if 'azimuth_static' in data and len(data['azimuth_static']) == range_bins * tx_azimuth_antennas * rx_antennas * 2:
        a = data['azimuth_static']

        a = np.array([a[i] + 1j * a[i + 1] for i in range(0, len(a), 2)])
        a = np.reshape(a, (range_bins, tx_azimuth_antennas * rx_antennas))
        a = np.fft.fft(a, angle_bins)

        a = np.abs(a)
        a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right
        a = a[:, 1:]  # cut off first angle bin

        zi = spi.griddata((x_azimuth.ravel(), y_azimuth.ravel()), a.ravel(), (xi, yi), method='linear')
        zi = zi[:-1, :-1]

        cm.set_array(zi[::-1, ::-1])  # rotate 180 degrees
        if heat_mode[heat_choice] == 'rel':
            cm.autoscale()  # reset colormap
        elif heat_mode[heat_choice] == 'abs':
            cm.set_clim(0, 256 ** 2 // 2)  # reset colormap

    fig.canvas.draw()
    fig.canvas.flush_events()
