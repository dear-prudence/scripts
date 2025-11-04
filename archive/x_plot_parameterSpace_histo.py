import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm


def create_halfway_array(arr):
    # Calculate the halfway points between consecutive elements
    halfway_points = (arr[:-1] + arr[1:]) / 2
    return halfway_points


def get_data(filename):
    data = np.load(filename)
    return create_halfway_array(data['bin_edges']), data['data']


def translate_label(var):
    if var == 'distance':
        return 'Distance (kpc)'
    elif var == 'temperature':
        return 'log(Temperature)'
    else:
        pass


plot_type = 'histo'
filename = '/Users/ursa/Desktop/09_18_snap103_gas_tempVdistHist_LMC_radius200kpc_wMasses.npz'
ind_var = 'distance'
dep_var = 'temperature'
ind_label = 'Radius (kpc)'
dep_label = 'Temperature (K)'
cmap = 'GnBu'

if plot_type == 'histo':
    data = np.load(filename)
    # The data gets rotated by 90 degrees somehow when processing and transferring,
    # so the np.rot90 is added to correct for the inconsistency
    H = np.rot90(data['data'])
    x_edges = data['d_edges']
    y_edges = data['logT_edges']
    print(np.max(H))
    print((float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])))
    lookback_time = round(float(data['time'][0][1]), 2)
    background_color = plt.get_cmap(cmap)(0)

    plt.figure(figsize=(14, 8))
    plt.gca().set_facecolor(background_color)
    plt.imshow(H, extent=(float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])),
               aspect=(x_edges[-1] - x_edges[0]) / ((y_edges[-1] - y_edges[0]) * 2),
               norm=PowerNorm(gamma=0.1, vmin=0, vmax=1), cmap=cmap)
    # plt.plot(data['bin_edges'][:-1] * 1e3, data['Tcurve'], linestyle='-', label='Avg Temperature', c='k')
    plt.legend(loc='lower right')
    plt.title('Temperature vs distance (indiscriminate of direction) of gas particles '
              'of LMC (Run 09_18), $t=-$' + str(lookback_time) + ' Gyr', fontsize='small')
else:
    pass

plt.xlabel(translate_label(ind_var))
plt.ylabel(translate_label(dep_var))
plt.savefig('/Users/dear-prudence/Desktop/tempVdist_LMC_histo_snap103.png', dpi=240)
plt.show()

plt.close()

'''# files to load in
#data109 = get_data('/Users/dear-prudence/Desktop/09_18_snap109_gas_tempVdist_LMC_radius200kpc.npz')
#data112 = get_data('/Users/dear-prudence/Desktop/09_18_snap112_gas_tempVdist_LMC_radius200kpc.npz')
# data114 = get_data('/Users/dear-prudence/Desktop/09_18_snap114_gas_tempVdist_LMC_radius200kpc.npz')
#data118 = get_data('/Users/dear-prudence/Desktop/09_18_snap118_gas_tempVdist_LMC_radius200kpc.npz')
#data121 = get_data('/Users/dear-prudence/Desktop/09_18_snap121_gas_tempVdist_LMC_radius200kpc.npz')
# data = np.load('/Users/dear-prudence/Desktop/09_18_snap114_gas_tempVdistScatter_LMC_radius200kpc.npz')

plt.figure(figsize=(12, 8))
#plt.plot(data109[0], data109[1], linestyle='-', label='$t=-2.85$ Gyr', c=plt.get_cmap(cmap)(200))
#plt.plot(data112[0], data112[1], linestyle='-', label='$t=-2.40$ Gyr', c=plt.get_cmap(cmap)(160))
plt.plot(data114[0], data114[1], linestyle='-', label='Avg Temperature', c='k')
#plt.plot(data118[0], data118[1], linestyle='-', label='$t=-1.47$ Gyr', c=plt.get_cmap(cmap)(80))
#plt.plot(data121[0], data121[1], linestyle='-', label='$t=-0.99$ Gyr', c=plt.get_cmap(cmap)(40))
plt.scatter(data['x_data'] * 10 ** 3, data['y_data'], s=2, alpha=0.01)
print(len(data['x_data']))
plt.xlabel(ind_label)
plt.ylabel(dep_label)
plt.yscale('log')
plt.xlim(right=200, left=1)
plt.ylim(top=1 * 10 ** 7, bottom=1 * 10 ** 4)
plt.plot([150, 165], [10 ** 6, 1.5 * 10 ** 6], linestyle=':', c='gray')
plt.text(163, 1.55 * 10 ** 6, s='Corona', c='gray')
plt.plot([5, 8], [5 * 10 ** 4, 9 * 10 ** 4], linestyle=':', c='gray')
plt.text(8, 9 * 10 ** 4, s='Disk', c='gray')
plt.plot([96, 135], [5 * 10 ** 4, 4 * 10 ** 4], linestyle=':', c='gray')
plt.plot([123, 135], [3 * 10 ** 4, 4 * 10 ** 4], linestyle=':', c='gray')
plt.text(136, 3.8 * 10 ** 4, s='Satellites', c='gray')
plt.legend(loc='lower right')
plt.title('Temperature vs distance (indiscriminate of direction) of gas particles '
          'of LMC (Run 09_18), $t=-2.05$ Gyr', fontsize='small')
plt.savefig('/Users/dear-prudence/Desktop/tempVdist_LMC_scatter_snap114.png', dpi=240)
plt.show()

plt.close()'''
