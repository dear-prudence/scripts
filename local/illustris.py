import requests
import h5py
import matplotlib.pyplot as plt
import numpy as np


def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"c51ed2f5eb5082652543f265469a2ce9"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r


baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key": "c51ed2f5eb5082652543f265469a2ce9"}

r = get(baseUrl)
names = [sim['name'] for sim in r['simulations']]
i = names.index('TNG50-1')
sim = get(r['simulations'][i]['url'])

snaps = get(sim['snapshots'])
snap = get(snaps[-1]['url'])
subs = get(snap['subhalos'], {'limit': 20, 'order_by': '-mass_stars'})
sub = get( subs['results'][1]['url'])

print(subs['results'][0].keys())

print([subs['results'][i]['id'] for i in range(20)])

mpb1 = get(sub['trees']['sublink_mpb'])  # file saved, mpb1 contains the filename

mpb1 = '/Users/dear-prudence/py_astro/dear-prudence/local/sublink_mpb_96762.hdf5'

"""with h5py.File(mpb1,'r') as f:
    pos = f['SubhaloPos'][:]
    snapnum = f['SnapNum'][:]
    subid = f['SubhaloNumber'][:]

    i = np.where(snapnum == 99)"""

sub_prog_url = "http://www.tng-project.org/api/TNG50-1/snapshots/99/subhalos/96762/"
sub_prog = get(sub_prog_url)
print(sub_prog['pos_x'])
print(sub_prog['pos_y'])

cutout_request = {'gas': 'Coordinates,Masses'}
cutout = get(sub_prog_url + "cutout.hdf5", cutout_request)

with h5py.File(cutout,'r') as f:
    x = f['PartType0']['Coordinates'][:,0] - sub_prog['pos_x']
    y = f['PartType0']['Coordinates'][:,1] - sub_prog['pos_y']
    dens = np.log10(f['PartType0']['Masses'][:])

    plt.hist2d(x, y, weights=dens, bins=[150, 100])
    plt.xlabel('$\Delta x$ [ckpc/h]')
    plt.ylabel('$\Delta y$ [ckpc/h]')

    plt.show()