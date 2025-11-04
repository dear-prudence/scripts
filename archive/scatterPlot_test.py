import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import itertools
import pandas as pd

# plt.style.use(astropy_mpl_style)

image_header = f.getheader('/Users/ursa/Desktop/SMC_askap_parkes_PBC_K.fits', ext=0)
image_data = f.getdata('/Users/ursa/Desktop/SMC_askap_parkes_PBC_K.fits', ext=0)


# image_data.shape is (220, 3471, 3901)

def find_coordinates(array, threshold):
    coordinates = []
    indices = np.where(array > threshold)

    for x, y, z in zip(indices[0], indices[1], indices[2]):
        coordinates.append((x, y, z))

    return coordinates


# img = find_coordinates(image_data, 20)
print(image_header, flush=True)

# df = pd.DataFrame(img)
# df.to_csv('/Users/dear-prudence/Desktop/temp.csv')

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zlow, zhigh)
#    ax.scatter(xs, ys, zs, marker=m)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
