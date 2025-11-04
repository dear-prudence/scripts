import numpy as np
from astropy.io import fits as f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

data = f.getdata('/Volumes/enceladus/SMC_substructures/SMC_askap_parkes_PBC_K.fits', ext=0)

# Define the threshold value for the isosurface
threshold_value = 0.5

# Get coordinates where the data is above the threshold
x, y, z = np.where(data > threshold_value)

# Create a 3D isosurface plot using plotly
fig = go.Figure(data=[go.Isosurface(
    x=x,
    y=y,
    z=z,
    value=data[x, y, z],
    isomin=threshold_value,
    isomax=1.0,
    opacity=0.5,
    surface_count=5,  # Increase for smoother surfaces
    caps=dict(x_show=False, y_show=False)
)])

fig.update_layout(scene=dict(
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    zaxis=dict(title='Z'),
    aspectmode='cube'
))

fig.show()
