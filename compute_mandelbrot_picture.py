from mandelbrot import mandelbrot_step
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import multiprocessing



# Define picture resolution
X_RES = 1680.0
Y_RES = 1050.0
MAX_DEPTH = 256

# Define center location and zoom level
Z_CENTER = -1.46 + 1j*(0.000)
ZOOM = 12.0 # 2^ZOOM

# Compute half the +/- bounds
y_interval = 1.0/2.0**(ZOOM)
x_interval = 1.0 * y_interval * (X_RES / Y_RES)

# Compute the points
x_linspace = np.linspace(Z_CENTER.real - x_interval, Z_CENTER.real + x_interval, X_RES )
y_linspace = np.linspace(Z_CENTER.imag - y_interval, Z_CENTER.imag + y_interval, Y_RES )

# Preallocate Image matrix
depth_array = np.zeros((Y_RES, X_RES), dtype=np.uint16)

# Parallel compute
num_cores = multiprocessing.cpu_count()

depth_array = Parallel(n_jobs = num_cores)(delayed(mandelbrot_step)(x_linspace, y_horizon, MAX_DEPTH)
        for y_horizon in y_linspace)

# Display Graph
plt.ion()
plt.figure()
plt.imshow(depth_array)
