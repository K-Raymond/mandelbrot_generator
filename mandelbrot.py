import numpy as np 

# Compute the depth values of an array from real array x and real array y
def mandelbrot_step(x, y, calc_depth):
    z = x + 1j*y
    z0 = z.copy()
    depth = np.zeros_like(z, dtype=np.int16)
    count = 0
    while (count < calc_depth):
        z = z**2 + z0
        depth[z<2] += 1
        count += 1
    return depth
