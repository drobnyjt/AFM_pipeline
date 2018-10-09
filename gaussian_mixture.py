from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import sklearn as skl
import skimage as ski
import pandas as pd
from matplotlib import cm
from noise import pnoise2, snoise2
from scipy import optimize as opt

def gaussian(x, height, center, width):
    return height * np.exp( -(x-center)**2/(2*width**2) )
#end def gaussian

#def ngaussian(x, n, heights, centers, widths):
def ngaussian(*list_args, **kwargs):
    x = list_args[0]
    n = list_args[1]

    heights = list_args[2:2+n]
    centers = list_args[2+n:2+2*n]
    widths = list_args[2+2*n:2+3*n]

    sum = 0
    for i in range(n):
        sum += gaussian(x, heights[i], centers[i], widths[i])
    #end for
    return sum
#end def ngaussian

#parameters
octaves = 1
freq1 = 8.0 * octaves
freq2 = 16.0 * octaves
x_size = 256
y_size = 256

#initialize x, y, and data vectors
X = np.zeros(x_size*x_size)
Y = np.zeros(y_size*y_size)
data = np.zeros(x_size*y_size)

#generate perlin noise
i=0

for x in range(x_size):
    for y in range(y_size):
        X[i] = float(x) / x_size
        Y[i] = float(y) / y_size
        noise1 = pnoise2(x/freq1, y/freq1, octaves)
        noise2 = snoise2(x/freq2, y/freq2, octaves)
        data[i] = noise1*noise1*noise2 - 0.5*noise2*noise2
        #data[i] = noise1*noise2 - np.abs(noise2*noise2*noise2)
        i+=1
    #end for
#end for

data = data - np.mean(data)

data_frame = pd.DataFrame({'x':X, 'y':Y, 'z':data})

#figure 1: heatmap
fig1 = plt.figure(1)
data_array = np.array(data_frame.z).reshape(x_size,y_size)
heatmap = plt.imshow(data_array)
fig1.colorbar(heatmap)
#end figure 1

#figure 2: surface
fig2 = plt.figure(2)
ax = Axes3D(fig2)
surf = ax.plot_trisurf(data_frame.x, data_frame.y, data_frame.z, cmap=cm.viridis, shade='true')
fig2.colorbar(surf, shrink=0.5, aspect = 5)
ax.set_zticks([-2.0, 0.0, 2.0])
ax.set_zlim([-3.0,3.0])
#end figure 2

#figure 3: histogram of heights
fig3 = plt.figure(3)
num_bins = 50
bins = np.linspace(-1.0,1.0, num_bins)
dz = bins[1]-bins[0]
bin_centers = bins + dz/2.0
groups = data_frame.groupby(pd.cut(data_frame.z, bins))
counts = groups.count().z.values
counts = counts / float(np.sum(counts))
plt.plot(bin_centers[:-1],counts)

#Fit to multi-gaussian
n_gaussian = 3
errfunc = lambda p, bin_centers, counts: (ngaussian(bin_centers, n_gaussian, *p) - counts)**2
guess1 = np.random.random(3*n_gaussian)*0.01

lower_bounds = np.concatenate([np.zeros(n_gaussian), np.ones(n_gaussian)*np.min(bin_centers), np.zeros(n_gaussian)])
upper_bounds = np.concatenate([np.ones(n_gaussian)*np.inf, np.ones(n_gaussian)*np.max(bin_centers), np.ones(n_gaussian)*np.inf])

optim1 = opt.least_squares( errfunc, guess1, args=(bin_centers[:-1],counts), bounds=(lower_bounds,upper_bounds), method='trf', ftol=1e-12 )
params = optim1.x
print(params)

solution = np.zeros(num_bins)

for i in range(num_bins):
    solution[i] = ngaussian(bin_centers[i], n_gaussian, *params)
#end for
plt.plot(bin_centers,solution)
#end figure 3

#figure 4: CDF
fig4 = plt.figure(4)
cdf = np.zeros(num_bins)

for i in range(num_bins):
    result = sp.integrate.quad(lambda x: ngaussian(x, n_gaussian, *params), -2.0, bin_centers[i])
    cdf[i] = result[0]
#end for
plt.plot(bin_centers, cdf)
#end figure 4

plt.show()
