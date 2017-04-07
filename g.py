import numpy as np
from matplotlib import pyplot as plt

vis_data = np.load('done.np.npy')
y_data = np.load('y.np.npy')

vis_x = vis_data[:25000, 0]
vis_y = vis_data[:25000, 1]

fig = plt.figure()

g1 = fig.add_subplot(211)
g1.scatter(vis_x, vis_y, c='b')
#plt.colorbar(ticks=range(10))
#plt.clim(-0.5, 9.5)

vis_x = vis_data[25000:, 0]
vis_y = vis_data[25000:, 1]

g2 = fig.add_subplot(212)
g2.scatter(vis_x, vis_y, c='r')
#g2.colorbar(ticks=range(10))
#g2.clim(-0.5, 9.5)

plt.show()
