import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tsne import bh_sne

def read_data(filename, input_shape):
  data = pd.read_csv(filename, header=None).as_matrix()
  return data[:,:input_shape]

dqn_data = read_data('train_dqn.csv', 4)
dqn_data = dqn_data[::2]
pg_data = read_data('train_pg.csv', 4)
dqn_y = np.zeros((len(dqn_data), 1))
pg_y = np.ones((len(pg_data),1)) * 8

x_data = np.concatenate((dqn_data, pg_data), axis=0)
y_data = np.concatenate((dqn_y, pg_y), axis=0)

print(x_data.shape)
print(y_data.shape)

# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')

# perform t-SNE embedding
vis_data = bh_sne(x_data)

np.save('done', vis_data)
np.save('y', y_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
