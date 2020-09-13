import numpy as np
from ct_charachterization import run_second_algorithm
from ct_charachterization.utility.utils import broadcast_3d_tile
from ct_charachterization import run_third_algorithm
import matplotlib.pyplot as plt

delta = -1030
mu = np.array([-1000, -700, -90, 50, 300])
# mu = np.array([340, 240, 100, 0, -160, -370, -540, -810, -987])
centered_mu = mu - delta
big_jay = len(mu)
y = np.load(f'''../resources/my_lungs.npy''') - delta
theta, gamma = run_second_algorithm(y, centered_mu=centered_mu, delta=delta, max_iter=10, tol=0.01)
form_of_first_mini_sclm = np.sum(np.expand_dims(y, axis=-1) * gamma, axis=(0, 1)).reshape((1, 1, big_jay))
form_of_second_mini_sclm = np.sum(np.power(np.expand_dims(y, axis=-1), 2) * gamma, axis=(0, 1)).reshape((1, 1, big_jay))
denominator_summation = np.sum(gamma, axis=(0, 1)).reshape((1, 1, big_jay))
first_mini_sclm = form_of_first_mini_sclm / denominator_summation
second_mini_sclm = form_of_second_mini_sclm / denominator_summation
first_sclm = np.sum(broadcast_3d_tile(first_mini_sclm, y.shape[0], y.shape[1], 1) * theta[0, :], axis=2)
second_sclm = np.sum(broadcast_3d_tile(second_mini_sclm, y.shape[0], y.shape[1], 1) * theta[0, :], axis=2)
var_of_y = second_sclm - np.power(first_sclm, 2)
print(var_of_y)
# plt.imshow(y, cmap=plt.cm.bone)
# plt.show()
