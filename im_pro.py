import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def im_processing(n_slice, coef):

	path = '/Users/josephlamarre/Dev/McGill Hackaton 2021/stars-nasa-e1465057969522.jpg'
	image = Image.open('/Users/josephlamarre/Dev/McGill Hackaton 2021/stars-nasa-e1465057969522.jpg')

	data_i = np.asarray(image)

	slice_size = int(np.shape(data_i)[1] / n_slice)

	for i in range(n_slice):
		print(i)
		data_i[:,i*slice_size : i*slice_size + slice_size] = gaussian_filter(data_i[:,i*slice_size : i*slice_size + slice_size], sigma=coef[i])


	fig , axs = plt.subplots(1,1)

	axs.imshow(data_i)

	axs.set_axis_off()

	plt.tight_layout()
	plt.show()