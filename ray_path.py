import numpy as np
import matplotlib.pyplot as plt

n = 100

rand = np.random.rand(n, n) + 1

def pathfinder(n_arr, t1, i, l):

	x = [i]
	y = [0]
	j = 1

	dir = np.zeros(2)
	dir[1] = 1

	dl = 0.5

	t2 = np.arcsin(n_arr[i, j - 1] / n_arr[i,j] * np.sin(t1))	

	while True:

		p = np.tan(t2) * l

		if p > (l - dl):
			print(dir)
			dir[:] = dir[::-1]
			print(dir)
			dl = np.tan(np.pi - t2) * (l - dl)
		else:
			dl = dl + p		

		t1 = np.abs(np.pi * dir[0]) - t2
		t2 = np.abs(np.arcsin(n_arr[i, j] / n_arr[i + int(dir[0]),j + int(dir[1])] * np.sin(t1)))

		i += int(dir[0])
		j += int(dir[1])

		x.append(i)
		y.append(j)

		if j == np.shape(n_arr)[1]-1:
			break
		if i == np.shape(n_arr)[0] - 2 or i == -1:
			break

	return x, y

x, y = pathfinder(rand, -0.1, 50, 1)

plt.plot(x, y[::-1])
plt.show()