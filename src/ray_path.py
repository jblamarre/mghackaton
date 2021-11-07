import numpy as np
import matplotlib.pyplot as plt

def pathfinder(n_arr, t1, i, l):

        x = [i]
        y = [0]
        
        t = []
        n = []
        j = 1

        dir = np.zeros(2)
        dir[1] = 1

        dl = 0.5

        t2 = np.arcsin(n_arr[j - 1, i] / n_arr[j,i] * np.sin(t1))

        while True:

            p = np.tan(t2) * l

            if p > (l - dl):
                dir[:] = dir[::-1]
                dl = np.tan(np.pi - t2) * (l - dl)
            else:
                dl = dl + p

            t1 = np.abs(np.abs(np.pi * dir[0]) - t2)
            t2 = np.abs(np.arcsin(n_arr[j, i] / n_arr[j + int(dir[1]),i + int(dir[0])] * np.sin(t1)))

            if t1 > np.arcsin(n_arr[j + int(dir[1]),i + int(dir[0])]/ n_arr[j, i]) + 100:
                dir[0] = dir[0] * -1
                t2 = t1
                continue
            else:
                i += int(dir[0])
                j += int(dir[1])

                x.append(i)
                y.append(j)
                t.append(t1)
                n.append(np.arcsin(n_arr[j, i]))

            if j == (np.shape(n_arr)[1] - 1):
                break
            if i == np.shape(n_arr)[0] - 1 or i == -1:
                break

        return x, y, n, t