#!/usr/bin/python

import math, random

import matplotlib.pyplot as plt

import numpy as np

# global constants
filename = 'windows_2048to256'
mat_filename = 'matrix.npz'
N = 2000

# random matrix for generating values
M = np.load(mat_filename, 'r')['M']


# linear data generation
class linear():
    def __init__(self, dim_x, dim_y, max_x):
        # dim_x is number of variables, but X is dim_x + 1 cols
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_x = max_x

        # samples x randomly in the hypercube [0, max_x]^(dim_x) for the input points
        #
        # includes a column of 1's for the constant offset
        self.X = np.array([[1] + [random.uniform(0, self.max_x) for i in range(self.dim_x)] for j in range(N)])


        # the matrix to be ''discovered''
        self.m = M[:(self.dim_x + 1), :self.dim_y]


        # for simplicity, use standard gaussian noise
        self.mean = 0.0
        self.sig = 1.0

    def f(self, x):
        return x @ self.m + np.random.normal(self.mean, self.sig, (1, dim_y))

    def generate(self):
        Y = np.vectorize(self.f, signature='(n)->(m)')(self.X)
        np.savez(filename, x=self.X, y=Y)


# methylation-like window data generation
class window():
    def __init__(self, dim_x, dim_y, max_windows, min_slope, win_size_mean):
        # dim_x and dim_y are number of sample points for input and output
        self.dim_x = dim_x
        self.dim_y = dim_y

        # instantiate the base arrays
        #
        # train_x is the profiles generated via window functions
        # train_y is their averaged derivatives
        self.train_x = np.zeros((N,self.dim_x))
        self.train_y = np.zeros((N,self.dim_y))

        self.aux = np.zeros((N,2))

        # the 'base space' X
        #
        # this is used for evaluating the window function
        self.X = np.linspace(0, 2048, 2048)

        # max_windows sets the maximum number of windows
        self.max_windows = max_windows

        # min_slope sets how sloped the shores can be
        self.min_slope = min_slope

        # set the mean size of each window
        self.win_size_mean = win_size_mean

    def window_function(self, x, w, z, c):
        return 1 - .5 * (np.tanh((x - w) / c) + np.tanh((z + w - x) / c))


    def create_random_windows(self):
        # creates a random series of starting locations, window sizes, and slopes that are mutually non-overlapping

        # the windows are stored as a list of tuples (start, size, slope)
        window_list = []

        # the last window is the allowed starting point for the next window
        last_window = 0

        # initialize the loop
        window_count = 0
        loop_break = 0

        # first determine the number of windows
        if self.max_windows > 1:
            win_number = random.randrange(1, self.max_windows)
        else:
            win_number = 1

        # postulate a window and add it to the list, provided that it doesn't overlap with any currently extant windows
        while window_count < win_number and loop_break < 20:
            # pick a random slope (c)
            slope = np.random.uniform(1, self.min_slope)

            # figure out the distance from halfway up the shore to the top
            # 4 ~ arctanh(.999)
            #
            # we use half because the tanh() is zero at the halfway point, not the bottom
            halfshore = slope * 4

            # pick a window size (z)
            win_size = np.random.poisson(self.win_size_mean)

            # pick a start location so the whole window fits (w)
            #
            # attempt to use a slightly less bad distribution than uniform, but it randomly (ha) has issues so just try again
            try:
                start = np.random.triangular(math.floor(last_window + halfshore), math.floor(last_window + halfshore + 0.25 * (self.dim_x - halfshore - win_size - last_window - halfshore)), self.dim_x - math.floor(halfshore + win_size))
            except:
                loop_break += 1
                continue
            #start = random.randrange(, self.dim_x - math.floor(halfshore + win_size))

            # the total window size depends on the slope
            win_total_size = 2 * halfshore + win_size

            #print('attempting %s %s %s' % (start, win_size, slope))

            # if this window fits, add it to the list
            if math.ceil(last_window + win_total_size) < self.dim_x and win_size > 0:
                # add the window the to the list
                window_list.append((start, win_size, slope))

                # update the last window
                last_window = start + win_size + halfshore

                # increment the counter
                window_count += 1

            # don't run forever
            loop_break += 1

        return window_list


    def create_vector(self):
        # given a window assignment from create_window, create a vector by summing the window functions

        y = np.zeros(self.dim_x)

        # get a random assignment of windows
        winlist = self.create_random_windows()

        # add the functions together
        #
        # this uses the fact that away from the window, window_function(x) = 1.
        for p in winlist:
            y += np.vectorize(lambda x: w.window_function(x, p[0], p[1], p[2]))(self.X)

        # fix the normalization
        y -= (len(winlist) - 1) * np.ones(2048)

        # report some auxiliary info
        win_sum_size = sum([z+4*c for (w, z, c) in winlist])
        win_sum_number = len(winlist)

        return y, (win_sum_size, win_sum_number)


    def create_training_data(self):
        # create N different window patterns, find their derivatives, average them, and then save

        for i in range(N):

            print('training data %s of %s' % (i,N))
            # the compression ratio
            #c = int(self.dim_x / self.dim_y)

            # the vector and the 'truth'
            Y = self.create_vector()
            dY = np.gradient(Y[0])

            self.train_x[i] = Y[0]
            self.train_y[i] = np.array([max(arr.min(), arr.max(), key=abs) for arr in np.split(dY, self.dim_y)])

            self.aux[i] = np.array(Y[1])

        np.savez(filename, x=self.train_x, y=self.train_y, aux=self.aux)





if __name__ == '__main__':
    #D = linear()
    #D.generate(X)

    w = window(2048, 256, 8, 16, 128)

    w.create_training_data()

    plt.plot(w.X,w.train_x[0])
    plt.plot(w.X,np.array([[x] * 8 for x in w.train_y[0]]).flatten())
    plt.show()

    print('waiting')