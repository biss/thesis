__author__ = 'biswajeet'

import numpy as np
import math
from numpy import linalg as LA
import copy
from numpy import genfromtxt

class SwitchingRegression:

    #total number of data points
    number = 200

    #each column of data is a data point,[x1, x2, x3, x4, y].T is a column
    data = np.zeros(shape = (6, number), dtype=float)
    #data = np.array([[1,2],[2,2.5],[5,4], [2,1],[3,4],[4,7]]).T

    #initialized parameters of the two regimes, in (y = a + b*x1+ c*x2 + d*x3 + e*x4
    # --> 1st component is a, 2nd is b, 3rd is c... ie [a, b, c, d, e]
    regime1_param = [0.0, 0, 0, 0, 0]
    regime2_param = [0.0, 0, 0, 0, 0]

    #membership matrix initialised uniformly
    U = np.random.uniform(0.0, 1.0, [2,number])
    #U = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]).T
    m = 1.2
    c = 2
    epsilon = 0.00005
    E = np.random.uniform(0, 1.0, [2,number])
    delta = 0
    training_iter = 0

    #Data generation-----------------------------
    '''def generateData(self):
        #1  print 'in data generation:'
        D = np.zeros(shape = (2, self.number))

        for i in range(self.c):
            for k in range(self.number):
                D[i, k] = self.U[i, k]/sum(self.U[:, k])

        self.U = copy.copy(D)
        #self.U = np.array([[1.0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]])

        #first regime intercept and slope
        a_1 = 20
        b_1 = 5
        c_1 = 2
        #d_1 = -2
        #e_1 = 5.62

        #second regime intercept and slope
        a_2 = 50
        b_2 = 1
        c_2 = -7
        #d_2 = -2.12
        #e_2 = 0.82

        noise_x3 = np.random.uniform(-1, 1, self.number/2)
        #noise_x4 = np.random.uniform(-.001, .001, self.number/2)

        #generate the independent variable randomly 200X4 matrix
        x_regime1 = np.array([[1 for i in range(self.number/2)], [np.random.uniform(-25,50) for i in range(self.number/2)],
                              [np.random.uniform(-25,50) for i in range(self.number/2)], [np.random.uniform(-5,5) for i in
                                range(self.number/2)],[np.random.uniform(-5,5) for i in range(self.number/2)]])

        #calculate the dependent variable without noise
        y_axis_regime1 = np.column_stack(np.array([a_1, b_1, c_1, 0, 0])).dot(x_regime1)
        #1  print y_axis_regime1.shape

        #adding noise to the dependent variable
        y_axis_regime1 = y_axis_regime1 + np.random.normal(0, 0.7, self.number/2) + noise_x3

        #1  print "unstaged data; ", x_regime1.shape, y_axis_regime1.shape

        #concantenating y values
        tmp_data = np.append(x_regime1, y_axis_regime1, axis=0)
        for i in range(self.number/2):
            self.data[:, i] = tmp_data[:, i]

        #generate second cluster points
        #print 'first regime: ', self.data[:, : self.number/2]
        x_regime2 = np.array([[1 for i in range(self.number/2)], [np.random.uniform(-25,50) for i in range(self.number/2)],
                              [np.random.uniform(-25,50) for i in range(self.number/2)], [np.random.uniform(-5,5) for i in
                                range(self.number/2)],[np.random.uniform(-5,5) for i in range(self.number/2)]])

        y_axis_regime2 = np.column_stack(np.array([a_2, b_2, c_2, 0, 0])).dot(x_regime2)

        #adding noise to the dependent variable
        y_axis_regime2 = y_axis_regime2 + np.random.normal(0, 0.9, self.number/2) + noise_x3

        tmp_data = np.append(x_regime2, y_axis_regime2, axis=0)
        for i in range(self.number/2, self.number, 1):
            self.data[:, i] = tmp_data[:, i-(self.number/2)]
        print 'data:', self.data

        out = open('out_m.csv', 'w')
        for row in self.data.T:
            for column in row:
                out.write('%f;' % column)
            out.write('\n')
        out.close()'''

    def load_data(self):
        self.data = genfromtxt('out_m.csv', delimiter=',')
        #print 'data:', self.data

    def update_param(self):
        #1  print 'in param_update:'

        X = self.data[:5, :].T
        Y = np.zeros(shape=(1, self.number))
        Y[0, :] = np.array(self.data[-1, :])

        #1  print Y

        for i in range(self.c):
            D = np.zeros(shape = (self.number, self.number))
            for k in range(self.number):
                D[k, k] = copy.copy(pow(self.U[i, k], self.m))

            #1  print 'X and D shape: ',X.shape, D.shape
            if i == 0:

                a = (X.T).dot(D).dot(X)

                #avoid the singular case
                b = np.linalg.pinv(a)

                self.regime1_param = np.dot(b,((X.T).dot(D).dot(Y.T)))
                #1  print i, ": ", np.dot(b,((X.T).dot(D).dot(Y.T)))
            else:
                a = (X.T).dot(D).dot(X)
                b = np.linalg.pinv(a)

                self.regime2_param = np.dot(b,((X.T).dot(D).dot(Y.T)))
                #1  print i,": " , np.dot(b,((X.T).dot(D).dot(Y.T)))

    def train(self):

        U_old = copy.copy(self.U)
        self.update_param()
        #1  print 'U before updation:', self.U

        self.update_membership()
        #1  print 'U after updation:', self.U
        error = LA.norm(U_old - self.U)
        #print '--------------->',error
        print 'Em:',self.Em()

        self.training_iter = 0
        while error > self.epsilon:
            #1  print 'U before updation:', self.U
            U_old = copy.copy(self.U)
            #1  print "error: ", self.E
            #1  print "U: ", self.U

            self.update_param()
            #1  print 'after update param'

            self.update_membership()
            #1  print 'after update membership'

            #this error depends on the membership values
            error = LA.norm(U_old - self.U)
            print 'error------>', error
            print 'Em:', self.Em()

            self.training_iter += 1
        #1  print 'U after updation:', self.U
        print "error after completing training: ", error

    def update_membership(self):
        self.construct_error_matrix()

        #1  print 'error terms:', self.E

        for k in range(self.number):
            if all(self.E[:, k] > 0) == True:
                #1  print 'all error terms are non zero.'
                den = np.zeros(shape=(self.c, 1))
                for i in range(2):
                    den[i, 0] = float(1/self.calc_denominator(i, k))

                #1  print "updated value", den[:, 0]
                self.U[:, k] = den[:, 0]

            else:
                #1  print 'some error terms are zero.'
                for i in range(self.c):
                    if self.E[i, k] > 0:
                        self.U[i, k] = 0.0
                    else:
                        if sum (x > 0 for x in self.E[:, k]) > 0:
                            self.U[i, k] = float(1 / (sum (x > 0 for x in self.E[:, k])))

    def calc_denominator(self, i, k):
        value = 0.0
        for j in range(self.c):
            value = value + math.pow(self.E[i, k]/self.E[j, k], 1/(self.m - 1))
        return value

    def construct_error_matrix(self):
        for i in range(2):
            for j in range(self.number):

                self.E[i, j] = self.calculate_error(i+1, self.data[-1, j], self.data[:-1, j])

    def calculate_error(self, regime, dep, indep):

        if regime == 1:
            #1  print "variables one: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)

            return math.pow(dep - (np.dot(self.regime1_param.T, indep) + self.delta), 2)
        if regime == 2:
            #1  print "variables two: ", dep, indep
            #1  print "calculated value E :", math.pow(dep - (np.dot(self.regime2_param.T, indep) + self.delta), 2)
            return math.pow(dep - (np.dot(self.regime2_param.T, indep)  + self.delta), 2)
        else:
            print "invalid regime"
            return 0

    def Em(self):
        error = 0
        for k in range(self.number):
            for i in range(self.c):
                error += pow(self.U[i, k], self.m) * self.E[i, k]
        return error

if __name__ == '__main__':

    sr = SwitchingRegression()

    sr.load_data()

    sr.train()

    #specifying limits
    print 'betas are: ', sr.regime1_param, sr.regime2_param
    print sr.training_iter