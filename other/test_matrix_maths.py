# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:10:52 2020

@author: iant

TEST MATRIX MATHS
BASIC:
    https://www.mathcentre.ac.uk/resources/uploaded/sigma-matrices8-2009-1.pdf
    
LEAST SQUARES:
    https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html
    


"""

import numpy as np
import matplotlib.pyplot as plt


#%%

"""simplest model: solve the equations 
x + 2y = 4
3x − 5y = 1

A = [[1, 2], [3, -5]]
X = [x, y]
B = [4, 1]

AX = B

A^−1 A X = A^−1 B where A^-1 A = I = [[1, 0], [0, 1]]

X = A^−1 B

Inverse of [[a, b], [c, d]] = 1/determinant [[d, -b], [-c, a]]
determinant = ad-bc

Transpose of [[a, b], [c, d]] = [[a, c], [b, d]]


"""

A = np.array([[1, 2], [3, -5]])
B = np.array([4, 1])
inv_A = np.linalg.inv(A)

X = np.matmul(inv_A, B)
print(X)

#%%

"""
least squares fitting without error

yi = a0 + a1 xi + a2 xi**2 + ak xi**k

y = Xa

XT y = XT X a

a = (XT X)-1 (XT y)
"""

n = 10 #n points
k = 3 # polynomial degree inc constant

x = np.arange(n)
# y = x**2.0 + np.random.rand(n) * 10.0 #random points roughly following squared
y = np.array([ 0.81046259,  4.54477851, 12.79814535,  9.78393232, 24.14602626,
       25.63567665, 43.40104419, 56.49700694, 73.71998862, 84.63786308])






coeffs = np.zeros(k+1)

plt.figure()
plt.title("Least squares no error")
plt.scatter(x, y)

#make vandermonde matrix
#[[1 x_1 x_1^2 ... x_1^k],
# [1 x_2 x_2^2 ... x_2^k],
# [1 x_n x_n^2 ... x_n^k]]
X = np.ones((n, k+1))
for poly_order in range(0, k + 1):
    X[:, poly_order] = x**poly_order

# y = X a
# XT y = XT X a
# a = (XT X)^-1 (XT y)

xt_y = np.matmul(X.T, y)
xt_x = np.matmul(X.T, X)

#linalg.solve(X, y). Outputs a, where y = X a
a = np.linalg.solve(xt_x, xt_y) #calculate coefficients

y_fit = np.matmul(X, a)

plt.plot(x, y_fit)




#%%
"""
weighted least squares fitting (with error)

yi = a0 + a1 xi + a2 xi**2 + ak xi**k

y = Xa


chisq(a) = (Xa − y)T Vy-1 (Xa − y) = aT XT Vy−1 X a − 2aT XT Vy−1 y + yT Vy−1 y

d(chisq(a))/da = 0 for best fit
XT Vy-1 X a − XT Vy−1 y = 0

a = [XT Vy−1 X]−1 XT Vy−1 y

"""

n = 10 #n points
k = 3 # polynomial degree inc constant

x = np.arange(n)
# y = x**2.0 + np.random.rand(n) * 10.0 #random points roughly following squared
y = np.array([ 0.81046259,  4.54477851, 12.79814535,  9.78393232, 24.14602626,
       25.63567665, 43.40104419, 56.49700694, 73.71998862, 84.63786308])


stdev = np.ones(n) * 10.0 # sqrt of variance
stdev[6] = 100.0 #set point larger (less accurate) as an example
stdev[5] = 2. #set point smaller (more accurate) as an example

coeffs = np.zeros(k+1)

plt.figure()
plt.title("Least squares including variance")
plt.scatter(x, y)

#make vandermonde matrix
#[[1 x_1 x_1^2 ... x_1^k],
# [1 x_2 x_2^2 ... x_2^k],
# [1 x_n x_n^2 ... x_n^k]]
X = np.ones((n, k+1))
for poly_order in range(0, k + 1):
    X[:, poly_order] = x**poly_order



precision = np.diag(1.0 / stdev**2) #precision matrix, diagonal array of 1/variance = inverse of covariance matrix
# (XT Vy^−1 X) a - (XT Vy^-1) y = 0
# (XT Vy^-1) y = (XT Vy^-1 X) a

y_precision = np.matmul(precision, y) #Vy^−1 X
x_precision = np.matmul(precision, X) #Vy^-1 y

# y = X a
# XT y = XT X a


xt_y_precision = np.matmul(X.T, y_precision)
xt_x_precision = np.matmul(X.T, x_precision)

a = np.linalg.solve(xt_x_precision, xt_y_precision)

y_fit = np.matmul(X, a)

plt.plot(x, y_fit)



#%%
"""
fitting gaussians to two spectra with an unknown coefficient using apriori and optimal estimation

"""
def gaussian(x, b):
    a = 0.4
    # b = 3.
    c = 2.
    return a * np.exp(-((x - b)/c)**2.0)

#jacobian
def jacobian(x, b):

    a = 0.4
    # b = 3.
    c = 2.

    # dg_da = np.exp(-((x - b)/c)**2.0)
    def dg_db(x, a, b, c):
        return  (2. * a * (x - b) * np.exp(-(x - b)**2./c**2.))/c**2.

    # dg_dc = (2 * a (x - b)**2 * np.exp(-(x - b)**2/c**2))/c**3
    arr = np.concatenate([dg_db(x, a, b[0], c), dg_db(x, a, b[1], c)])
    return np.array([arr, arr]).T


n = 20 #n points


x = np.arange(-n/2, n/2)
y = np.array([gaussian(x, 4.2), gaussian(x, 4.2)])

# y = np.zeros_like(x) + 0.0
stdev = np.ones_like(y) * 0.1


# a = 0.4
b = np.array([3.5, 3.0]) #first guess
# c = 2.

y_fit = np.array([gaussian(x, b[0]), gaussian(x, b[1])])


plt.figure()
plt.plot(x, y[0, :], label="spectrum 1")
plt.plot(x, y[1, :], label="spectrum 2")
plt.plot(x, gaussian(x, b[0]), label="guess 1")
plt.plot(x, gaussian(x, b[1]), label="guess 2")

#error on points
Se_inv = np.diag(1.0 / stdev.ravel()**2) #precision matrix i.e. Sy-1

#error on a priori
Sa = np.ones(2) + 10.
Sa_inv = np.diag(1.0 / Sa)

Ky = jacobian(x, b)
#xi+1 = xa  + (Sa–1 + KT Se–1 K)–1 KT Se–1 [y – F(x,b) + K(x – xa)],
#xi+1 = xa + (W2)-1 KT Se-1 

niter_max = 5
for step in range(niter_max):

    chisq = np.sqrt(np.sum(((y-y_fit)/stdev)**2))
    print("step %d: chi^2 = "%step, chisq)
    
    print(Se_inv.shape, Ky.shape) 
    W1 = np.matmul(Se_inv, Ky)
    print(Sa_inv.shape, Ky.T.shape, W1.shape)
    W2 = Sa_inv + np.matmul(Ky.T, W1)
    print(Ky.shape, b.shape)
    w1 = np.matmul(Ky, b)
    print(W1.T.shape, y.shape, y_fit.shape, w1.shape)
    w2 = np.matmul(W1.T, y.ravel() - y_fit.ravel() + w1.ravel())
    print(W2.shape, w2.shape)
    dx = np.linalg.solve(W2, w2)

    print(dx)
    b = dx
    y_fit = np.array([gaussian(x, b[0]), gaussian(x, b[1])])


plt.plot(x, gaussian(x, dx[0]), linestyle="--", label="OE 1")
plt.plot(x, gaussian(x, dx[1]), linestyle="--", label="OE 2")

plt.legend()


#%%
"""
fitting gaussians to spectrum with a two unknown coefficients using apriori and optimal estimation

"""
def gaussian(x, a, b, c):
    # a = 0.4
    # b = 3.
    # c = 2.
    return a * np.exp(-((x - b)/c)**2.0)


#jacobian
def jacobian(x, a, b, c):

    # a = 0.4
    # b = 3.
    # c = 2.

    def dg_da(x, a, b, c):
        return np.exp(-((x - b)/c)**2.)
    
    def dg_db(x, a, b, c):
        return  (2. * a * (x - b) * np.exp(-(x - b)**2./c**2.))/c**2.

    def dg_dc(x, a, b, c):
        return (2. * a * (x - b)**2. * np.exp(-(x - b)**2./c**2.))/c**3.


    arr = np.array([dg_da(x, a, b, c), dg_db(x, a, b, c), dg_dc(x, a, b, c)])
    return arr.T


n = 20 #n points


x = np.arange(-n/2, n/2)
y = gaussian(x, 0.5, 2.2, 1.8)

stdev = np.ones_like(y) * 0.01


a = np.array([0.4]) #first guess
b = np.array([3.0]) #first guess
c = np.array([2.5]) #first guess
n_coeffs = 3


plt.figure()
plt.plot(x, y, label="spectrum")
plt.plot(x, gaussian(x, a, b, c), label="guess 1")

#error on points
Se_inv = np.diag(1.0 / stdev.ravel()**2) #precision matrix i.e. Sy-1

#error on a priori
Sa = np.ones(3) + 10.
Sa_inv = np.diag(1.0 / Sa)

Ky = jacobian(x, a, b, c)
#xi+1 = xa  + (Sa–1 + KT Se–1 K)–1 KT Se–1 [y – F(x,b) + K(x – xa)],
#xi+1 = xa + (W2)-1 KT Se-1 

niter_max = 5
for step in range(niter_max):

    y_fit = gaussian(x, a, b, c)
    chisq = np.sqrt(np.sum(((y-y_fit)/stdev)**2)/n_coeffs)
    print("step %d: chi^2 = "%step, chisq)
    
    # print(Se_inv.shape, Ky.shape) 
    W1 = np.matmul(Se_inv, Ky)
    # print(Sa_inv.shape, Ky.T.shape, W1.shape)
    W2 = Sa_inv + np.matmul(Ky.T, W1)
    # print(Ky.shape, b.shape)
    w1 = np.matmul(Ky, [a, b, c])
    # print(W1.T.shape, y.shape, y_fit.shape, w1.shape)
    w2 = np.matmul(W1.T, y.ravel() - y_fit.ravel() + w1.ravel())
    # print(W2.shape, w2.shape)
    dx = np.linalg.solve(W2, w2)

    print(dx)
    a, b, c = dx
    plt.plot(x, gaussian(x, a, b, c), linestyle="--", label="OE %i chi %0.3f" %(step, chisq))



plt.legend()