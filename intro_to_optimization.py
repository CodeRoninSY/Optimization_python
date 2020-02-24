#!/usr/bin/env python
# intro_to_optimization.py
# <2020-02-22> CodeRoninSY
#
# https://www.youtube.com/watch?v=geFER2oVvvU&t=1870s
#

import numpy as np
import scipy.special as ss
import scipy.optimize as opt
import scipy.stats as stats
import mystic
import mystic.models as models
import cvxopt as cvx
from cvxopt import solvers as cvx_solvers
import matplotlib.pyplot as plt

# Optimizer
objective = np.poly1d([1.3, 4.0, 0.6])
print(objective)

x_ = opt.fmin(objective, [3])
print("solved: x={}".format(x_))

x = np.linspace(-4, 1, 101)
plt.plot(x, objective(x))
plt.plot(x_, objective(x_), 'ro')
plt.show()

# Box constraints
x = np.linspace(2, 7, 200)

# 1st order Bessel
j1x = ss.j1(x)
plt.plot(x, j1x)

# use scipy.optimize's more modern "result object" interface
result = opt.minimize_scalar(ss.j1, method="bounded", bounds=[2, 4])

j1_min = ss.j1(result.x)
plt.plot(result.x, j1_min, 'ro')
plt.show()

# The gradient and/or Hessian
# print(models.rosen.__doc__)

mystic.model_plotter(mystic.models.rosen,
                    kwds='-f -d -l "x,y," -x 1 -b "-3:3:.1, -1:5:.1, 1"')

# initial guess
x0 = [1.3, 1.6, -0.5, -1.8, 0.8]

result = opt.minimize(opt.rosen, x0)
print("result.x: {}".format(result.x))

# number of function evaluations
print("function evals: {}".format(result.nfev))

# again, but this time provide the derivative
result = opt.minimize(opt.rosen, x0, jac=opt.rosen_der)
print("result.x: {}".format(result.x))

# number of function evaluations and derivative evaluations
print("nfev: {}, njev: {}".format(result.nfev, result.njev))
print('')

# however, note for a different x0...
for i in range(5):
    x0 = np.random.randint(-20,20,5)
    result = opt.minimize(opt.rosen, x0, jac=opt.rosen_der)
    print("{} @ {} evals".format(result.x, result.nfev))

# The penalty functions
# psi(x) = f(x) + k * p(x)
# http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp
'''
  Maximize: f(x) = 2*x0*x1 + 2*x0 - x0**2 - 2*x1**2

  Subject to:    x0**3 - x1 == 0
                         x1 >= 1
'''

def objective(x, sign=1.0):
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)


def derivative(x, sign=1.0):
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([dfdx0, dfdx1])


# unconstrained
result = opt.minimize(objective, [-1.0, 1.0], args=(-1.0,),
                      jac=derivative, method='SLSQP',
                      options={'disp': True})
print("unconstrained: {}".format(result.x))


cons = ({'type': 'eq',
         'fun': lambda x: np.array([x[0]**3 - x[1]]),
         'jac': lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([x[1] - 1]),
         'jac': lambda x: np.array([0.0, 1.0])})

# constrained
result = opt.minimize(objective, [-1.0, 1.0], args=(-1.0,), jac=derivative,
                      constraints=cons, method='SLSQP',
                      options={'disp': True})

print("constrained: {}".format(result.x))

# Optimizer classifications
# Constrained vs unconstrained (and importantly LP QP)

# constrained: linear (i.e. A*x + b)
print(opt.cobyla.fmin_cobyla)
print(opt.linprog)

# constrained: quadratic programming  (i.e. up to x**2)
print(opt.fmin_slsqp)

# http://cvxopt.org/examples/tutorial/lp.html
'''
minimize:  f = 2*x0 + x1

subject to:
           -x0 + x1 <= 1
            x0 + x1 >= 2
            x1 >= 0
            x0 - 2*x1 <= 4
'''

A = cvx.matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
b = cvx.matrix([1.0, -2.0, 0.0, 4.0])
cost = cvx.matrix([2.0, 1.0])
sol = cvx_solvers.lp(cost, A, b)

print(sol['x'])

# http://cvxopt.org/examples/tutorial/qp.html
'''
minimize:  f = 2*x1**2 + x2**2 + x1*x2 + x1 + x2

subject to:
            x1 >= 0
            x2 >= 0
            x1 + x2 == 1
'''

Q = 2*cvx.matrix([[2, .5], [.5, 1]])
p = cvx.matrix([1.0, 1.0])
G = cvx.matrix([[-1.0, 0.0], [0.0, -1.0]])
h = cvx.matrix([0.0, 0.0])
A = cvx.matrix([1.0, 1.0], (1, 2))
b = cvx.matrix(1.0)
sol = cvx_solvers.qp(Q, p, G, h, A, b)

print(sol['x'])

# Local vs Global
# probabilistic solvers, that use random hopping/mutations
print(opt.differential_evolution)
print(opt.basinhopping)

# bounds instead of an initial guess
bounds = [(-10., 10)]*5

for i in range(10):
    result = opt.differential_evolution(opt.rosen, bounds)
    # result and number of function evaluations
    print(result.x, '@ {} evals'.format(result.nfev))

# Least-squares fitting
# Define the function to fit.

def function(x, a, b, f, phi):
    result = a * np.exp(-b * np.sin(f * x + phi))
    return result

# Create a noisy data set around the actual parameters
true_params = [3, 2, 1, np.pi/4]
print("target parameters: {}".format(true_params))
x = np.linspace(0, 2*np.pi, 25)
exact = function(x, *true_params)
noisy = exact + 0.3*stats.norm.rvs(size=len(x))

# Use curve_fit to estimate the function parameters from the noisy data.
initial_guess = [1, 1, 1, 1]
estimated_params, err_est = opt.curve_fit(function, x, noisy,
                                          p0=initial_guess)
print("solved parameters: {}".format(estimated_params))

# err_est is an estimate of the covariance matrix of the estimates
print("covariance: {}".format(err_est.diagonal()))

plt.plot(x, noisy, 'ro')
plt.plot(x, function(x, *estimated_params))
plt.show()


# Integer programming
def system(x, a, b, c):
    x0, x1, x2 = x
    eqs = [
        3 * x0 - np.cos(x1*x2) + a,  # == 0
        x0**2 - 81*(x1+0.1)**2 + np.sin(x2) + b,  # == 0
        np.exp(-x0*x1) + 20*x2 + c  # == 0
    ]
    return eqs


# coefficients
a = -0.5
b = 1.06
c = (10 * np.pi - 3.0) / 3

# initial guess
x0 = [0.1, 0.1, -0.1]

# Solve the system of non-linear equations.
result = opt.root(system, x0, args=(a, b, c))
print("root:", result.x)
print("solution:", result.fun)

# Parameter estimation
# Create clean data.
x = np.linspace(0, 4.0, 100)
y = 1.5 * np.exp(-0.2 * x) + 0.3

# Add a bit of noise.
noise = 0.1 * stats.norm.rvs(size=100)
noisy_y = y + noise

# Fit noisy data with a linear model.
linear_coef = np.polyfit(x, noisy_y, 1)
linear_poly = np.poly1d(linear_coef)
linear_y = linear_poly(x)

# Fit noisy data with a quadratic model.
quad_coef = np.polyfit(x, noisy_y, 2)
quad_poly = np.poly1d(quad_coef)
quad_y = quad_poly(x)

plt.plot(x, noisy_y, 'ro')
plt.plot(x, linear_y)
plt.plot(x, quad_y)
#plt.plot(x, y)
plt.show()
