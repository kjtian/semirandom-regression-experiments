import numpy as np
import math
from scipy.linalg import expm, solve
from scipy.stats import ortho_group
from scipy import sparse
from sklearn.linear_model import Ridge
from scipy import stats
import time
from scipy.sparse.linalg import eigs, eigsh, lsqr
import scipy as sp
import sklearn.linear_model as lin
from scipy.integrate import quad

# Compute matrix exponential vector products
def mexpvec_naive(Mhalf, v, degree):
    return np.dot(expm(-degree * np.dot(Mhalf.T, Mhalf)), v)

# compute exp(-degree * Mhalf^T Mhalf) * v approximately via ridge regression on Mhalf, make sure to normalize Mhalf by sqrt{degree}
def mexpvec_ridge(Mhalf, v, coeffs, reg):
	degree = len(coeffs) - 1
	(n, d) = np.shape(Mhalf)
	sol = np.zeros(d)
	for i in xrange(degree):
		sol += coeffs[i] * v
		b = -np.dot(Mhalf, sol)
		sol = sol + reg.fit(Mhalf, b).coef_ # compute (argmin ||M(x + v)||_2^2 + ||x||_2^2) + v aka sol = (M^T M + I)^{-1} sol
	return sol + (coeffs[degree] * v)

# Helper methods for MPC

def pmone(k, d):
    return np.random.choice([-np.sqrt(1.0/k), np.sqrt(1.0/k)], d)

def jl(d, k):
	return np.random.choice([-np.sqrt(1.0/k), np.sqrt(1.0/k)], size = (d, k))

def jlt(d, k):
	return np.random.choice([-np.sqrt(1.0/k), np.sqrt(1.0/k)], size = (k, d))

# Compute all tau * <a_i, exp(-tau * A^T diag(w) A) a_i> / Tr exp(-tau * A^T diag(w) A)
def cov_gradients(A, w, k, tau, coeffs):
	reg = lin.Ridge(alpha=1.0, tol = 1e-3, solver = 'saga') # regularizes by identity already
	(n, d) = np.shape(A)
	degree = len(coeffs) - 1
	# start = time.time()
	scalevec = (tau / (2.0 * np.sqrt(degree))) * np.sqrt(w)
	Mhalf = A * np.reshape(scalevec, (n, 1))
	# end = time.time()
	# print "Time to make Mhalf: " + str(end - start)
	# start = time.time()
	jlvecs = jlt(d, k)
	QP = np.asarray([mexpvec_ridge(Mhalf, jlvecs[j], coeffs, reg) for j in xrange(k)]) # Q is JL sketch, P is poly
	# end = time.time()
	# print "Time to matexpvec_ridge: " + str(end - start)
	# start = time.time()
	texp = np.linalg.norm(QP) ** 2.0
	reduced_A = np.dot(A, QP.T)
	v = tau * np.asarray([np.linalg.norm(reduced_A[i]) ** 2 for i in xrange(n)]) / texp
	# end = time.time()
	# print "Time for everything else: " + str(end - start)
	return v

def cov_gradients_naive(A, w, k, tau):
    (n, d) = np.shape(A)
    fullmhalf = AWA(A, -tau * w / 2.0)
    expmhalf = expm(fullmhalf) # This should be exp(-tau * A^T W A / 2)
    expAmult = np.dot(A, expmhalf)
    Q = jl(d, k)
    traceest = np.dot(expmhalf, Q)
    texp = np.linalg.norm(traceest) ** 2.0
    return tau * np.asarray([np.linalg.norm(expAmult[i]) ** 2 for i in xrange(n)]) / texp

def cov_gradients_really_naive(A, w, tau):
	(n, d) = np.shape(A)
	fullmat = AWA(A, -tau * w)
	expmatrix = expm(fullmat)
	return tau * np.asarray([np.dot(A[i], np.dot(expmatrix, A[i])) for i in xrange(n)]) / np.trace(expmatrix)

# This is ideally our final algorithm. Other versions are for benchmarking speed.
# alpha is step size, jlk is JL directions, coeffs is rational approx, run for niters
def mpc_ideal_niters(A, tau, alpha, niters, jlk, coeffs):    
    (n, d) = np.shape(A)
    gradP = np.asarray([np.dot(A[i], A[i]) for i in xrange(n)])
    w = 1.0 / (n * gradP) # can choose how small to init
    for k in xrange(niters):
        gradC = cov_gradients(A, w, jlk, tau, coeffs) # trace product covering
        ind = gradP < gradC
        if ind.sum() == 0:
            return w
        delta = (1.0 - gradP / gradC) * ind
        w = np.asarray(w * (np.ones(n) + alpha * delta))
    return w

def mpc_naive_niters(A, tau, alpha, niters):    
    (n, d) = np.shape(A)
    gradP = np.asarray([np.dot(A[i], A[i]) for i in xrange(n)])
    w = 0.0001 / gradP # can choose how small to init
    for k in xrange(niters):
        gradC = cov_gradients_naive(A, w, tau) # trace product covering
        ind = gradP < gradC
        if ind.sum() == 0:
            return w
        delta = (1.0 - gradP / gradC) * ind
        w = np.asarray(w * (np.ones(n) + alpha * delta))
    return w

# Compute rational coefficients via np.asarray(p(degree))

def integrand_gamma(t, k, n):
    return ((-2.0 * n) * np.exp(-n * (1 + t) / (1 - t)) / ((1 - t) ** 2)) * sp.special.legendre(k)(t)

def gamma(n):
    return [quad(integrand_gamma, -1, 1, args=(k, n))[0] for k in range(n)]

def r(n):
    g = gamma(n)
    return sum([np.poly1d(((k + 0.5) * g[k] * sp.special.legendre(k))) for k in range(n)])

def q(n):
    rn = np.asarray(r(n))
    scaling = np.asarray([(1.0 / (n - i)) for i in range(n)]) # gets r back when you differentiate
    q_noconstant = np.poly1d(np.append(np.multiply(rn, scaling), 0))
    return q_noconstant + np.poly1d([-q_noconstant(1)])

def ratapprox(n): # convert q into an explicit polynomial in z = (1 + x/d)^{-1}
    qn = np.asarray(q(n))
    running_sum = np.poly1d([0])
    for k in range(n):
        running_sum += np.poly1d([qn[k]])
        running_sum *= np.poly1d([-2, 1])
    return running_sum + np.poly1d([qn[n]])

# Helper functions
def tau_lmax(A):
    evals = np.linalg.eig(np.dot(A.T, A))[0]
    return (np.sum(evals) / np.min(evals), np.max(evals))

def tau(A):
    evals = np.linalg.eig(np.dot(A.T, A))[0]
    return np.sum(evals) / np.min(evals)

def taufull(M):
    evals = np.linalg.eig(M)[0]
    return np.sum(evals) / np.min(evals)

def kappa(A):
    evals = np.linalg.eig(np.dot(A.T, A))[0]
    return np.max(evals) / np.min(evals)

def AWA(A, w):
	wscale = np.reshape(w, (np.shape(A)[0], 1))
	WA = A * wscale
	return np.dot(A.T, WA)

def WhalfA(A, w):
	whalfscale = np.reshape(np.sqrt(w), (np.shape(A)[0], 1))
	return A * whalfscale

def AWAv(A, w, v):
    return np.dot(A.T, w * np.dot(A, v))

# Generate matrices

def rand_gauss_rows(n, covsqrt):
    d = np.shape(covsqrt)[0]
    return np.dot(np.random.normal(size = (n, d), scale = 1.0 / np.sqrt(n)), covsqrt)

def gauss_mat(n, d, m, s):
    return np.random.normal(loc = m, scale = s, size = (n, d))

def rand_matrix(d, l, h):
    U = ortho_group.rvs(d)
    D = np.diag(np.random.uniform(low=l, high=h, size=d))
    return np.dot(U.T, np.dot(D, U))