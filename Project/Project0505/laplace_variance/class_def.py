import numpy as np 
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.integrate import quad


class variance_heston():
    def __init__(self, v0_H, k_H, θ_H, ε_H, ρ_H):
        self.v0, self.k, self.θ, self.ε, self.ρ = v0_H, k_H, θ_H, ε_H, ρ_H


class variance_32():
    def __init__(self, v0_32, k_32, θ_32, ε_32, ρ_32):
        self.v0, self.k, self.θ, self.ε, self.ρ = v0_32, k_32, θ_32, ε_32, ρ_32


class cbarheston():

    def __init__(self, texp, k, v0, theta) -> None:
        self.k, self.v0, self.theta, self.texp = k, v0, theta, texp

    def c0partial(self, t, T):
        return self.theta / self.k * (1 - np.exp(-self.k * t) + np.exp(-self.k * T))

    def c0(self, n=50):
        eq1 = self.v0 / self.k * (1 - np.exp(-self.k * self.texp))
        eq2 = 0
        t_discrete = np.arange(n) * self.texp / n
        for t, T in zip(t_discrete, np.r_[t_discrete[1:], self.texp]):
            eq2 += self.c0partial(t, T)
        eq3 = -n * self.theta / self.k + self.theta * self.texp
        return eq1 + eq2 + eq3
    
    def beta(self, sigma=1):
        return self.c0() * (np.exp(sigma ** 2 * self.texp) - 1)
    
    def alpha(self):
        return self.c0() / self.beta()
    
    def cbar(self, strike):
        alpha, beta = self.alpha(), self.beta()
        res = (
            alpha * beta * (1 - gamma.cdf(strike, alpha + 1, scale=beta))      # beta already inverted in the version given
            + strike * (1 - gamma.cdf(strike, alpha, scale=beta))        
        )
        return res
    
    def Laplace_transform(self, s):
        alpha, beta = self.alpha(), self.beta()
        return alpha*beta/s - 1/s**2 - (beta**2+1) * (alpha*beta**alpha) / (s+beta)**(alpha+1)
    

class cbar32():

    def __init__(self, texp, k, v0, theta, epsilon) -> None:
        self.k, self.v0, self.theta, self.texp, self.epsilon = round(k), v0, theta, texp, epsilon
        self.a = 2 / self.epsilon**2
        self.b = -2 * self.k / self.epsilon**2

    def R_bar(self, y, m_over_k):
        m = int(m_over_k * self.k)
        _sum = sum([1/(self.k+j) for j in range(m+1)])
        return 1/2 * (1/self.k + poisson.cdf(self.k, self.a/y) / (m+1) *_sum)

    def E(self, x):
        func = lambda t: np.exp(-t)/t
        return quad(func, x, np.inf)[0]
        
    def mid_sum(self, y):
        return sum([poisson.cdf(n, self.a/y) / (n*(n-self.b+1)) for n in range(1, self.k)])

    def h(self, y, m_over_k):
        h = self.a * (self.E(self.a/y) / (1-self.b) + self.mid_sum(y) + self.R_bar(y, m_over_k))
        return h

    def y_sum(self, N):
        dt = self.texp / N
        k_theta = self.k*self.theta
        return np.exp(k_theta*self.texp) * (np.exp(k_theta*dt)-1) / k_theta
    
    def c0(self, N = 1000, m_over_k = 2):
        return self.h(self.y_sum(N), m_over_k)
    
    
    def beta(self, sigma=1):
        return self.c0() * (np.exp(sigma ** 2 * self.texp) - 1)
    
    def alpha(self):
        return self.c0() / self.beta()
    
    def cbar(self, strike):
        alpha, beta = self.alpha(), self.beta()
        res = (
            alpha * beta * (1 - gamma.cdf(strike, alpha + 1, scale=beta))      # beta already inverted in the version given
            + strike * (1 - gamma.cdf(strike, alpha, scale=beta))        
        )
        return res
    
    def Laplace_transform(self, s):
        alpha, beta = self.alpha(), self.beta()
        return alpha*beta/s - 1/s**2 - (beta**2+1) * (alpha*beta**alpha) / (s+beta)**(alpha+1)