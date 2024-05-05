import pyfeng as pf
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import gamma


def plot_figure1(sigma, vov, mr, rho, theta, strike, spot, texp, texp2):

    m = pf.HestonFft(sigma, vov=vov, mr=mr, rho=rho, theta=theta)
    p = m.price(strike, spot, texp)
    p2 = m.price(strike, spot, texp2)

    demovol = pf.Bsm(sigma=1)   # initialize a Bsm
    impvol = demovol.impvol(p, strike, spot, texp)
    impvol2 = demovol.impvol(p2, strike, spot, texp2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 1. Simultaneous fit of Heston model to S&P 500 implied volatility on July 31, 2009', fontsize = 14)

    ax[0].plot(strike / spot, impvol, color = 'black')
    ax[0].set_title('3-month S&P 500 implied volatility')
    ax[0].set_xlabel('Relative strike')
    ax[0].set_ylabel('Implied volatility')

    ax[1].plot(strike / spot, impvol2, color = 'black')
    ax[1].set_title('6-month S&P 500 implied volatility')
    ax[1].set_xlabel('Relative strike')
    ax[1].set_ylabel('Implied volatility')
    ax[1].sharey(ax[0])


def plot_figure2(sigma, vov, mr, rho, theta, strike, spot, texp, texp2):

    m = pf.Sv32Fft(sigma, vov=vov, mr=mr, rho=rho, theta=theta)
    p = m.price(strike, spot, texp)
    p2 = m.price(strike, spot, texp2)

    demovol = pf.Bsm(sigma=1)   # initialize a Bsm
    impvol = demovol.impvol(p, strike, spot, texp)
    impvol2 = demovol.impvol(p2, strike, spot, texp2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 2. Simultaneous fit of 3/2 model on S&P 500 implied volatility on July31, 2009', fontsize = 14)

    ax[0].plot(strike / spot, impvol, color = 'black')
    ax[0].set_title('3-month S&P 500 implied volatility')
    ax[0].set_xlabel('Relative strike')
    ax[0].set_ylabel('Implied volatility')

    ax[1].plot(strike / spot, impvol2, color = 'black')
    ax[1].set_title('6-month S&P 500 implied volatility')
    ax[1].set_xlabel('Relative strike')
    ax[1].set_ylabel('Implied volatility')
    ax[1].sharey(ax[0])


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
    
    

def plot_figure5(c, v0, max_strike, texp):

    strike = np.linspace(0, max_strike, 1000)[1:]
    p = np.array([])
    for i in strike:
        p = np.r_[p, c.cbar(i)]

    demovol = pf.Bsm(sigma=1)   # initialize a Bsm
    impvol = demovol.impvol(p, strike, v0, texp)

    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    fig.suptitle('Figure 5. Estimate of variance call price and implied volatility in Heston model control variate')

    ax[0].plot(strike, p*1e4, color = 'black')
    ax[0].set_xlabel('Variance Strike')
    ax[0].set_ylabel('Variance Call Price')
    ax[0].set_title('Variance call prices as a function of variance strike')

    ax[1].plot(np.sqrt(strike[50:]), impvol[50:], color = 'black')
    ax[1].set_xlabel('Variance Strike')
    ax[1].set_ylabel('Implied Volatility of variance')
    ax[1].set_title('Implied volatility of variance as a function of volatility strike')
    
def mc(path):
    var_strikes = np.linspace(1e-2, .25, 100)
    nonenan = path[~ np.isnan(path[:, -1]), :]
    m = nonenan.mean(axis=1)
    price = np.zeros(100)
    l = nonenan.shape[0]
    for i, vs in enumerate(var_strikes):
        temp = m - vs
        over0 = temp[temp > 0]
        price[i] = sum(over0)/l
    plt.plot(var_strikes, price)

def b_sT(lbd, s, k, gamma, texp):
    up = 2 * lbd * (np.exp(gamma * (texp - s)) - 1)
    low = (gamma + k) * (np.exp(gamma * (texp - s)) - 1) + 2 * gamma
    return up / low

def lower_l(lbd, v_heston, texp, n=100):
    v0, k, theta, epsilon = v_heston.v0, v_heston.k, v_heston.θ, v_heston.ε
    gamma = np.sqrt(k ** 2 + epsilon ** 2 * 2 * lbd)
    a0T = 0
    for s in np.linspace(0, texp, n):
        bsT = b_sT(lbd, s, k, gamma, texp)
        a0T -= k * bsT * theta * texp / n
    b0T = b_sT(lbd, 0, k, gamma, texp)
    return np.exp(a0T - bsT * v0)

def upper_l(lbd, c0, v_heston, texp):
    l =  lower_l(lbd, v_heston, texp)
    return (l - 1)/lbd ** 2 + c0/lbd

def inv_lap(k, params, g_hat):
    M, M2, a, n, λ_coef, β = params
    temp = 0
    for j in range(0, M2):
        sum = 0
        for l in range(n):
            complex_part = λ_coef[l] + 2*np.pi*j/M2
            complex_num = complex(a, complex_part)
            
            # function g_hat is needed
            g_hat_num = g_hat(complex_num)

            sum += β[l] * g_hat_num
        temp += sum*np.exp(complex(0, 2*np.pi*k*j/M2))
    return temp / M2 * np.exp(a*k)
    