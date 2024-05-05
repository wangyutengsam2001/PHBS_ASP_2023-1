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