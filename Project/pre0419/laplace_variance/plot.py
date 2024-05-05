import numpy as np
import pyfeng as pf
import matplotlib.pyplot as plt


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


def plot_figure3(path_heston, path_32, dt):

    t = np.arange(0, dt+1, dt)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 3. Instantaneous volatility paths', fontsize = 14)

    # Heston Model Sigma path
    for time in range(path_heston.shape[0]):
        ax[0].plot(t, path_heston[time,:]**0.5, color = 'black', linewidth = 0.5)
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Instantaneous Volatility')
    ax[0].set_title('Heston Model')

    # 3/2 Model Sigma Path
    for time in range(path_32.shape[0]):
        ax[1].plot(t, path_32[time,:]**0.5, color = 'black', linewidth = 0.5)
    ax[1].set_ylim(0, 1.5)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Instantaneous Volatility')
    ax[1].sharey(ax[0])
    ax[1].set_title('3/2 Model')


def plot_figure4(v_heston, v_32, strike, spot, texp):

    # Heston
    v0, k, θ, ε, ρ = v_heston.v0, v_heston.k, v_heston.θ, v_heston.ε, v_heston.ρ
    X = np.log(strike/spot*np.exp(0.03*texp))
    ## MS Expansion
    sqrt_v0 =  np.sqrt(v0)
    I_heston = sqrt_v0 + ρ*ε*X/(4*sqrt_v0) + (1-5*ρ**2/2)*(ε**2*X**2)/(24*sqrt_v0**3) + \
            (k*(θ-v0)/(4*sqrt_v0) + ρ*ε*sqrt_v0/8 + ρ**2*ε**2/(96*sqrt_v0) - ε**2/(24*sqrt_v0)) * texp
    ## True model
    m = pf.HestonFft(v0, vov = ε, mr = k, rho = ρ, theta = θ)
    p = m.price(strike, spot, texp)
    demovol = pf.Bsm(sigma = 1)
    impvol_heston = demovol.impvol(p, strike, spot, texp)

    # 3/2
    v0, k, θ, ε, ρ = v_32.v0, v_32.k, v_32.θ, v_32.ε, v_32.ρ
    X = np.log(strike/spot*np.exp(0.33*texp))
    ## MS Expansion
    sqrt_v0 =  np.sqrt(v0)
    I_32 = sqrt_v0 + ρ*ε*X*sqrt_v0/4 + (1-ρ**2/2)*(ε**2*X**2*sqrt_v0)/24 + \
            (k*(θ-v0)/4 + ρ*ε*v0/8 + 7*ρ**2*ε**2*v0/96 - ε**2*v0/24) * (texp*sqrt_v0)
    ## True model
    m = pf.Sv32Fft(v0, vov = ε, mr = k, rho = ρ, theta = θ)
    p = m.price(strike, spot, texp)
    demovol = pf.Bsm(sigma = 1)
    impvol_32 = demovol.impvol(p, strike, spot, texp)

    # plot
    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    fig.suptitle('Figure 4. Comparison between true model implied volatility and Medvedev and Scaillet approximation', fontsize = 14)

    ax[0].plot(strike, I_heston, color = 'black')
    ax[0].plot(strike, impvol_heston, color = 'black', linestyle = '--')
    ax[0].legend(['Medvedev and Scaillet approximation', 'True model implied volatility'])
    ax[0].set_title('Heston Model')
    ax[0].set_xlabel('Relative strike')
    ax[0].set_ylabel('Implied volatility')

    ax[1].plot(strike, I_32, color = 'black')
    ax[1].plot(strike, impvol_32, color = 'black', linestyle = '--')
    ax[1].legend(['Medvedev and Scaillet approximation', 'True model implied volatility'])
    ax[1].set_title('3/2 Model')
    ax[1].sharey(ax[0])
    ax[1].set_xlabel('Relative strike')
    ax[1].set_ylabel('Implied volatility')


def plot_figure5(vshes, phes, vs32, p32):

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 5. Pricing Variance Options with MC Simulation', fontsize = 14)
    
    ax[0].set_xlabel('Variance Strike')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Heston Model')
    ax[0].plot(vshes, phes, color = 'black')
    
    ax[1].set_xlabel('Variance Strike')
    ax[1].set_ylabel('Price')
    ax[1].set_title('3/2 Model')
    ax[1].plot(vs32, p32, color = 'black')
    

def plot_figure6(var_strikes, price_heston, price_32):
    fig, ax = fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 6. Pricing Variance Options with Direct Laplace Transform', fontsize = 14)

    ax[0].set_xlabel('Variance Strike')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Heston Model')
    ax[0].plot(var_strikes, np.array(price_heston).real, color = 'black')
    
    ax[1].set_xlabel('Variance Strike')
    ax[1].set_ylabel('Price')
    ax[1].set_title('3/2 Model')
    ax[1].plot(var_strikes, np.array(price_32).real[::-1], color = 'black')


def plot_figure7(c, v0, max_strike, texp):

    strike = np.linspace(0, max_strike, 1000)[1:]
    p = np.array([])
    for i in strike:
        p = np.r_[p, c.cbar(i)]

    demovol = pf.Bsm(sigma=1)   # initialize a Bsm
    impvol = demovol.impvol(p, strike, v0, texp)

    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    fig.suptitle('Figure 7. Estimate of variance call price and implied volatility in Heston model control variate')

    ax[0].plot(strike, p, color = 'black')
    ax[0].set_xlabel('Variance Strike')
    ax[0].set_ylabel('Variance Call Price')
    ax[0].set_title('Variance call prices as a function of variance strike')

    ax[1].plot(np.sqrt(strike[50:]), impvol[50:], color = 'black')
    ax[1].set_xlabel('Variance Strike')
    ax[1].set_ylabel('Implied Volatility of variance')
    ax[1].set_title('Implied volatility of variance as a function of volatility strike')


def plot_figure8(var_strikes, price_heston, price_32):
    fig, ax = fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 8. Pricing Variance Options with Laplace Transform and Control Variate', fontsize = 14)

    ax[0].set_xlabel('Variance Strike')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Heston Model')
    ax[0].plot(var_strikes, np.array(price_heston).real, color = 'black')
    
    ax[1].set_xlabel('Variance Strike')
    ax[1].set_ylabel('Price')
    ax[1].set_title('3/2 Model')
    ax[1].plot(var_strikes, np.array(price_32).real, color = 'black')