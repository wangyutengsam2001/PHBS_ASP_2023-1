import numpy as np
import matplotlib.pyplot as plt
import pyfeng as pf
from scipy.special import gamma


class variance_heston():
    def __init__(self, v0_H, k_H, θ_H, ε_H, ρ_H):
        self.v0, self.k, self.θ, self.ε, self.ρ = v0_H, k_H, θ_H, ε_H, ρ_H

class variance_32():
    def __init__(self, v0_32, k_32, θ_32, ε_32, ρ_32):
        self.v0, self.k, self.θ, self.ε, self.ρ = v0_32, k_32, θ_32, ε_32, ρ_32


def plot_figure3(v_heston, v_32, n_observe, dt, texp):
    
    n_dt = int(texp / dt)
    t = np.arange(0, dt+1, dt)

    v0_H, k_H, θ_H, ε_H = v_heston.v0, v_heston.k, v_heston.θ, v_heston.ε
    v0_32, k_32, θ_32, ε_32 = v_32.v0, v_32.k, v_32.θ, v_32.ε

    Z1 = np.random.standard_normal((n_observe, n_dt))
    Z2 = np.random.standard_normal((n_observe, n_dt))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 3. Instantaneous volatility paths', fontsize = 14)

    # Heston Model Sigma path
    for time in range(n_observe):
        vt = [v0_H]
        for _ in range(n_dt):
            dvt = k_H * (θ_H - vt[-1]) * dt + ε_H * np.sqrt(vt[-1]) * Z1[time, _] * np.sqrt(dt)
            vt.append(vt[-1] + dvt)
        ax[0].plot(t, np.sqrt(vt), color = 'black', linewidth = 0.5)
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Instantaneous Volatility')
    ax[0].set_title('Heston Model')

    # 3/2 Model Sigma Path
    for time in range(n_observe):
        vt = [v0_32]
        sign = 1 # path usable or not
        for _ in range(n_dt):
            dvt = vt[-1] * (k_32 * (θ_32 - vt[-1]) * dt + ε_32 * np.sqrt(vt[-1]) * Z2[time, _] * np.sqrt(dt))
            vt.append(vt[-1] + dvt)
            if vt[-1] < 0:
                sign = 0
                break
        if sign == 1:
            ax[1].plot(t, np.sqrt(vt), color = 'black', linewidth = 0.5)
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

    ax[1].plot(strike, I_32, color = 'black')
    ax[1].plot(strike, impvol_32, color = 'black', linestyle = '--')
    ax[1].legend(['Medvedev and Scaillet approximation', 'True model implied volatility'])
    ax[1].set_title('3/2 Model')
    ax[1].sharey(ax[0])


def L_C0_heston(v_heston, λ, T):

    v0, k, θ, ε = v_heston.v0, v_heston.k, v_heston.θ, v_heston.ε

    λ /= T
    γ = np.sqrt(k**2 + 2*λ*ε**2)
    b_0T = 2*λ*(np.exp(γ*T)-1)/((γ+k)*(np.exp(γ*T)-1) + 2*γ)
    α = γ + k
    β = γ - k

    a_0T = (k*θ/ε**2) * (k-γ)*T

    dt = 1e-3
    n_dt = int(T / dt)
    t = np.arange(0, T, dt)
    for k in range(n_dt-1):
        part1 = α*np.exp(γ*(T-t[k+1]))
        part2 = β*np.exp(-γ*dt)
        a_0T -= (2*k*θ/ε**2) * np.log((part1+part2) / (part1+β))

    L = np.exp(a_0T - b_0T*v0)
    return L


def L_C0_32(v_32, λ, T):

    v0, k, θ, ε = v_32.v0, v_32.k, v_32.θ, v_32.ε

    λ /= T
    p = -k
    q = λ
    α = -(1/2 + p/ε**2) + np.sqrt((1/2 + p/ε**2)**2 + 2*q/ε**2)
    γ = 2*(α+1-p/ε**2)
    y = v0/(k*θ)*(np.exp(k*θ*T) - 1)

    # scipy.special.hyp1f1 does not apply to complex number
    # Instead we expand M to 3rd term and it is precision enough
    z = -2/(ε**2*y)
    M = 1 + α/γ*z + α*(α+1)/(γ*(γ+1))*z**2/2 + α*(α+1)*(α+2)/(γ*(γ+1)*(γ+2))*z**3/6

    L = gamma(γ-α)/gamma(γ) * (2/(ε**2*y))**α * M
    return L