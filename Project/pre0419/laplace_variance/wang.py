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


def C0_simulation_heston(v_heston, n_observe, dt, texp):
    '''
    Simulate sigma path in Heston model with MC method
    Return the sigma path and C0 estimation
    '''

    n_dt = int(texp / dt)
    v0_H, k_H, θ_H, ε_H = v_heston.v0, v_heston.k, v_heston.θ, v_heston.ε
    Z = np.random.standard_normal((n_observe, n_dt))
    
    path = np.zeros((n_observe, n_dt+1))
    for time in range(n_observe):
        vt = [v0_H]
        for _ in range(n_dt):
            dvt = k_H * (θ_H - vt[-1]) * dt + ε_H * np.sqrt(vt[-1]) * Z[time, _] * np.sqrt(dt)
            vt.append(vt[-1] + dvt) 
        path[time,:] = vt

    C0 = np.nanmean(np.nansum(path, axis = 1) / texp *dt)
    return path, C0


def C0_simulation_32(v_32, n_observe, dt, texp):
    '''
    Simulate sigma path in 3/2 model with MC method
    Return the sigma path and C0 estimation
    '''

    n_dt = int(texp / dt)
    v0_32, k_32, θ_32, ε_32 = v_32.v0, v_32.k, v_32.θ, v_32.ε
    Z = np.random.standard_normal((n_observe, n_dt))

    path = np.zeros((n_observe, n_dt+1))
    for time in range(n_observe):
        vt = [v0_32]
        sign = 1 # path usable or not
        for _ in range(n_dt):
            dvt = vt[-1] * (k_32 * (θ_32 - vt[-1]) * dt + ε_32 * np.sqrt(vt[-1]) * Z[time, _] * np.sqrt(dt))
            vt.append(vt[-1] + dvt)
            if vt[-1] < 0:
                sign = 0
                break
        if sign == 1:
            path[time,:] = vt

    C0 = np.nanmean(np.nansum(path, axis = 1) / texp *dt)
    return path, C0


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


def direct_Lk(λ, C0, model_type, v, T):
    """
    To directly calculate L(λ) = (\mathcal{L}(λ)-1)/λ**2 + C(0)/λ
    input:
        λ: a complex number
        C0: C0 estimation for Heston model or 3/2 model
        model_type: 'Heston' or '3/2'
        v: v_heston or v_32
        T: maturity
    """
    assert( model_type in ['Heston', '3/2'] )
    if model_type == 'Heston':
        math_L = L_C0_heston(v, λ, T)
    else:
        math_L = L_C0_32(v, λ, T)
    L = (math_L-1)/λ**2 + C0/λ
    
    return L


def parameters_inv(times = 5):
    """
    input:
        times: a hyperparameter for M, M = 2**times
    output:
        parameters for GQ-FFT
    """
    M = 2**times
    M2 = 8*M
    a = 44/M2

    n = 16
    λ_coef = np.array([4.4409e-016, 6.2832, 12.5664, 18.8503, 25.2872, 34.2970, 56.1726, 170.5331])
    λ_coef = np.append(λ_coef, -λ_coef[::-1] - 2*np.pi)
    β = np.array([1, 1.0000, 1.0000, 1.0008, 1.0958, 2.0069, 5.9428, 54.9537])
    β = np.append(β, β[::-1])

    return M, M2, a, n, λ_coef, β


def inv_algo_ghat(params, g_hat):
    """
    gives the inverse Laplace transform of g_hat
    """
    M, M2, a, n, λ_coef, β = params

    res = list()
    for k in range(M):
        temp = complex(0, 0)
        for j in range(0, M2):
            sum = complex(0, 0)
            for l in range(0, n):
                complex_part = λ_coef[l] + 2*np.pi*j/M2
                complex_num = complex(a, complex_part)
                
                # function g_hat is needed
                g_hat_num = g_hat(complex_num)

                sum += β[l] * g_hat_num / (complex_num * complex_num)
            temp += sum*np.exp(complex(0, 2*np.pi*k*j/M2))
        res.append(temp / M2 * np.exp(a*k))

    return res


def invlt(t, fs, sigma = 0.2, omega = 200):
    """
    inverse Laplace transform using trapizod method
    """
    nint = omega * 50

    omegadim = np.linspace(0, omega, nint+1, endpoint=True)
    y = [(np.exp(1j*o*t) * fs(sigma+1j*o)).real for o in omegadim]
    y_left = y[:-1]
    y_right = y[0:]
    T = sum(y_right + y_left) * omega/nint
    return np.exp(sigma*t) * T/ np.pi / 2