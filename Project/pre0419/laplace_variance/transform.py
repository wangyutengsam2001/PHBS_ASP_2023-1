import numpy as np
from scipy.special import gamma


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
    # Instead we expand M to 3rd term and it is precise enough
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
    return np.exp(a0T - b0T * v0)

def upper_l(lbd, c0, v_heston, texp):
    l =  lower_l(lbd, v_heston, texp)
    return (l - 1)/lbd ** 2 + c0/lbd


def inv_lap(k, params, g_hat):
    _, M2, a, n, λ_coef, β = params
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


def pricing_cv(var_strikes, c, price, params, g_hat):
    max_strike = var_strikes[-1] + 0.05
    cv = np.array([c.cbar(k) for k in var_strikes])
    const = c.cbar(max_strike) + inv_lap(max_strike, params, g_hat)
    price = price + cv - const
    return 0.1 * price.real