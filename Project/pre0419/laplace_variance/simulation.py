import numpy as np


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


def mc_simulation(path):
    var_strikes = np.linspace(1e-2, .25, 100)
    nonenan = path[~ np.isnan(path[:, -1]), :]
    m = nonenan.mean(axis=1)
    price = np.zeros(100)
    l = nonenan.shape[0]
    for i, vs in enumerate(var_strikes):
        temp = m - vs
        over0 = temp[temp > 0]
        price[i] = sum(over0)/l
    return var_strikes, price