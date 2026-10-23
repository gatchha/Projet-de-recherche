"""Un skew qui explose a court terme, sans aucun saut de prix.

Modele (volatilite rugueuse, prix continu) :
  dS/S = sigma_t (rho dW + sqrt(1-rho^2) dW')
  sigma_t = xi * exp(eta * X_t - eta^2/2 * Var(X_t)),  X_t = sqrt(2H) * int_0^t (t-s)^(H-1/2) dW_s
H = 0.5 : volatilite brownienne classique. H = 0.1 : volatilite rugueuse.
"""
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
XI, ETA, RHO = 0.2, 1.0, -0.7
N_STEPS, N_PATHS = 64, 400_000


def bs_price(k, tau, vol, call):
    d1 = (-k + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    K = np.exp(k)
    return norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - norm.cdf(-d1)


def implied_vol(price, k, tau, call):
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(k, tau, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def simulate(H, tau, n=N_STEPS, paths=N_PATHS, keep=0):
    dt = tau / n
    m = np.arange(1, n + 1)
    w = ((m * dt) ** (H + 0.5) - ((m - 1) * dt) ** (H + 0.5)) / ((H + 0.5) * dt)
    var = 2 * H * dt * np.cumsum(w**2)               # Var(X_i), i = 1..n
    T = np.zeros((n, n))
    for i in range(1, n):                              # X_i = sum_{j<i} w_{i-j} dW_j
        T[i, :i] = np.sqrt(2 * H) * w[:i][::-1]
    half = paths // 2
    Z = rng.standard_normal((half, n))
    Z = np.vstack([Z, -Z])                             # antithetiques
    Zp = rng.standard_normal((half, n))
    Zp = np.vstack([Zp, -Zp])
    X = (Z * np.sqrt(dt)) @ T.T
    v = np.concatenate([[0.0], var[:-1]])
    sig = XI * np.exp(ETA * X - 0.5 * ETA**2 * v)
    dB = RHO * Z + np.sqrt(1 - RHO**2) * Zp
    dlogS = -0.5 * sig**2 * dt + sig * np.sqrt(dt) * dB
    logS = np.cumsum(dlogS, axis=1)
    S_T = np.exp(logS[:, -1])
    S_T /= S_T.mean()                                  # forward = 1
    return S_T, (np.exp(logS[:keep]) if keep else None)


def smile(S_T, tau, ks):
    out = []
    for k in ks:
        K = np.exp(k)
        call = k >= 0
        pay = np.maximum(S_T - K, 0) if call else np.maximum(K - S_T, 0)
        out.append(implied_vol(pay.mean(), k, tau, call))
    return np.array(out)


def atm_skew(H, tau):
    S_T, _ = simulate(H, tau)
    kk = 0.3 * XI * np.sqrt(tau)
    iv = smile(S_T, tau, np.array([-kk, kk]))
    return (iv[1] - iv[0]) / (2 * kk)


if __name__ == "__main__":
    taus = np.array([0.003, 0.01, 0.03, 0.1, 0.3])
    res = {H: np.array([atm_skew(H, t) for t in taus]) for H in (0.1, 0.5)}
    for H in res:
        slope = np.polyfit(np.log(taus), np.log(np.abs(res[H])), 1)[0]
        print(f"H={H}: pente log-log mesuree = {slope:.2f}  (theorie H-1/2 = {H-0.5:.2f})")
        print("  skew ATM :", np.round(res[H], 3))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    S_T, paths = simulate(0.1, 1.0, keep=8)
    t = np.linspace(0, 1, N_STEPS + 1)
    for p in paths:
        ax[0].plot(t, np.concatenate([[1], p]), lw=1)
    ax[0].set(title="Prix simules (H = 0,1) : continus, aucun saut", xlabel="temps", ylabel="prix")

    tau = 0.01
    ks = np.linspace(-0.12, 0.05, 10)  # au-dela, trop peu de trajets atteignent le strike (prix MC = 0)
    for H, c in ((0.5, "tab:blue"), (0.1, "tab:red")):
        S_T, _ = simulate(H, tau)
        ax[1].plot(ks, smile(S_T, tau, ks), "o-", color=c, label=f"H = {H}")
    ax[1].set(title=f"Volatilite implicite selon k (echeance tau = {tau})", xlabel="k = log(K/F)", ylabel="vol implicite")
    ax[1].legend()

    for H, c in ((0.5, "tab:blue"), (0.1, "tab:red")):
        ax[2].loglog(taus, np.abs(res[H]), "o-", color=c, label=f"H = {H} (simule)")
        ref = np.abs(res[H][-1]) * (taus / taus[-1]) ** (H - 0.5)
        ax[2].loglog(taus, ref, "--", color=c, alpha=.5, label=f"pente theorique H-1/2 = {H-0.5:.1f}")
    ax[2].set(title="Taille du skew ATM selon l'echeance", xlabel="echeance tau", ylabel="|skew ATM|")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("/home/user/Projet-de-recherche/simulations/skew_sans_saut.png", dpi=130)
