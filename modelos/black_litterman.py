import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def calcular_momentum(train_returns, ventana=63):
    """
    Calcula momentum acumulado de cada acción usando los últimos 'ventana' días.
    """
    momentum = (1 + train_returns.tail(ventana)).prod() - 1
    return momentum


def seleccionar_top_bottom_momentum(train_returns, ventana=63, porcentaje=0.20):
    """
    Selecciona acciones top y bottom según momentum reciente.
    """
    momentum = calcular_momentum(train_returns, ventana)

    n_activos = len(momentum)
    n_grupo = max(1, int(n_activos * porcentaje))

    top = momentum.sort_values(ascending=False).head(n_grupo).index.tolist()
    bottom = momentum.sort_values(ascending=True).head(n_grupo).index.tolist()

    return top, bottom, momentum

def construir_view_momentum(train_returns, top, bottom):
    """
    Construye una view relativa:
    retorno promedio del grupo top momentum
    menos retorno promedio del grupo bottom momentum.
    """

    tickers = list(train_returns.columns)
    n = len(tickers)

    P = np.zeros((1, n))

    peso_top = 1 / len(top)
    peso_bottom = -1 / len(bottom)

    for ticker in top:
        idx = tickers.index(ticker)
        P[0, idx] = peso_top

    for ticker in bottom:
        idx = tickers.index(ticker)
        P[0, idx] = peso_bottom

    # Q es la diferencia histórica reciente observada entre top y bottom
    retorno_top = (1 + train_returns[top].tail(63)).prod().mean() - 1
    retorno_bottom = (1 + train_returns[bottom].tail(63)).prod().mean() - 1

    Q = np.array([(retorno_top - retorno_bottom) / 63])

    # Omega representa incertidumbre de la view
    omega_valor = train_returns[top + bottom].mean(axis=1).var()
    omega_valor = max(omega_valor, 1e-8)
    Omega = np.array([[omega_valor]])

    return P, Q, Omega

def calcular_mu_black_litterman(train_returns, tau=0.05):
    """
    Calcula retornos esperados ajustados por Black-Litterman
    usando una view de momentum.
    """

    tickers = list(train_returns.columns)

    mu_hist = train_returns.mean().values
    #Sigma = train_returns.cov().values
    lw = LedoitWolf()
    lw.fit(train_returns.values)
    Sigma = lw.covariance_

    # Prior simple: usamos mu histórico como punto de partida
    pi = mu_hist

    top, bottom, momentum = seleccionar_top_bottom_momentum(train_returns)

    P, Q, Omega = construir_view_momentum(
        train_returns,
        top,
        bottom
    )

    tauSigma = tau * Sigma

    inv_tauSigma = np.linalg.pinv(tauSigma)
    inv_Omega = np.linalg.pinv(Omega)

    parte_izquierda = inv_tauSigma + P.T @ inv_Omega @ P

    parte_derecha = inv_tauSigma @ pi + P.T @ inv_Omega @ Q

    mu_bl = np.linalg.pinv(parte_izquierda) @ parte_derecha

    return mu_bl, top, bottom, momentum