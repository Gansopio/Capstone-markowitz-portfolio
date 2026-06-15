import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp

import modelo_markowitz

def graficar_frontera_eficiente(train_returns):

    lambdas = np.logspace(-2, 3, 100)

    retornos = []
    volatilidades = []
    lambdas_validos = []

    for lam in lambdas:

        model, w = modelo_markowitz.model_markowitz(
            train_returns,
            lam,
            0.40,
            "frontera"
        )

        if model.status == gp.GRB.OPTIMAL:

            mu = train_returns.mean().values
            Sigma = train_returns.cov().values
            n = len(mu)

            pesos = np.array([w[i].X for i in range(n)])

            retorno = np.dot(mu, pesos) * 252
            volatilidad = np.sqrt(pesos @ Sigma @ pesos) * np.sqrt(252)

            retornos.append(retorno)
            volatilidades.append(volatilidad)
            lambdas_validos.append(lam)

    retornos = np.array(retornos)
    volatilidades = np.array(volatilidades)
    lambdas_validos = np.array(lambdas_validos)

    lambdas_destacados = [0.01, 0.1, 1, 10, 100, 1000]

    plt.figure(figsize=(12,7))

    plt.plot(volatilidades, retornos, linewidth=1)
    plt.scatter(volatilidades, retornos, s=25)

    for i, lam in enumerate(lambdas_validos):
        plt.annotate(
            f"{lam:.2g}",
            (volatilidades[i], retornos[i]),
            textcoords="offset points",
            xytext=(3,3),
            fontsize=7
        )

    plt.xlabel("Volatilidad anual")
    plt.ylabel("Retorno esperado anual")
    plt.title("Frontera eficiente de Markowitz con lambdas")
    plt.grid(True)
    plt.show()