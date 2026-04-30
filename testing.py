import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def testing(model, w, train_returns, test_returns, perfil, R_objetivo_anual):
    mu = train_returns.mean().values
    Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)
    n = len(tickers)

    if model.status == GRB.OPTIMAL:
        pesos = np.array([w[i].X for i in range(n)])

        retorno_esperado = np.dot(mu, pesos)
        varianza = pesos @ Sigma @ pesos
        volatilidad = np.sqrt(varianza)

        portfolio_test_returns = test_returns @ pesos
        retorno_acumulado_test = (1 + portfolio_test_returns).prod() - 1

        resultado = pd.DataFrame({
            "Ticker": tickers,
            "Peso": pesos
        })

        resultado = resultado[resultado["Peso"] > 1e-6]
        resultado = resultado.sort_values("Peso", ascending=False)

        print("\n" + "="*80)
        print(f"👤 PERFIL: {perfil}")
        print("="*80)

        print("\n📊 PORTAFOLIO ÓPTIMO")
        print(resultado)

        print("\n🎯 Pérdida máxima anual aceptada:", R_objetivo_anual)

        print("\n📈 Retorno esperado train:", retorno_esperado)
        print("📉 Varianza train:", varianza)
        print("📊 Volatilidad train:", volatilidad)

        print("\n🧪 TESTING")
        print("Fecha inicio test:", test_returns.index[0])
        print("Fecha fin test:", test_returns.index[-1])
        print("Retorno acumulado test:", retorno_acumulado_test)
        print("Retorno acumulado test (%):", retorno_acumulado_test * 100)

        return {
            "perfil": perfil,
            "perdida_maxima_anual": R_objetivo_anual,
            "retorno_esperado_train": retorno_esperado,
            "varianza_train": varianza,
            "volatilidad_train": volatilidad,
            "test_inicio": test_returns.index[0],
            "test_fin": test_returns.index[-1],
            "retorno_acumulado_test": retorno_acumulado_test,
            "num_activos_usados": np.sum(pesos > 1e-6),
            "peso_maximo": pesos.max()
        }

    else:
        print(f"\n❌ No se encontró solución óptima para perfil: {perfil}")
        return {
            "perfil": perfil,
            "perdida_maxima_anual": R_objetivo_anual,
            "error": "No óptimo"
        }