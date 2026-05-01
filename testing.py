import pandas as pd
import numpy as np
from gurobipy import GRB

def testing(model, w, train_returns, test_returns, perfil, perdida_max_anual):
    
    mu = train_returns.mean().values
    Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)
    n = len(tickers)

    if model.status == GRB.OPTIMAL:
        pesos = np.array([w[i].X for i in range(n)])

        retorno_esperado_diario = np.dot(mu, pesos)
        varianza_diaria = pesos @ Sigma @ pesos
        volatilidad_diaria = np.sqrt(varianza_diaria)

        retorno_esperado_anual = retorno_esperado_diario * 252
        volatilidad_anual = volatilidad_diaria * np.sqrt(252)

        portfolio_test_returns = test_returns @ pesos
        retorno_acumulado_test = (1 + portfolio_test_returns).prod() - 1

        dias_test = len(test_returns)
        retorno_anualizado_test = (1 + retorno_acumulado_test) ** (252 / dias_test) - 1

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

        print("\n🎯 Pérdida máxima anual aceptada:", perdida_max_anual)

        print("\n📈 Retorno esperado diario train:", retorno_esperado_diario)
        print("📈 Retorno esperado anual train:", retorno_esperado_anual)
        print("📉 Varianza diaria train:", varianza_diaria)
        print("📊 Volatilidad diaria train:", volatilidad_diaria)
        print("📊 Volatilidad anual train:", volatilidad_anual)

        print("\n🧪 TESTING")
        print("Fecha inicio test:", test_returns.index[0])
        print("Fecha fin test:", test_returns.index[-1])
        print("Retorno acumulado test:", retorno_acumulado_test)
        print("Retorno acumulado test (%):", retorno_acumulado_test * 100)
        print("Retorno anualizado test:", retorno_anualizado_test)
        print("Retorno anualizado test (%):", retorno_anualizado_test * 100)

        return {
            "perfil": perfil,
            "perdida_maxima_anual": perdida_max_anual,
            "retorno_esperado_diario_train": retorno_esperado_diario,
            "retorno_esperado_anual_train": retorno_esperado_anual,
            "varianza_diaria_train": varianza_diaria,
            "volatilidad_diaria_train": volatilidad_diaria,
            "volatilidad_anual_train": volatilidad_anual,
            "test_inicio": test_returns.index[0],
            "test_fin": test_returns.index[-1],
            "retorno_acumulado_test": retorno_acumulado_test,
            "retorno_anualizado_test": retorno_anualizado_test,
            "num_activos_usados": np.sum(pesos > 1e-6),
            "peso_maximo": pesos.max()
        }

    else:
        print(f"\n❌ No se encontró solución óptima para perfil: {perfil}")
        return {
            "perfil": perfil,
            "perdida_maxima_anual": perdida_max_anual,
            "error": "No óptimo"
        }