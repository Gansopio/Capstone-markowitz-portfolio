import pandas as pd
import numpy as np
from gurobipy import GRB
from sklearn.covariance import LedoitWolf
import modelos.montecarlo as montecarlo 

def testing(model, w, train_returns, test_returns, perfil, perdida_max_anual, mu_personalizado=None):
    
    if mu_personalizado is not None:
        mu = mu_personalizado
    else:
        mu = train_returns.mean().values
    
    lw = LedoitWolf()
    lw.fit(train_returns.values)
    Sigma = lw.covariance_
    # Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)
    n = len(tickers)

    if model.status == GRB.OPTIMAL:
        pesos = np.array([w[i].X for i in range(n)])
        peso_caja = 1 - pesos.sum()

        retorno_esperado_diario = np.dot(mu, pesos)
        varianza_diaria = pesos @ Sigma @ pesos
        volatilidad_diaria = np.sqrt(varianza_diaria)

        escenarios_mc = montecarlo.simular_montecarlo_normal(
            retorno_esperado_diario,
            volatilidad_diaria,
            dias=252,
            n_simulaciones=10000,
            capital_inicial=1000
)

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
        print(f"PERFIL: {perfil}")
        print("="*80)

        print("\n PORTAFOLIO ÓPTIMO")
        
        print(resultado)
        print("\n Caja chica")
        print("Peso en caja:", round(peso_caja, 4))
        print("Caja chica USD:", round(peso_caja * 1000, 2))

        print("\n Escenarios Monte Carlo Normal")
        print("Desfavorable P5 (%):", round(escenarios_mc["escenario_desfavorable_pct"], 2))
        print("Neutro P50 (%):", round(escenarios_mc["escenario_neutro_pct"], 2))
        print("Favorable P95 (%):", round(escenarios_mc["escenario_favorable_pct"], 2))

        print("Capital desfavorable USD:", round(escenarios_mc["capital_desfavorable"], 2))
        print("Capital neutro USD:", round(escenarios_mc["capital_neutro"], 2))
        print("Capital favorable USD:", round(escenarios_mc["capital_favorable"], 2))

        print("\n Pérdida máxima anual aceptada:", perdida_max_anual)

        print("\n Retorno esperado diario train:", retorno_esperado_diario)
        print(" Retorno esperado anual train:", retorno_esperado_anual)
        print(" Varianza diaria train:", varianza_diaria)
        print(" Volatilidad diaria train:", volatilidad_diaria)
        print(" Volatilidad anual train:", volatilidad_anual)

        print("\n TESTING")
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
            "peso_maximo": pesos.max(),
            "caja_chica_usd": peso_caja * 1000, 
            "escenario_desfavorable_pct": escenarios_mc["escenario_desfavorable_pct"],
            "escenario_neutro_pct": escenarios_mc["escenario_neutro_pct"],
            "escenario_favorable_pct": escenarios_mc["escenario_favorable_pct"],
            "capital_desfavorable": escenarios_mc["capital_desfavorable"],
            "capital_neutro": escenarios_mc["capital_neutro"],
            "capital_favorable": escenarios_mc["capital_favorable"],
        }

    else:
        print(f"\n No se encontró solución óptima para perfil: {perfil}")
        return {
            "perfil": perfil,
            "perdida_maxima_anual": perdida_max_anual,
            "error": "No óptimo"
        }