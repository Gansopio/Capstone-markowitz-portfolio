import pandas as pd
import numpy as np
from gurobipy import GRB
import modelo_markowitz


def backtesting_mensual_con_decision(
    returns,
    perfil,
    perdida_max_anual,
    lam,
    train_days=252*4,
    test_months=12
):

    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    resumen = []

    fecha_inicio_backtest = returns.index[train_days]
    meses_disponibles = returns[returns.index >= fecha_inicio_backtest].index.to_period("M").unique()
    meses_test = meses_disponibles[:test_months]

    pesos_actuales = None

    for mes in meses_test:

        test_returns = returns[returns.index.to_period("M") == mes]
        fecha_inicio_test = test_returns.index[0]

        train_returns = returns[returns.index < fecha_inicio_test].tail(train_days)

        print("\n" + "="*80)
        print("MES TEST:", mes)
        print("Train:", train_returns.index[0], "a", train_returns.index[-1])
        print("Test :", test_returns.index[0], "a", test_returns.index[-1])
        print("="*80)

        model, w = modelo_markowitz.model_markowitz(
            train_returns,
            lam,
            perdida_max_anual,
            perfil
        )

        if model.status != GRB.OPTIMAL:
            print("No se encontró solución óptima")
            continue

        mu = train_returns.mean().values
        Sigma = train_returns.cov().values
        tickers = list(train_returns.columns)
        n = len(tickers)

        pesos_nuevos = np.array([w[i].X for i in range(n)])

        retorno_esperado_nuevo = np.dot(mu, pesos_nuevos)
        varianza_nueva = pesos_nuevos @ Sigma @ pesos_nuevos
        volatilidad_nueva = np.sqrt(varianza_nueva)
        escenario_favorable = retorno_esperado_nuevo + volatilidad_nueva
        escenario_desfavorable = retorno_esperado_nuevo - volatilidad_nueva

        if pesos_actuales is None:
            pesos_actuales = pesos_nuevos
            decision = "primer_portafolio"

        else:
            retorno_esperado_actual = np.dot(mu, pesos_actuales)
            varianza_actual = pesos_actuales @ Sigma @ pesos_actuales
            volatilidad_actual = np.sqrt(varianza_actual)
            escenario_favorable_actual = retorno_esperado_actual + volatilidad_actual
            escenario_desfavorable_actual = retorno_esperado_actual - volatilidad_actual

            print("\n PORTAFOLIO ACTUAL\n")
            print("Retorno esperado anual:", retorno_esperado_actual*252)

            print("\nRetorno esperado mensual aprox:", retorno_esperado_actual * 21)
            print("Escenario favorable mensual aprox:", escenario_favorable_actual*21)
            print("Escenario desfavorable mensual aprox:", escenario_desfavorable_actual*21)

            print("\nVolatilidad diaria:", volatilidad_actual)
            print("Peso máximo:", pesos_actuales.max())
            print("Activos usados:", np.sum(pesos_actuales > 1e-6))

            print("\n NUEVO PORTAFOLIO PROPUESTO\n")
            print("Retorno esperado anual:", retorno_esperado_nuevo*252)

            print("\nRetorno esperado mensual aprox:", retorno_esperado_nuevo * 21)
            print("Escenario favorable mensual aprox:", escenario_favorable*21)
            print("Escenario desfavorable mensual aprox:", escenario_desfavorable*21)

            print("\nVolatilidad diaria:", volatilidad_nueva)
            print("Peso máximo:", pesos_nuevos.max())
            print("Activos usados:", np.sum(pesos_nuevos > 1e-6))
            

            opcion = input(
                "\n¿Aceptar nuevo portafolio? Escribe 's' para aceptar o ENTER para mantener anterior: "
            ).lower().strip()

            if opcion == "s":
                pesos_actuales = pesos_nuevos
                decision = "acepta_nuevo"
            else:
                decision = "mantiene_anterior"

        portfolio_test_returns = test_returns @ pesos_actuales
        retorno_mes = (1 + portfolio_test_returns).prod() - 1

        print("\n Resultado real del mes usando portafolio elegido")
        print("Retorno mensual real:", retorno_mes)
        print("Retorno mensual real (%):", retorno_mes * 100)
        print("Decisión:", decision)

        resumen.append({
            "perfil": perfil,
            "lambda": lam,
            "perdida_max_anual": perdida_max_anual,
            "mes_test": str(mes),
            "train_inicio": train_returns.index[0],
            "train_fin": train_returns.index[-1],
            "test_inicio": test_returns.index[0],
            "test_fin": test_returns.index[-1],
            "decision": decision,
            "retorno_esperado_nuevo": retorno_esperado_nuevo,
            "volatilidad_nueva": volatilidad_nueva,
            "retorno_mes": retorno_mes,
            "retorno_mes_pct": retorno_mes * 100,
            "num_activos_actuales": np.sum(pesos_actuales > 1e-6),
            "peso_maximo_actual": pesos_actuales.max()
        })
        input("\nPresione ENTER para continuar al siguiente mes...")

    resumen_df = pd.DataFrame(resumen)

    retorno_acumulado = (1 + resumen_df["retorno_mes"]).prod() - 1

    print("\n" + "="*80)
    print("RESULTADO FINAL DEL BACKTESTING")
    print("="*80)

    print("\n Retorno acumulado total del año:")
    print(retorno_acumulado)

    print("\n Retorno acumulado total del año (%):")
    print(retorno_acumulado * 100)

    print("\n Meses evaluados:")
    print(len(resumen_df))

    print("\n Perfil:")
    print(perfil)

    print("\n Lambda utilizado:")
    print(lam)


    resumen_df.to_csv(
        "resumen_backtesting_mensual_decision.csv",
        index=False
    )
    return resumen_df