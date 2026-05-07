import pandas as pd
import numpy as np
import backtesting_mensual_automatico


def grid_search_backtesting(returns, perfiles):

    resultados = []

    train_years_list = [2, 3, 4, 5]
    test_years_list = [0.5, 1, 2]

    for perfil, datos in perfiles.items():

        for train_years in train_years_list:
            for test_years in test_years_list:

                train_days = int(252 * train_years)
                test_months = int(12 * test_years)

                print("\n" + "="*80)
                print(f"Perfil: {perfil} | Train: {train_years} años | Test: {test_years} años")
                print("="*80)

                try:
                    resumen_df = backtesting_mensual_automatico.backtesting_mensual_automatico(
                        returns,
                        perfil,
                        datos["perdida_max_anual"],
                        datos["lambda"],
                        train_days=train_days,
                        test_months=test_months
                    )

                    if resumen_df.empty:
                        continue

                    retorno_total = (1 + resumen_df["retorno_mes"]).prod() - 1

                    abandono = resumen_df["probabilidad_abandono_P1"].mean()
                    aceptacion = resumen_df["probabilidad_aceptacion_P2"].dropna().mean()

                    resultados.append({
                        "perfil": perfil,
                        "train_years": train_years,
                        "test_years": test_years,
                        "retorno_total_pct": retorno_total * 100,
                        "abandono_promedio_pct": abandono * 100,
                        "aceptacion_promedio_pct": aceptacion * 100,
                        "score": retorno_total - abandono  # métrica simple
                    })

                except Exception as e:
                    print("Error:", e)

    resultados_df = pd.DataFrame(resultados)

    resultados_df = resultados_df.sort_values("score", ascending=False)

    print("\n🏆 MEJORES CONFIGURACIONES")
    print(resultados_df.head(10))

    resultados_df.to_csv("grid_search_resultados.csv", index=False)

    return resultados_df