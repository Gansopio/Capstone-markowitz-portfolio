import numpy as np


def simular_montecarlo_normal(
    retorno_esperado_diario,
    volatilidad_diaria,
    dias=252,
    n_simulaciones=10000,
    capital_inicial=1000,
    random_state=42
):
    np.random.seed(random_state)

    simulaciones = []

    for _ in range(n_simulaciones):
        retornos_simulados = np.random.normal(
            loc=retorno_esperado_diario,
            scale=volatilidad_diaria,
            size=dias
        )

        capital_final = capital_inicial * np.prod(1 + retornos_simulados)
        retorno_total = capital_final / capital_inicial - 1

        simulaciones.append(retorno_total)

    simulaciones = np.array(simulaciones)

    p5 = np.percentile(simulaciones, 5)
    p50 = np.percentile(simulaciones, 50)
    p95 = np.percentile(simulaciones, 95)

    return {
        "escenario_desfavorable_pct": p5 * 100,
        "escenario_neutro_pct": p50 * 100,
        "escenario_favorable_pct": p95 * 100,
        "capital_desfavorable": capital_inicial * (1 + p5),
        "capital_neutro": capital_inicial * (1 + p50),
        "capital_favorable": capital_inicial * (1 + p95),
    }