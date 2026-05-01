import gurobipy as gp
from gurobipy import GRB 

def model_markowitz(train_returns, lam, perdida_max_anual, perfil):

    mu = train_returns.mean().values
    Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)

    n = len(tickers)

    # pérdida máxima diaria
    perdida_max_diaria = perdida_max_anual / 252

    model = gp.Model(f"Markowitz_lambda_{lam}_{perfil}")
    model.Params.OutputFlag = 0

    w = model.addVars(n, lb=0, name="w")

    model.addConstr(
        gp.quicksum(w[i] for i in range(n)) == 1,
        name="presupuesto"
    )

    portfolio_return = gp.quicksum(
        mu[i] * w[i]
        for i in range(n)
    )

    # restricción de pérdida esperada
    model.addConstr(
        portfolio_return >= -perdida_max_diaria,
        name="perdida_esperada_maxima"
    )

    portfolio_variance = gp.quicksum(
        w[i] * Sigma[i, j] * w[j]
        for i in range(n)
        for j in range(n)
    )

    model.setObjective(
        portfolio_return - lam * portfolio_variance,
        GRB.MAXIMIZE
    )

    model.optimize()

    return model, w