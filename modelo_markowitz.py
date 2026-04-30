import gurobipy as gp
from gurobipy import GRB 


def model_markowitz(train_returns, R_objetivo_anual, perfil):
    mu = train_returns.mean().values
    Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)

    n = len(tickers)
    R_objetivo_diario = R_objetivo_anual / 252

    model = gp.Model(f"Markowitz_{perfil}")
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

    model.addConstr(
        portfolio_return >= -R_objetivo_diario,
        name="retorno_minimo_perfil"
    )

    portfolio_variance = gp.quicksum(
        w[i] * Sigma[i, j] * w[j]
        for i in range(n)
        for j in range(n)
    )

    model.setObjective(portfolio_variance, GRB.MINIMIZE)

    model.optimize()
    return model, w
    