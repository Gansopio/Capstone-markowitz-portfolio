import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def graficos_grid_search(csv_path):

    df = pd.read_csv(csv_path)

    # ===============================
    # 1. Score vs Train/Test
    # ===============================
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x="train_years",
        y="test_years",
        size="score",
        hue="score",
        sizes=(50, 500),
        palette="viridis"
    )
    plt.title("Score según años de Train/Test")
    plt.xlabel("Años de entrenamiento")
    plt.ylabel("Años de testing")
    plt.show()

    # ===============================
    # 2. Retorno vs Abandono
    # ===============================
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x="abandono_promedio_pct",
        y="retorno_total_pct",
        hue="perfil",
        size="score",
        sizes=(50, 400)
    )
    plt.title("Trade-off Retorno vs Abandono")
    plt.xlabel("Abandono (%)")
    plt.ylabel("Retorno (%)")
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.show()

    # ===============================
    # 3. Score promedio por perfil
    # ===============================
    plt.figure(figsize=(10,6))
    df_perfil = df.groupby("perfil")["score"].mean().reset_index()

    sns.barplot(
        data=df_perfil,
        x="perfil",
        y="score"
    )
    plt.title("Score promedio por perfil")
    plt.xticks(rotation=45)
    plt.show()

    # ===============================
    # 4. Heatmap Train vs Test
    # ===============================
    pivot = df.pivot_table(
        values="score",
        index="train_years",
        columns="test_years"
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn"
    )
    plt.title("Heatmap Score (Train vs Test)")
    plt.show()

    # ===============================
    # 5. Ranking mejores modelos
    # ===============================
    top = df.sort_values("score", ascending=False).head(10)

    print("\n🏆 TOP 10 MODELOS")
    print(top[[
        "perfil",
        "train_years",
        "test_years",
        "retorno_total_pct",
        "abandono_promedio_pct",
        "score"
    ]])