import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text




def plot_rankingboard(df, Task='Generation', dataset='MetaModulus', metrics='Mean Validity',
                      save_path='D:\\Workspace\\PhD_workspace\\metabench-proj\\Metamaterial-Benchmark\\WebInterface\\data\\images\\'):
    df_plot = df.dropna(subset=[metrics])

    # -------------------------------
    if 'MTT' in metrics or "MET" in metrics:
        df_best = df_plot.loc[df_plot.groupby("Year")[metrics].idxmin()].sort_values(by="Year")
    else:
        df_best = df_plot.loc[df_plot.groupby("Year")[metrics].idxmax()].sort_values(by="Year")

    # -------------------------------
    # -------------------------------
    plt.figure(figsize=(10, 2))
    sns.set(style="whitegrid", context='paper')

    color_best = "darkorange"
    color_other = "lightgray"
    color_line = "navy"

    sns.scatterplot(x=df_plot['Year'], y=df_plot[metrics], color=color_other, s=50, label="Other models", zorder=1)

    sns.lineplot(x=df_best['Year'], y=df_best[metrics], linewidth=2, marker='o', markersize=8, label="Best R² per Year",
                 zorder=2)
    texts = []
    for i, row in df_plot.iterrows():
        text_obj = plt.text(row['Year'], row[metrics], row['Method'], fontsize=10, fontweight='bold', ha='left',
                 va='bottom', color="black",
                horizontalalignment='left',zorder=3)

        texts.append(text_obj)
    adjust_text(
        texts,
        expand_text=(1.05, 1.2),
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        only_move={'points': 'y', 'text': 'xy'},  # 控制移动方向
        force_points=0.5,
        force_text=5.0,
        lim=200  # 迭代次数上限
    )

    # -------------------------------
    # -------------------------------
    plt.xlabel("Publication Year", fontsize=12)
    plt.ylabel(f"{metrics}", fontsize=12)
    plt.title(f"Ranking Board: Best {metrics} vs Publication Year", fontsize=14, fontweight='bold')
    plt.xticks(range(df_plot['Year'].min(), df_plot['Year'].max() + 1), fontsize=10)
    plt.yticks(fontsize=10)
    # std = df[metrics].std()
    # plt.ylim([df[metrics].mean() - std - 0.05, df[metrics].max() + 0.05])
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.show()
    plt.savefig(
        f'{save_path}/{Task}_{dataset}_{metrics}.png',bbox_inches='tight',
        dpi=2048)

