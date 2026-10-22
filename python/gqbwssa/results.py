
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .algorithms import DISPLAY_NAMES

_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def summarize(results: dict[str, dict], feature_names: list, dataset_name: str) -> pd.DataFrame:
    rows = []
    for k, (name, r) in enumerate(results.items(), start=1):
        ind, conve, pos = r["ind"], r["conve"], r["pos"]
        final = conve[:, -1]
        best_run = int(np.argmin(final))
        best_pos = pos[best_run].astype(int)
        rows.append({
            "序号": k,
            "算法": DISPLAY_NAMES.get(name, name),
            "平均准确度": ind[0].mean(),
            "平均特征选择率": ind[1].mean(),
            "平均运行时间(s)": ind[2].mean(),
            "最佳适应度": final.min(),
            "最差适应度值": final.max(),
            "标准差": final.std(ddof=1) if len(final) > 1 else 0.0,
            "平均适应度": final.mean(),
            "最佳子集特征数": int(best_pos.sum()),
            "最佳特征子集(0/1)": " ".join(map(str, best_pos)),
            "最佳特征子集(名称)": ", ".join(f for f, b in zip(feature_names, best_pos) if b),
            "数据集": dataset_name,
        })
    return pd.DataFrame(rows)


def save_summary(df: pd.DataFrame, results: dict[str, dict], out_dir: str, tag: str):
    df.to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False, encoding="utf-8-sig")
    xlsx = os.path.join(out_dir, f"{tag}_结果.xlsx")
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, sheet_name="汇总", index=False)
        for name, r in results.items():
            m = r["ind"].shape[1]
            runs = pd.DataFrame({
                "运行次数": np.arange(1, m + 1),
                "准确度": r["ind"][0],
                "特征选择率": r["ind"][1],
                "运行时间(s)": r["ind"][2],
                "最终适应度": r["conve"][:, -1],
            })
            runs.to_excel(w, sheet_name=f"{name}_每次运行", index=False)
            pd.DataFrame(r["conve"].T, columns=[f"run{i + 1}" for i in range(m)]).rename_axis(
                "迭代").to_excel(w, sheet_name=f"{name}_迭代曲线")
    return xlsx


def _setup_font():
    import matplotlib
    from matplotlib import font_manager
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            matplotlib.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _gqbwssa_best_run_index(results: dict[str, dict]) -> int:
    
    conve = results["GQBWSSA"]["conve"]
    return int(np.argmin(conve[:, -1]))


def plot_convergence(results: dict[str, dict], out_dir: str, tag: str, dataset_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import MaxNLocator

    zh = _setup_font()
    styles = {"GWO": ("#FBEAA6", "^"), "WOA": ("#D2D2D2", "d"), "DBO": ("#9ACE87", "s"),
              "HHO": ("#C76B60", "o"), "GQBWSSA": ("#999BB9", "o")}
    n_runs = next(iter(results.values()))["conve"].shape[0]
    best_run = _gqbwssa_best_run_index(results) if "GQBWSSA" in results else 0
    paths = []
    
    for mode in ("best", "mean"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, r in results.items():
            color, marker = styles.get(name, ("#3E90BF", "v"))
            conve = r["conve"]
            curve = conve[best_run] if mode == "best" else conve.mean(axis=0)
            ax.plot(np.arange(1, len(curve) + 1), curve, "-" + marker,
                    color=color, lw=2, markerfacecolor="#C76B60", ms=3,
                    label=DISPLAY_NAMES.get(name, name))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        ax.legend(fontsize=12, prop={"family": "Times New Roman"} if not zh else None)
        if zh:
            ax.set_xlabel("进化代数", fontsize=12)
            ax.set_ylabel("适应度", fontsize=12)
            sub = f"第{best_run + 1}次" if mode == "best" else f"{n_runs}次运行平均"
            ax.set_title(f"算法的迭代进化曲线({dataset_name}，{sub})", fontsize=14)
        else:
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Fitness", fontsize=12)
            sub = f"run {best_run + 1}" if mode == "best" else "mean"
            ax.set_title(f"Convergence curves ({dataset_name}, {sub})", fontsize=14)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{tag}_convergence_{mode}.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths
