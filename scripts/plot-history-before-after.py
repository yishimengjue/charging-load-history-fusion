"""Plot the same forecast day before and after recent-history reorganization.

Run from any directory with matplotlib, numpy, pymupdf and fonttools installed.
CHINESE_FONT may point to a Simplified Chinese TTF on non-macOS systems.
"""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/charging-figure-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fontTools.ttLib import TTFont
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper"
NAME = "Fig05_history_before_after"
ORIGIN = "2022-03-23T00:00:00.000000000"


def read_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    with tempfile.TemporaryDirectory(prefix="charging-sc-font-") as font_dir:
        path = os.environ.get("CHINESE_FONT")
        if not path:
            path = str(Path(font_dir) / "HeitiSC.ttf")
            font = TTFont("/System/Library/Fonts/STHeiti Light.ttc", fontNumber=1)
            assert font["name"].getDebugName(1) == "Heiti SC"
            font.save(path)
            font.close()
        fm.fontManager.addfont(path)
        plt.rcParams.update({
            "font.family": fm.FontProperties(fname=path).get_name(),
            "font.weight": 300, "font.size": 7, "axes.labelsize": 7,
            "axes.titlesize": 8, "axes.titleweight": 300,
            "axes.unicode_minus": False, "axes.spines.top": False,
            "axes.spines.right": False, "axes.linewidth": .6,
            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
            "legend.frameon": False, "pdf.fonttype": 42,
            "svg.fonttype": "none", "svg.hashsalt": NAME,
        })
        render()


def render():
    history_rows = read_csv("Fig01_continuous_history.csv")
    forecast = [row for row in read_csv("Fig03_source_data.csv") if row["origin_utc"] == ORIGIN]
    refs = read_csv("Fig01_selected_records.csv")
    assert len(history_rows) == 168 and len(forecast) == 24 and len(refs) == 7
    assert all(row["entity"] == "7" for row in history_rows + forecast)
    assert all(int(row["origin_ns"]) == int(np.datetime64(ORIGIN, "ns").astype(np.int64)) for row in history_rows)
    x = np.array([int(row["hours_relative_to_origin"]) for row in history_rows])
    h = np.array([int(row["horizon_hours"]) for row in forecast])
    assert np.array_equal(x, np.arange(-167, 1)) and np.array_equal(h, np.arange(1, 25))
    past = np.array([float(row["power_kw"]) if row["power_kw"] else np.nan for row in history_rows])
    y = np.array([float(row["observed_kw"]) for row in forecast])
    predictions = [np.array([float(row[field]) for row in forecast])
                   for field in ("recent_seed42_kw", "near_seed42_kw")]
    ref_x = np.array([18 - 24 * int(row["reference_age_days"]) for row in refs])
    ref_y = np.array([float(row["power_kw"]) for row in refs])
    assert np.allclose(past[ref_x + 167], ref_y)
    assert np.isfinite(y).all() and all(np.isfinite(p).all() for p in predictions)
    errors = [np.abs(p - y) for p in predictions]
    for i, row in enumerate(forecast):
        assert abs((errors[0][i] - errors[1][i]) - float(row["absolute_error_reduction_kw"])) < 1e-12

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.55), sharey=True)
    fig.subplots_adjust(left=.08, right=.985, top=.83, bottom=.13, hspace=.52, wspace=.19)
    colors, styles = ["#737373", "#0072B2"], [(0, (4, 2)), "-"]
    titles = ["使用前：仅近期168小时序列", "使用后：近期序列＋7个同小时值"]
    ymax = np.nanmax(np.concatenate([past, y, *predictions])) * 1.16
    stats = []
    for col, (prediction, color, style) in enumerate(zip(predictions, colors, styles)):
        ax, zoom = axes[:, col]
        ax.plot(x, past, color="#444444", linewidth=.65)
        ax.axvspan(0, 24, color="#EAF1FC", zorder=0)
        ax.axvline(0, color="#6c7280", linestyle="--", linewidth=.7)
        ax.plot(np.r_[0, h], np.r_[past[-1], y], color="#202020", linewidth=.85)
        ax.plot(h, prediction, color=color, linestyle=style, linewidth=1)
        if col == 1:
            ax.scatter(ref_x, ref_y, s=18, facecolors="white", edgecolors=color, linewidths=.85, zorder=4)
        ax.set_xlim(-168, 24)
        ax.set_ylim(0, ymax)
        ax.set_xticks([-168, -120, -72, -24, 24], ["3月16日", "3月18日", "3月20日", "3月22日", "3月24日"])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.tick_params(axis="x", labelsize=5.8)
        ax.set_title(titles[col], loc="left", pad=22)
        ax.text(.02, 1.025, "过去7天", transform=ax.transAxes, fontsize=6.5)
        ax.text(.97, 1.025, "预测日", transform=ax.transAxes, ha="right", fontsize=6.5, color="#355a90")
        ax.set_xlabel("日期（协调世界时）")
        ax.text(-.12, 1.13, "ab"[col], transform=ax.transAxes, fontsize=9)

        zoom.plot(np.r_[0, h], np.r_[past[-1], y], color="#202020", marker="o", markersize=2, linewidth=1)
        zoom.plot(h, prediction, color=color, linestyle=style, linewidth=1.1)
        zoom.axvspan(17.6, 18.4, color="#EAF1FC", zorder=0)
        zoom.vlines(18, y[17], prediction[17], color="#C04A35", linewidth=2, zorder=5)
        zoom.scatter([18, 18], [y[17], prediction[17]], color=["#202020", color], s=15, zorder=6)
        zoom.annotate(f"18点真实值：{y[17]:.2f} kW\n预测值：{prediction[17]:.2f} kW\n绝对误差：{errors[col][17]:.2f} kW",
                      xy=(18, (y[17] + prediction[17]) / 2), xytext=(.03, .89),
                      textcoords="axes fraction", va="top", fontsize=6.7, linespacing=1.55,
                      arrowprops={"arrowstyle": "-", "color": "#C04A35", "lw": .75},
                      bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": "#dadde3", "lw": .5})
        zoom.set_title("3月23日预测日放大", loc="left", pad=10)
        zoom.set_xlim(0, 24)
        zoom.set_xticks([0, 6, 12, 18, 24], ["0时", "6时", "12时", "18时", "次日0时"])
        zoom.get_xticklabels()[0].set_ha("left")
        zoom.get_xticklabels()[-1].set_ha("right")
        zoom.set_xlabel("预测日时刻（协调世界时）")
        zoom.text(-.12, 1.075, "cd"[col], transform=zoom.transAxes, fontsize=9)
        stats.append({"model": ["recent", "near"][col], "observed_18_kw": float(y[17]),
                      "predicted_18_kw": float(prediction[17]), "absolute_error_18_kw": float(errors[col][17]),
                      "daily_mae_kw": float(errors[col].mean())})
    axes[0, 0].set_ylabel("功率（kW）")
    axes[1, 0].set_ylabel("功率（kW）")
    fig.text(.08, .98, "同一段历史，同一天预测｜OP编号7｜2022年3月23日零点起报｜随机种子42", va="top", fontsize=7)
    handles = [Line2D([], [], color="#202020", label="真实负荷"),
               Line2D([], [], color=colors[0], linestyle=styles[0], label="使用前预测"),
               Line2D([], [], color=colors[1], label="使用后预测"),
               Line2D([], [], color=colors[1], marker="o", markerfacecolor="white", linestyle="none", label="取出的同小时值")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .958), ncol=4, fontsize=6.3, columnspacing=1.15)
    fig.text(.08, .015, f"18点绝对误差：{errors[0][17]:.2f} → {errors[1][17]:.2f} kW，减少{errors[0][17] - errors[1][17]:.2f} kW。红色线段表示该时刻误差。", fontsize=7)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{NAME}.{extension}", dpi=600)
    plt.close(fig)
    svg = OUT / f"{NAME}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    pdf = pymupdf.open(OUT / f"{NAME}.pdf")
    text = pdf[0].get_text()
    assert all(value in text for value in ["1.38", "11.30", "5.98", "9.92", "4.60", "5.32"])
    for block in pdf[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                assert min(x0, y0) >= -.5 and x1 <= pdf[0].rect.width + .5 and y1 <= pdf[0].rect.height + .5, span["text"]
    pdf[0].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save("/private/tmp/history-before-after-check.png")
    pdf.close()
    merged = pymupdf.open()
    for path in sorted(OUT.glob("Fig*.pdf")):
        with pymupdf.open(path) as document:
            merged.insert_pdf(document)
    merged.save(OUT / "paper_figures.pdf")
    merged.close()
    report = {"passed": True, "origin_utc": ORIGIN, "entity": "7", "seed": 42,
              "history_hours": len(x), "forecast_hours": len(h), "stats": stats,
              "sources_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                  for name in ["Fig01_continuous_history.csv", "Fig01_selected_records.csv", "Fig03_source_data.csv"]}}
    (OUT / f"{NAME}_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
