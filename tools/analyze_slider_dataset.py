"""Concept slider 数据集统计分析。

对一个图片目录批量计算 9 个分布指标，输出：
- stats.csv               每张图一行的指标表
- health_hist.png         9 个指标的直方图（一眼看分布偏差）
- correlation.png         相关性热图 + 关键 scatter（saturation ↔ brightness 等）
- outliers.txt            可疑离群图（接近灰度 / 接近单色等无训练信号的）

用法：
  python tools/analyze_slider_dataset.py tmp/slider_data --out tmp/slider_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, filters


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_metrics(img: Image.Image) -> dict:
    """对单张 PIL Image 算 9 个指标。

    全部对 512×512 中心 crop 算（POC 跟训练时 resolution 解耦；只看分布）。
    """
    # 中心 crop 到正方形，缩到 512（统计层够用）
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS)

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H,W,3] in [0,1]

    # ─── 颜色空间转换 ───
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0  # H,S,V in [0,1]
    lab = color.rgb2lab(arr)  # L* in [0,100], a*/b* in ~[-128,127]

    # ─── 9 个指标 ───
    sat_hsv = float(hsv[..., 1].mean())                                 # HSV 饱和度
    val_hsv = float(hsv[..., 2].mean())                                 # HSV 亮度（max）
    L_star = float(lab[..., 0].mean())                                  # CIE 感知亮度
    L_star_std = float(lab[..., 0].std())                               # 亮度对比度
    chroma = float(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).mean()) # Lab 色彩饱和（独立于亮度）
    chroma_std = float(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).std())  # 色彩对比度
    r_mean = float(arr[..., 0].mean())
    g_mean = float(arr[..., 1].mean())
    b_mean = float(arr[..., 2].mean())

    # 边缘密度（图像复杂度代理）：Sobel on L*
    L_norm = lab[..., 0] / 100.0
    edges = filters.sobel(L_norm)
    edge_density = float(edges.mean())

    return {
        "sat_hsv": sat_hsv,
        "val_hsv": val_hsv,
        "L_star": L_star,
        "L_star_std": L_star_std,
        "chroma": chroma,
        "chroma_std": chroma_std,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "edge_density": edge_density,
    }


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    rows = []
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        print(f"ERROR: 没扫到图片 {data_dir}", file=sys.stderr)
        sys.exit(1)
    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            metrics = compute_metrics(img)
            metrics["path"] = str(p.relative_to(data_dir))
            rows.append(metrics)
        except Exception as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"  scanned {i+1}/{len(paths)}")
    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("sat_hsv", "HSV Saturation", 0, 1),
        ("chroma", "Lab Chroma (色彩饱和)", 0, 80),
        ("val_hsv", "HSV Value (亮度 max)", 0, 1),
        ("L_star", "CIE L* (感知亮度)", 0, 100),
        ("L_star_std", "L* spatial std (亮度对比)", 0, 40),
        ("chroma_std", "Chroma std (色彩对比)", 0, 40),
        ("r_mean", "R channel mean", 0, 1),
        ("g_mean", "G channel mean", 0, 1),
        ("b_mean", "B channel mean", 0, 1),
        ("edge_density", "Edge density (复杂度)", 0, 0.2),
    ]
    n = len(metrics)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    axes = axes.flatten()
    for i, (col, label, lo, hi) in enumerate(metrics):
        ax = axes[i]
        ax.hist(df[col], bins=25, range=(lo, hi), color="steelblue", edgecolor="black")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(col, fontsize=8)
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1, label=f"μ={df[col].mean():.3f}")
        ax.axvline(df[col].median(), color="orange", linestyle=":", linewidth=1, label=f"med={df[col].median():.3f}")
        ax.legend(fontsize=7)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"Slider Dataset Health: {len(df)} images", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out_path: Path) -> None:
    """相关性热图 + 关键 scatter（saturation ↔ brightness ↔ contrast 等）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["sat_hsv", "chroma", "val_hsv", "L_star",
            "L_star_std", "chroma_std", "edge_density",
            "r_mean", "g_mean", "b_mean"]
    corr = df[cols].corr()

    fig = plt.figure(figsize=(16, 8))
    # 左：热图
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(cols)))
    ax1.set_yticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=45, ha="right")
    ax1.set_yticklabels(cols)
    ax1.set_title("Correlation matrix (|ρ|>0.4 警报阈)")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax1.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if abs(corr.iloc[i, j]) > 0.6 else "black",
                     fontsize=7)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 右：4 个关键 scatter
    pairs = [
        ("chroma", "L_star", "饱和 ↔ 亮度 (最警觉 confounder)"),
        ("chroma", "L_star_std", "饱和 ↔ 亮度对比"),
        ("chroma", "chroma_std", "饱和 ↔ 色彩对比"),
        ("chroma", "edge_density", "饱和 ↔ 图像复杂度"),
    ]
    for i, (x, y, title) in enumerate(pairs):
        ax = fig.add_subplot(2, 4, 3 + 2 * (i // 2) + (i % 2))
        ax.scatter(df[x], df[y], s=8, alpha=0.5, color="steelblue")
        rho = df[x].corr(df[y])
        ax.set_title(f"{title}\nρ={rho:.3f}", fontsize=9)
        ax.set_xlabel(x, fontsize=8)
        ax.set_ylabel(y, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def detect_outliers(df: pd.DataFrame) -> dict:
    """识别对训练有害的图：饱和度过低（≈ 灰度，pair 信号≈0）/ 单色（chroma_std≈0 没空间多样性）"""
    out = {
        "too_grayscale": df[df["chroma"] < 5].sort_values("chroma")[["path", "chroma", "sat_hsv"]].to_dict("records"),
        "near_monotone": df[df["chroma_std"] < 5].sort_values("chroma_std")[["path", "chroma_std", "edge_density"]].to_dict("records"),
        "extreme_dark": df[df["L_star"] < 15].sort_values("L_star")[["path", "L_star"]].to_dict("records"),
        "extreme_light": df[df["L_star"] > 90].sort_values("L_star", ascending=False)[["path", "L_star"]].to_dict("records"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="图片目录（递归扫）")
    ap.add_argument("--out", default="tmp/slider_analysis", help="输出目录")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir}...")
    df = scan_dataset(data_dir)
    print(f"  done: {len(df)} images")

    csv_path = out_dir / "stats.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV → {csv_path}")

    hist_path = out_dir / "health_hist.png"
    plot_histograms(df, hist_path)
    print(f"  Histograms → {hist_path}")

    corr_path = out_dir / "correlation.png"
    plot_correlation(df, corr_path)
    print(f"  Correlation → {corr_path}")

    outliers = detect_outliers(df)
    outlier_path = out_dir / "outliers.txt"
    with outlier_path.open("w", encoding="utf-8") as f:
        for cat, items in outliers.items():
            f.write(f"=== {cat} ({len(items)}) ===\n")
            for it in items[:10]:
                f.write(f"  {it}\n")
            f.write("\n")
    print(f"  Outliers → {outlier_path}")

    # 印一段 summary 到 stdout 方便直接读
    print("\n=== SUMMARY ===")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3))
    print("\n=== CORRELATIONS (|ρ|>0.4) ===")
    cols = ["sat_hsv", "chroma", "val_hsv", "L_star", "L_star_std",
            "chroma_std", "edge_density"]
    corr = df[cols].corr()
    flagged = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.4:
                flagged.append((cols[i], cols[j], r))
    if flagged:
        for a, b, r in sorted(flagged, key=lambda x: -abs(x[2])):
            print(f"  {a:20s} ↔ {b:20s}: ρ={r:+.3f}")
    else:
        print("  none (good)")

    print(f"\nOutliers: too_grayscale={len(outliers['too_grayscale'])}, "
          f"near_monotone={len(outliers['near_monotone'])}, "
          f"extreme_dark={len(outliers['extreme_dark'])}, "
          f"extreme_light={len(outliers['extreme_light'])}")


if __name__ == "__main__":
    main()
