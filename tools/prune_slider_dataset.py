"""根据 analyze_slider_dataset.py 的 stats.csv 把 dead-signal 图移到 dead/ 子目录。

不删除文件，只移动；如果改主意可以再 mv 回来。

dead-signal 判定（OR 关系）：
- Lab chroma < 5             pair op 后 pos ≈ neg
- chroma_std < 5             全屏色调单一
- L_star > 95 或 < 5         接近全白或全黑

用法：
  python tools/prune_slider_dataset.py tmp/slider_analysis/stats.csv tmp/slider_data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats_csv", help="analyze_slider_dataset.py 出的 stats.csv")
    ap.add_argument("data_dir", help="对应的图片目录")
    ap.add_argument("--dest", default=None, help="dead 子目录（默认 <data_dir>/_dead/）")
    ap.add_argument("--chroma-min", type=float, default=5.0)
    ap.add_argument("--chroma-std-min", type=float, default=5.0)
    ap.add_argument("--lstar-min", type=float, default=5.0)
    ap.add_argument("--lstar-max", type=float, default=95.0)
    ap.add_argument("--dry-run", action="store_true", help="只打印不移动")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    dest = Path(args.dest) if args.dest else data_dir / "_dead"
    df = pd.read_csv(args.stats_csv)
    flag = (
        (df["chroma"] < args.chroma_min)
        | (df["chroma_std"] < args.chroma_std_min)
        | (df["L_star"] < args.lstar_min)
        | (df["L_star"] > args.lstar_max)
    )
    dead = df[flag].copy()
    print(f"扫到 {len(df)} 张图，标记 dead-signal: {len(dead)} 张 ({len(dead)/len(df)*100:.1f}%)")
    print(f"目标目录: {dest}")
    print()

    if args.dry_run:
        print("(dry-run) 候选清单：")
        for _, row in dead.iterrows():
            print(f"  {row['path']:50s}  chroma={row['chroma']:.1f}  chroma_std={row['chroma_std']:.1f}  L*={row['L_star']:.1f}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    missing = 0
    for _, row in dead.iterrows():
        src = data_dir / row["path"]
        if not src.exists():
            missing += 1
            continue
        dst = dest / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1
    print(f"移动 {moved} 张；{missing} 张找不到（可能已被移动）")


if __name__ == "__main__":
    main()
