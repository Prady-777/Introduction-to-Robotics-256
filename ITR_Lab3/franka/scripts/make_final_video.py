#!/usr/bin/env python3
"""
Join the three per-method recordings into ONE submission video with title cards.

  python scripts/make_final_video.py                       # uses results/videos/{mink,dls,qp}.mp4
  python scripts/make_final_video.py --max-seconds 60      # trim each part to 1 min

Record the parts first, e.g. (each episode is ~15 s of sim time, 4 episodes ≈ 1 min):
  python scripts/run_episodes.py --ik mink --n 4 --record results/videos/mink.mp4
"""
import argparse
import os
import sys

import numpy as np
import imageio
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TITLES = {"mink": "Stage 1 - Off-the-shelf IK (Mink)",
          "dls": "Stage 2 - Damped Least Squares IK",
          "qp": "Stage 2 - QP IK (joint + velocity limits)"}


def title_card(text, w, h):
    img = Image.new("RGB", (w, h), (15, 20, 30))
    d = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=36)
    except TypeError:
        font = None
    box = d.textbbox((0, 0), text, font=font)
    d.text(((w - box[2]) / 2, (h - box[3]) / 2), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO_ROOT, "results", "videos"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--title-seconds", type=float, default=2.0)
    args = ap.parse_args()
    out = args.out or os.path.join(args.dir, "final_submission.mp4")
    writer, fps = None, 30
    for m in ("mink", "dls", "qp"):
        path = os.path.join(args.dir, f"{m}.mp4")
        if not os.path.exists(path):
            print(f"missing {path} - skipped")
            continue
        reader = imageio.get_reader(path)
        fps = int(round(reader.get_meta_data().get("fps", 30)))
        for i, frame in enumerate(reader):
            if writer is None:
                h, w = frame.shape[:2]
                writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
            if i == 0:
                card = title_card(TITLES[m], frame.shape[1], frame.shape[0])
                for _ in range(int(args.title_seconds * fps)):
                    writer.append_data(card)
            if i >= args.max_seconds * fps:
                break
            writer.append_data(frame)
        reader.close()
        print(f"added {m}")
    if writer is None:
        sys.exit("no videos found")
    writer.close()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
