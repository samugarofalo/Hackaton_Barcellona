"""Ball-only YOLO dataset from other games' hand-labelled frames (allowed: RULES.md, other games).

Sources, both open:
  SoccerNet-v3 H250 (Zenodo 10.5281/zenodo.7808511, CC BY 4.0): long-shot broadcast frames,
      YOLO labels, class 0 = ball (class 1 persons dropped).
  SoccerNet-Tracking 2023 (Hugging Face SoccerNet/SN-Tracking-2023): 30 s broadcast clips with a
      ball box on every frame; every STEP-th frame is used (consecutive frames are near copies).

Images are symlinked, labels rewritten with the ball only. Frames without a ball are kept as
negatives. Split: H250 train + SN-Tracking train -> train; H250 valid + SN-Tracking test -> val.

    .venv/bin/python pipeline/ball_dataset.py
Writes data/external/ball_yolo/{train,val}/{images,labels} and data/external/ball_yolo/data.yaml.
"""

import configparser
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(EXT, "ball_yolo")
STEP = {"train": 5, "val": 10}


def put(split, name, image, lines):
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT, split, sub), exist_ok=True)
    link = os.path.join(OUT, split, "images", name + ".jpg")
    if not os.path.lexists(link):
        os.symlink(image, link)
    with open(os.path.join(OUT, split, "labels", name + ".txt"), "w") as f:
        f.writelines(lines)


def h250(split, src):
    n = 0
    for lab in glob.glob(os.path.join(EXT, "soccernet-v3-h250", "YOLO", src, "labels", "*.txt")):
        stem = os.path.basename(lab)[:-4]
        image = os.path.join(EXT, "soccernet-v3-h250", "YOLO", src, "images", stem + ".jpg")
        if os.path.exists(image):
            put(split, "h250_" + stem, image, [l for l in open(lab) if l.split()[:1] == ["0"]])
            n += 1
    return n


def sn_tracking(split, src):
    n = 0
    for clip in sorted(glob.glob(os.path.join(EXT, "soccernet-tracking", src, "SNMOT-*"))):
        info = configparser.ConfigParser()
        info.read(os.path.join(clip, "gameinfo.ini"))
        ball = {int(k.split("_")[1]) for k, v in info["Sequence"].items()
                if k.startswith("trackletid_") and v.strip().startswith("ball")}
        seq = configparser.ConfigParser()
        seq.read(os.path.join(clip, "seqinfo.ini"))
        W, H = int(seq["Sequence"]["imWidth"]), int(seq["Sequence"]["imHeight"])
        boxes = {}
        for row in open(os.path.join(clip, "gt", "gt.txt")):
            f, tid, x, y, w, h = row.split(",")[:6]
            if int(tid) in ball:
                x, y, w, h = map(float, (x, y, w, h))
                boxes.setdefault(int(f), []).append("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        frames = sorted(glob.glob(os.path.join(clip, "img1", "*.jpg")))
        for image in frames[::STEP[split]]:
            f = int(os.path.basename(image)[:-4])
            put(split, "{}_{:06d}".format(os.path.basename(clip), f), image, boxes.get(f, []))
            n += 1
    return n


def main():
    counts = {"train": h250("train", "train") + sn_tracking("train", "train"),
              "val": h250("val", "valid") + sn_tracking("val", "test")}
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: {}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n".format(OUT))
    for split in ("train", "val"):
        labs = glob.glob(os.path.join(OUT, split, "labels", "*.txt"))
        with_ball = sum(os.path.getsize(p) > 0 for p in labs)
        print("{}: {} images ({} written now), {} with a ball".format(split, len(labs), counts[split], with_ball))


if __name__ == "__main__":
    main()
