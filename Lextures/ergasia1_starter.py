#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ρομποτική Όραση - Εργασία 1: Βλέπει το ρομπότ τα αντικείμενα στο τραπέζι;

Αρχείο εκκίνησης (starter). Το παραδοτέο σας είναι ένα ΕΝΙΑΙΟ αρχείο
`ergasia1.py` που προκύπτει από αυτό. Τρέχει με μία εντολή:

    python ergasia1.py --images data --truth data/ground_truth.json --out results

Δοκιμή χωρίς δικές σας φωτογραφίες (συνθετικά δεδομένα):

    python ergasia1.py --demo

Μελέτη αφαίρεσης (απενεργοποίηση ενός βήματος της αλυσίδας σας):

    python ergasia1.py --images data --truth data/ground_truth.json --disable denoise

Επιτρεπτά εργαλεία: numpy, opencv-python, matplotlib και η standard library.

ΟΔΗΓΙΑ ΧΡΗΣΗΣ ΤΟΥ ΑΡΧΕΙΟΥ
-------------------------
  [ΜΗΝ ΑΛΛΑΞΕΤΕ]   ενότητες που σημειώνονται έτσι: το ground truth και η
                   βαθμολόγηση στηρίζονται σε αυτές.
  [ΣΥΜΠΛΗΡΩΣΤΕ]    η ενότητα CFG και η συνάρτηση detect_targets, καθώς και
                   όσες βοηθητικές συναρτήσεις θέλετε να προσθέσετε.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# [ΣΥΜΠΛΗΡΩΣΤΕ] Παράμετροι της λύσης σας
# -----------------------------------------------------------------------------
# Όλες οι ρυθμίσεις (όρια, μεγέθη πυρήνων, κ.λπ.) μπαίνουν εδώ και όχι
# διάσπαρτες στον κώδικα. Ίδιες τιμές για ΟΛΕΣ τις φωτογραφίες. Επιτρέπεται
# να υπολογίζει το πρόγραμμα τιμές από την ίδια την εικόνα (π.χ. από το
# ιστόγραμμα), αλλά όχι να αλλάζετε χειροκίνητα παραμέτρους ανά φωτογραφία.
#
# Το "stages" λέει ποια βήματα της αλυσίδας σας είναι ενεργά. Τα ονόματα
# είναι ενδεικτικά: βάλτε τα δικά σας. Η εκτέλεση με --disable ΟΝΟΜΑ
# απενεργοποιεί το αντίστοιχο βήμα (μελέτη αφαίρεσης).
# =============================================================================
CFG = {
    "stages": {
        "denoise": True,
        "equalize": True,
        "color_mask": True,
        "morphology": True,
    },
    # TODO: προσθέστε εδώ τις παραμέτρους σας, π.χ.:
    # "blur_ksize": 5,
    # "hsv_ranges": [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    # "morph_ksize": 5,
    # "min_area_px": 150,
}


# =============================================================================
# [ΜΗΝ ΑΛΛΑΞΕΤΕ] Σταθερές και βοηθητικές συναρτήσεις του starter
# =============================================================================
PREP_WIDTH = 640          # πλάτος των εικόνων μετά τη load_and_prepare
MATCH_TOL_CM = 2.0        # μέγιστη απόσταση για να "ταιριάξει" ανίχνευση με ground truth
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_and_prepare(path):
    """Φορτώνει μια φωτογραφία (BGR) και την αλλάζει σε πλάτος 640 pixels,
    κρατώντας την αναλογία. Το ground truth δίνεται σε αυτές τις συντεταγμένες."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Δεν διαβάστηκε η εικόνα: {path}")
    h, w = img.shape[:2]
    new_h = int(round(h * PREP_WIDTH / w))
    return cv2.resize(img, (PREP_WIDTH, new_h), interpolation=cv2.INTER_AREA)


def stage_on(cfg, name):
    """True αν το βήμα `name` είναι ενεργό (δεν έχει απενεργοποιηθεί με --disable)."""
    return bool(cfg.get("stages", {}).get(name, True))


def evaluate(detections, truth_centers, px_per_cm):
    """Αντιστοιχίζει ανιχνεύσεις με το ground truth και επιστρέφει λεξικό με:
        tp, fp, fn      σωστές, πλεονάζουσες, χαμένες ανιχνεύσεις
        err_px, err_cm  μέσο σφάλμα θέσης των σωστών ανιχνεύσεων (None αν tp = 0)
        ok              True όταν fp == 0 και fn == 0 (σωστό πλήθος και θέσεις)
        pairs           λίστα (index_ανίχνευσης, index_truth) των αντιστοιχίσεων
    """
    tol_px = MATCH_TOL_CM * px_per_cm
    det = [tuple(map(float, d)) for d in detections]
    gt = [tuple(map(float, g)) for g in truth_centers]

    candidates = []
    for i, d in enumerate(det):
        for j, g in enumerate(gt):
            dist = float(np.hypot(d[0] - g[0], d[1] - g[1]))
            if dist <= tol_px:
                candidates.append((dist, i, j))
    candidates.sort()

    used_d, used_g, pairs, dists = set(), set(), [], []
    for dist, i, j in candidates:
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
        dists.append(dist)

    tp = len(pairs)
    fp = len(det) - tp
    fn = len(gt) - tp
    err_px = float(np.mean(dists)) if dists else None
    err_cm = err_px / px_per_cm if err_px is not None else None
    return {"tp": tp, "fp": fp, "fn": fn, "err_px": err_px, "err_cm": err_cm,
            "ok": fp == 0 and fn == 0, "pairs": pairs}


def draw_results(img, detections, truth_centers, out_path):
    """Αποθηκεύει εικόνα με πράσινους σταυρούς (ανιχνεύσεις) και
    κόκκινους κύκλους (ground truth)."""
    vis = img.copy()
    for (x, y) in truth_centers:
        cv2.circle(vis, (int(round(x)), int(round(y))), 14, (0, 0, 255), 2)
    for (x, y) in detections:
        cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
    cv2.putText(vis, "green: detection | red: ground truth", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def scene_and_light(path):
    """'scene1_dim.jpg' -> ('scene1', 'dim'). Χωρίς κάτω παύλα: φωτισμός 'unknown'."""
    stem = Path(path).stem
    if "_" in stem:
        scene, light = stem.split("_", 1)
        return scene, light
    return stem, "unknown"


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"Ο φάκελος εικόνων δεν υπάρχει: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"Δεν βρέθηκαν εικόνες στον φάκελο: {folder}")
    return files


def load_truth(path):
    try:
        with open(path, encoding="utf-8") as f:
            truth = json.load(f)
        ref = truth["reference"]
        px_per_cm = float(ref["length_px"]) / float(ref["length_cm"])
    except (OSError, KeyError, ValueError, ZeroDivisionError) as e:
        sys.exit(f"Πρόβλημα στο ground truth ({path}): {e!r}")
    return truth, px_per_cm


def print_table(rows):
    header = f"{'εικόνα':<22}{'βρέθηκαν/πραγμ.':>16}{'TP':>4}{'FP':>4}{'FN':>4}" \
             f"{'σφάλμα px':>11}{'σφάλμα cm':>11}{'OK':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ep = "-" if r["err_px"] is None else f"{r['err_px']:.1f}"
        ec = "-" if r["err_cm"] is None else f"{r['err_cm']:.2f}"
        found_true = f"{r['found']}/{r['count']}"
        print(f"{r['image']:<22}{found_true:>16}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}"
              f"{ep:>11}{ec:>11}{('ναι' if r['ok'] else 'όχι'):>5}")


def print_summary_by_light(rows):
    lights = sorted({r["light"] for r in rows})
    print("\nΣύνοψη ανά φωτισμό (το δεδομένο για τον πίνακα της αναφοράς):")
    for light in lights:
        sub = [r for r in rows if r["light"] == light]
        n_ok = sum(r["ok"] for r in sub)
        errs = [r["err_cm"] for r in sub if r["err_cm"] is not None]
        mean_err = f"{np.mean(errs):.2f} cm" if errs else "-"
        print(f"  {light:<8} σωστό πλήθος και θέσεις: {n_ok}/{len(sub)}"
              f"   μέσο σφάλμα θέσης: {mean_err}")
    total_ok = sum(r["ok"] for r in rows)
    print(f"  ΣΥΝΟΛΟ   σωστές εικόνες: {total_ok}/{len(rows)}")


def save_csv(rows, path):
    cols = ["image", "scene", "light", "count", "found", "tp", "fp", "fn",
            "err_px", "err_cm", "ok"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})


# -----------------------------------------------------------------------------
# Συνθετικά δεδομένα για δοκιμή (--demo). ΜΗΝ τα παραδώσετε ως δικά σας.
# -----------------------------------------------------------------------------
def make_demo_data(folder):
    """Φτιάχνει 6 συνθετικές εικόνες (2 διατάξεις x 3 φωτισμοί) και ground truth.
    Κλίμακα: 10 px = 1 cm (ο 'χάρακας' των 10 cm έχει μήκος 100 px)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W, H = PREP_WIDTH, 480
    red, orange, blue, green = (30, 30, 200), (30, 120, 235), (200, 80, 30), (60, 170, 60)

    scenes = {
        "scene1": {
            "targets": [(120, 140), (300, 110), (470, 200), (220, 330)],
            "others": [((400, 360), blue), ((560, 330), green)],
        },
        "scene2": {
            "targets": [(160, 250), (210, 250), (520, 120)],   # δύο κοντινοί (κενό 1 cm)
            "others": [((350, 300), orange), ((540, 340), blue)],
        },
    }
    truth = {"reference": {"length_cm": 10, "length_px": 100}}
    for name, sc in scenes.items():
        base = np.full((H, W, 3), (95, 120, 150), np.uint8)       # "τραπέζι"
        base = cv2.add(base, rng.normal(0, 4, base.shape).clip(0, 255).astype(np.uint8))
        for (c, col) in sc["others"]:
            cv2.circle(base, c, 20, col, -1)
        for c in sc["targets"]:
            cv2.circle(base, c, 20, red, -1)
        cv2.rectangle(base, (20, 440), (120, 452), (20, 20, 20), -1)   # χάρακας 10 cm

        dim = (base.astype(np.float32) * 0.40).astype(np.uint8)
        shadow = base.astype(np.float32)
        shadow[:, W // 2:W // 2 + 170] *= 0.45                        # λωρίδα σκιάς
        variants = {"normal": base, "dim": dim, "shadow": shadow.clip(0, 255).astype(np.uint8)}
        for light, im in variants.items():
            noisy = im.astype(np.float32) + rng.normal(0, 6, im.shape)
            cv2.imwrite(str(folder / f"{name}_{light}.jpg"), noisy.clip(0, 255).astype(np.uint8))
        truth[name] = {"count": len(sc["targets"]),
                       "centers_px": [list(t) for t in sc["targets"]]}
    with open(folder / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    return folder


# =============================================================================
# [ΣΥΜΠΛΗΡΩΣΤΕ] Η λύση σας
# =============================================================================
def detect_targets(img_bgr, cfg):
    """Βρίσκει τους στόχους (π.χ. τα κόκκινα αντικείμενα) στην εικόνα.

    Είσοδος:  img_bgr  εικόνα BGR (uint8), πλάτος 640 pixels.
              cfg      το λεξικό CFG (όρια, μεγέθη πυρήνων, ενεργά βήματα).
    Έξοδος:   λίστα κέντρων [(x, y), ...] σε pixels, στις συντεταγμένες της
              εικόνας εισόδου. Άδεια λίστα αν δεν βρεθεί τίποτα.

    Ο στόχος και οι περιορισμοί είναι στην εκφώνηση. Φτιάξτε πρώτα το baseline
    (γκρι εικόνα και κατωφλίωση Otsu), καταγράψτε το αποτέλεσμά του και
    προσθέστε βήματα μόνο όταν μπορείτε να δείξετε ότι βελτιώνουν την επίδοση.
    Περικλείστε κάθε βήμα που θέλετε να μπορεί να απενεργοποιηθεί με:

        if stage_on(cfg, "όνομα_βήματος"):
            ...
    """
    # TODO: υλοποιήστε την αλυσίδα όρασης.
    centers = []
    return centers


# =============================================================================
# [ΜΗΝ ΑΛΛΑΞΕΤΕ τη δομή] main και ορίσματα γραμμής εντολών
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Εργασία 1: ανίχνευση στόχων στο τραπέζι")
    p.add_argument("--images", default="data", help="φάκελος με τις φωτογραφίες")
    p.add_argument("--truth", default="data/ground_truth.json", help="αρχείο ground truth")
    p.add_argument("--out", default="results", help="φάκελος εξόδου")
    p.add_argument("--disable", nargs="*", default=[], metavar="ΒΗΜΑ",
                   help="απενεργοποιεί βήματα της αλυσίδας (μελέτη αφαίρεσης)")
    p.add_argument("--demo", action="store_true",
                   help="δημιουργεί και χρησιμοποιεί συνθετικά δεδομένα (φάκελος demo_data)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.demo:
        demo_dir = make_demo_data("demo_data")
        args.images = str(demo_dir)
        args.truth = str(demo_dir / "ground_truth.json")
        print(f"[demo] Συνθετικά δεδομένα στον φάκελο: {demo_dir}")

    cfg = json.loads(json.dumps(CFG))            # αντίγραφο, ώστε να μην αλλοιώνεται το CFG
    for name in args.disable:
        if name not in cfg.get("stages", {}):
            print(f"[προειδοποίηση] Το βήμα '{name}' δεν υπάρχει στο CFG['stages'] "
                  f"(γνωστά: {', '.join(cfg.get('stages', {})) or '-'})")
        cfg.setdefault("stages", {})[name] = False
    if args.disable:
        print(f"[ablation] Απενεργοποιημένα βήματα: {', '.join(args.disable)}")

    truth, px_per_cm = load_truth(args.truth)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in list_images(args.images):
        scene, light = scene_and_light(path)
        if scene not in truth:
            print(f"[παράλειψη] {path.name}: δεν υπάρχει '{scene}' στο ground truth")
            continue
        img = load_and_prepare(path)
        detections = detect_targets(img, cfg)

        truth_centers = truth[scene]["centers_px"]
        res = evaluate(detections, truth_centers, px_per_cm)
        draw_results(img, detections, truth_centers, out_dir / f"{path.stem}_result.png")
        rows.append({"image": path.name, "scene": scene, "light": light,
                     "count": truth[scene]["count"], "found": len(detections),
                     "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                     "err_px": res["err_px"], "err_cm": res["err_cm"], "ok": res["ok"]})

    if not rows:
        sys.exit("Δεν αξιολογήθηκε καμία εικόνα. Ελέγξτε τα ονόματα αρχείων και το ground truth.")

    print()
    print_table(rows)
    print_summary_by_light(rows)
    save_csv(rows, out_dir / "results.csv")
    print(f"\nΑποτελέσματα στον φάκελο: {out_dir}/")


if __name__ == "__main__":
    main()
