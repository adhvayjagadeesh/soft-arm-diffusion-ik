#!/usr/bin/env python3
"""Camera calibration and ArUco pose measurement for the continuum arm.

The measurement half of `real_arm.py` (servo_bus.py is the actuation half).
Nothing here needs the arm to exist, so all of it can be finished and
characterised before a single disc is bonded.

Run in this order:

    python vision.py --preview                  # find the camera, frame the arm
    python vision.py --capture-calib            # 15-20 checkerboard shots
    python vision.py --calibrate                # -> camera_calib.npz
    python vision.py --size-test                # is 1.25 in enough at your distance?
    python vision.py --noise-floor --sep 100.0  # the number the paper needs

    pip install opencv-contrib-python
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("pip install opencv-contrib-python")

# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0
CALIB_FILE = "camera_calib.npz"
CALIB_DIR = "calib_shots"

CHESSBOARD = (9, 6)          # inner corners, matches the printed board
SQUARE_M = 0.025             # 25.0 mm squares

# Marker side, in metres. THIS IS A MEASURED QUANTITY, NOT A PREFERENCE.
# solvePnP recovers range from apparent size, so range scales linearly with
# whatever is put here: a marker declared 5% large reads 5% far, which on a
# 305 mm arm is 15 mm of pure bias that averaging will never remove and that
# looks exactly like a sim-to-real gap.
#
# 30 mm is what the printed tabs take - a 40 mm face less a 5 mm quiet zone
# each side. This was 1.25 in (31.75 mm) from the era of hand-cut markers on
# hand-glued tabs, a 5.8% error left behind when the tabs became printed.
# Measure the marker you actually printed, with calipers, black edge to black
# edge, and put that number here.
MARKER_M = 0.030

# --------------------------------------------------------------------------- #
# OpenCV moved the ArUco API around in 4.7 and again in 5.x. Bind once here so
# the rest of the file is version-agnostic.
def _aruco_detector(dict_id=None):
    dict_id = cv2.aruco.DICT_4X4_50 if dict_id is None else dict_id
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(dict_id)
    else:                                             # pragma: no cover
        d = cv2.aruco.Dictionary_get(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)
    return lambda gray: cv2.aruco.detectMarkers(gray, d)   # pragma: no cover


DETECT = _aruco_detector()


def _marker_object_points(side_m: float) -> np.ndarray:
    """Corners of a marker in its own frame, in ArUco's corner order."""
    h = side_m / 2.0
    return np.array([[-h,  h, 0.0], [h,  h, 0.0],
                     [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32)


class Camera:
    def __init__(self, index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index}. Try other indices (0,1,2), and on "
                "macOS make sure the terminal app has Camera permission in "
                "System Settings > Privacy & Security > Camera.")

    def frame(self, gray=True):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# --------------------------------------------------------------------------- #
def capture_calib_shots(n=20, outdir=CALIB_DIR):
    """Grab checkerboard views. Press SPACE when the board is detected."""
    os.makedirs(outdir, exist_ok=True)
    saved = 0
    print("SPACE saves a shot (only when CORNERS FOUND), q quits.\n"
          "Vary angle, distance, and position in frame. Include strong tilts,\n"
          "and cover the frame corners - that is where distortion lives.")
    with Camera() as cam:
        while saved < n:
            gray = cam.frame()
            ok, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ok:
                cv2.drawChessboardCorners(vis, CHESSBOARD, corners, ok)
            cv2.putText(vis, f"{saved}/{n}  {'CORNERS FOUND' if ok else 'no board'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if ok else (0, 0, 220), 2)
            cv2.imshow("calibration capture", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" ") and ok:
                path = os.path.join(outdir, f"calib_{saved:02d}.png")
                cv2.imwrite(path, gray)
                saved += 1
                print(f"  saved {path}")
    cv2.destroyAllWindows()
    print(f"{saved} shots in {outdir}/")


def calibrate(indir=CALIB_DIR, out=CALIB_FILE):
    """Fit intrinsics + distortion. Do this once, then never move the lens."""
    files = sorted(glob.glob(os.path.join(indir, "*.png")))
    if len(files) < 8:
        sys.exit(f"only {len(files)} shots in {indir}/ - want 15-20")

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_M

    obj_pts, img_pts, shape = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        if not ok:
            print(f"  skip {os.path.basename(f)} (no board)")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
    print(f"using {len(obj_pts)} of {len(files)} shots")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, shape, None, None)

    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        errs.append(cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj))
    np.savez(out, K=K, dist=dist, rms=rms, image_size=shape)

    print(f"\nRMS reprojection error: {rms:.4f} px  (under ~0.5 is good, over 1.0 redo)")
    print(f"per-image mean error: min {min(errs):.3f}  max {max(errs):.3f} px")
    print(f"focal length px: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
    print(f"principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})  image {shape}")
    print(f"wrote {out}")


def load_calib(path=CALIB_FILE):
    if not os.path.exists(path):
        sys.exit(f"no {path} - run --capture-calib then --calibrate first")
    z = np.load(path)
    return z["K"], z["dist"]


# --------------------------------------------------------------------------- #
class MarkerTracker:
    """Detects markers and returns pose in the CAMERA frame (metres)."""

    def __init__(self, K, dist, marker_m=MARKER_M):
        self.K, self.dist, self.marker_m = K, dist, marker_m
        self.objp = _marker_object_points(marker_m)

    def detect(self, gray):
        corners, ids, _ = DETECT(gray)
        if ids is None:
            return {}
        out = {}
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2).astype(np.float32)
        return out

    def poses(self, gray) -> dict:
        """-> {id: {'t': xyz metres, 'rvec': ..., 'px': marker size in pixels}}"""
        res = {}
        for mid, c in self.detect(gray).items():
            ok, rvec, tvec = cv2.solvePnP(
                self.objp, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4])
                                     for i in range(4)]))
            res[mid] = {"t": tvec.flatten(), "rvec": rvec.flatten(), "px": side_px}
        return res


# --------------------------------------------------------------------------- #
def size_test():
    """Live pixel counts for the three test markers on page 1 of the PDF."""
    print("Tape the size-test page where the ARM will be. q to quit.\n")
    hist = {}
    with Camera() as cam:
        while True:
            gray = cam.frame()
            corners, ids, _ = DETECT(gray)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            seen = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                for c, i in zip(corners, ids.flatten()):
                    p = c.reshape(4, 2)
                    px = float(np.mean([np.linalg.norm(p[k] - p[(k + 1) % 4])
                                        for k in range(4)]))
                    seen[int(i)] = px
                    hist.setdefault(int(i), []).append(px)
            for row, (mid, px) in enumerate(sorted(seen.items())):
                verdict = ("TOO SMALL" if px < 30 else
                           "marginal" if px < 50 else
                           "good" if px < 100 else "excellent")
                cv2.putText(vis, f"id {mid}: {px:5.1f} px  {verdict}",
                            (10, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2)
            cv2.imshow("marker size test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print(f"\n{'id':>4} {'size':>9} {'mean px':>9} {'detected':>9}")
    known = {10: "1.25 in", 11: "1.00 in", 12: "0.75 in"}
    for mid in sorted(hist):
        v = np.array(hist[mid])
        print(f"{mid:>4} {known.get(mid,'?'):>9} {v.mean():>9.1f} {len(v):>9}")
    if hist:
        print("\nunder ~30 px flickers, 50-100 px reliable, >100 px good pose")


def noise_floor(true_sep_mm: float, n: int = 50, id_a: int = 0, id_b: int = 1):
    """Measure the vision system's own error, with the arm absent.

    Two markers a caliper-measured distance apart, estimated n times. The
    spread is the floor below which no transfer gap can be interpreted. This
    is the substitute, on hardware, for a simulator's exact determinism.
    """
    K, dist = load_calib()
    tr = MarkerTracker(K, dist)
    meas, misses = [], 0
    print(f"measuring |{id_a} - {id_b}| against a true {true_sep_mm:.2f} mm, {n} frames")
    with Camera() as cam:
        while len(meas) < n:
            p = tr.poses(cam.frame())
            if id_a in p and id_b in p:
                meas.append(float(np.linalg.norm(p[id_a]["t"] - p[id_b]["t"])) * 1000)
            else:
                misses += 1
                if misses > n * 4:
                    sys.exit(f"markers {id_a}/{id_b} not both visible - check framing")
            time.sleep(0.02)

    v = np.array(meas)
    bias = v.mean() - true_sep_mm
    print(f"\n  measured mean : {v.mean():8.2f} mm")
    print(f"  true          : {true_sep_mm:8.2f} mm")
    print(f"  bias          : {bias:+8.2f} mm   <- systematic; scale/calibration error")
    print(f"  std dev       : {v.std(ddof=1):8.2f} mm   <- random; THE NOISE FLOOR")
    print(f"  min / max     : {v.min():8.2f} / {v.max():.2f} mm")
    print(f"  dropped frames: {misses}")
    out = {"true_mm": true_sep_mm, "n": len(meas), "mean_mm": float(v.mean()),
           "bias_mm": float(bias), "std_mm": float(v.std(ddof=1)),
           "min_mm": float(v.min()), "max_mm": float(v.max()),
           "dropped_frames": misses}
    with open("vision_noise_floor.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote vision_noise_floor.json")
    print("Report no transfer gap smaller than a few times the std dev.")
    print("A large bias usually means the printed marker is not the size you told")
    print("the code it is - re-measure it with calipers and fix MARKER_M.")


def preview():
    """Frame the workspace and sanity-check detection."""
    print("q quits. Get the whole future workspace in frame, then measure the\n"
          "camera-to-arm distance with a tape - that is your working distance.")
    with Camera() as cam:
        while True:
            gray = cam.frame()
            corners, ids, _ = DETECT(gray)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.putText(vis, f"{gray.shape[1]}x{gray.shape[0]}  "
                             f"markers: {sorted(ids.flatten().tolist()) if ids is not None else '-'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--capture-calib", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--size-test", action="store_true")
    ap.add_argument("--noise-floor", action="store_true")
    ap.add_argument("--sep", type=float, default=None,
                    help="caliper-measured marker centre separation, mm")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if args.preview:
        preview()
    elif args.capture_calib:
        capture_calib_shots(args.n if args.n != 50 else 20)
    elif args.calibrate:
        calibrate()
    elif args.size_test:
        size_test()
    elif args.noise_floor:
        if args.sep is None:
            sys.exit("--noise-floor needs --sep <true separation in mm>")
        noise_floor(args.sep, args.n)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
