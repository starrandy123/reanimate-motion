"""THE CLOCK FROM GRAVITY: a body in the air falls at 9.81 m/s²; the fall it takes in a video's frames says how many video frames make
a real second, and so how many times slower than life the video runs. Cameras record at standard rates and edits slow them by standard
amounts, so a measured factor within 15 % of one of them is taken as that one (the measured one is always reported)."""
import numpy as np
FACTORS = [1, 2, 2.5, 4, 5, 8, 10, 16, 20, 32, 33.33, 40]     # 120/240/960/1000 fps recordings played at 24/25/30/60

def fit_slowmo(t, y, fps, snap=True):
    """t: video frame numbers, y: height in metres (up) of a body in free flight. Returns (measured, taken, rms_m) or None."""
    t = np.asarray(t, float); y = np.asarray(y, float); ok = np.isfinite(y)
    if ok.sum() < 8: return None
    for _ in range(3):                                         # a parabola, outliers (3 robust sigmas) dropped twice
        p = np.polyfit(t[ok], y[ok], 2); res = y - np.polyval(p, t); mad = np.nanmedian(np.abs(res[ok])) * 1.4826 + 1e-9; ok = ok & (np.abs(res) < 3 * mad)
    g = -2 * p[0]                                              # metres per video frame², falling
    if g <= 0: return None
    measured = float(np.sqrt(9.81 / g) / fps)
    near = min(FACTORS, key=lambda q: abs(np.log(q / measured)))
    taken = float(near) if snap and abs(np.log(near / measured)) < np.log(1.15) else measured
    return measured, taken, float(np.sqrt(np.mean(res[ok] ** 2)))
