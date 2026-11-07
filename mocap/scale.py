"""THE SCALE: pixels per metre, from the board when the body agrees with it, else from an adult's leg.

A box the detector calls a skateboard is the rider's board only if the leg it implies is a person's. On the free clips a dancer's shoes
and a long jumper's feet were called a skateboard in 31 and 16 frames, under the feet each time, and the board made their legs 2.2 m and
8.3 m long. The leg is the one the depth step itself will take: its 95th percentile in the picture (straight and across the frame). With no board,
that leg is an adult's 0.87 m: it was the MEDIAN pictured leg until 7 Nov, and a rider crouched through most of a clip (his leg bent, its
picture short) had his straight leg made 1.26 m long (the rules' R8 found it on the kickflip)."""
import numpy as np

BOARD_M = 0.81       # a deck is 31-33 in long (0.79-0.84 m); side-on to a camera in front of the rider
LEG_M = 0.87         # an adult's leg, hip to ankle, when there is no board to read
# a person's leg, hip to ankle, is 0.49 of their height (thigh 0.245 H + shank 0.246 H: Drillis & Contini 1966, the segment-length
# table in D. A. Winter, Biomechanics and Motor Control of Human Movement): 0.64-1.08 m for heights of 1.3-2.2 m, and 5 % for the
# tracker's jitter. Measured 7 Nov: real boards read 0.77-1.01 m on seven riders; the false ones 2.18-8.30 m.
LEG_RANGE = (0.60, 1.15)


def pick(bw, leg95_px, leg_range=LEG_RANGE):
    """bw: the board box's width in each frame (NaN where none); leg95_px: the pictured leg (hip to knee to ankle), its 95th percentile,
    pixels. Returns (pixels per metre, where the scale came from, why the board was refused or None)."""
    bw = np.asarray(bw, float)
    why = None
    if np.isfinite(bw).sum() > 20:
        ppm = float(np.nanpercentile(bw, 75) / BOARD_M)
        leg = float(leg95_px / ppm)
        if leg_range[0] <= leg <= leg_range[1]:
            return ppm, 'the board (0.81 m)', None
        why = f'the board would make the leg {leg:.2f} m (a person\'s is {leg_range[0]:.2f}-{leg_range[1]:.2f} m): not the rider\'s board'
    return float(leg95_px / LEG_M), 'an adult leg (0.87 m)', why
