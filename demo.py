# Test parameters
import sys
import numpy as np

from scipy.constants import speed_of_light, pi

if __name__ == "__main__" and __package__ is None:
    sys.path.append('..')
    __package__ = "doamusic"
# import doamusic
from doamusic import music
from doamusic import _music
from doamusic import util

from util import antenna_array, makesamples, awgn

n_antennas = 6
freq = 2.477e9  # signal frequency
nsamp = 21
snr = -6
wavelength = speed_of_light / freq
ants = antenna_array(n_antennas, "linear")

# Incoming signals
s1_aoa = (pi / 2, 0)
s2_aoa = (pi / 4, 0)

s1, s2 = [makesamples(ants, *i, num_samples=nsamp) for i in (s1_aoa, s2_aoa)]
samples = s2 + s1
samples = awgn(samples, snr)

# add noise to s1 and s2
s1 = awgn(s1, snr)
s2 = awgn(s2, snr)

# Solving the problem
R = music.covar(samples)
est = music.Estimator(ants, R, nsignals=2)

# Display found results in degrees
res = est.doasearch()
res = np.rad2deg(res)
print(res)
