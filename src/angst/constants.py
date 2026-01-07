"""Module to hold useful constants.

JAX doesn't play well with astropy units, and there is not yet a comparable package for JAX.
We include here pre-converted quantities and constants as JAX-compatible types.

Attributes
----------
YR_SEC: float
    Number of seconds in 1 year
YR_DAY: float
    Number of days in 1 year
YR_HZ: float
    Frequency in Hz of 1/yr
YR_NHZ: float
    Frequency in nHz of 1/yr
DAY_SEC: float
    Number of seconds in 1 day
"""

import astropy.units as u

YR_SEC: float = u.yr.to(u.s)
YR_DAY: float = u.yr.to(u.day)
YR_HZ: float = 1 / YR_SEC
YR_NHZ: float = YR_HZ * 1e9

DAY_SEC: float = u.day.to(u.s)
