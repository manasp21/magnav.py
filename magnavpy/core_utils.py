import numpy as np
from typing import Union
from .constants import R_EARTH, E_EARTH
# -*- coding: utf-8 -*-
"""
Core utility functions for MagNavPy.
"""

def get_years(year: float, day_of_year: float) -> float:
    """
    Convert year and day of year to decimal years.

    Args:
        year: Year (e.g., 2023).
        day_of_year: Day of the year (1-365 or 1-366 for leap year).

    Returns:
        Decimal year.
    """
    return year + (day_of_year - 1) / 365.25
def dn2dlat(dn: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert north-south position (northing) difference to latitude difference.

    Args:
        dn: north-south position (northing) difference [m]
        lat: nominal latitude [rad]

    Returns:
        dlat: latitude difference [rad]
    """
    dlat = dn * np.sqrt(1 - (E_EARTH * np.sin(lat))**2) / R_EARTH
    return dlat

def de2dlon(de: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert east-west position (easting) difference to longitude difference.

    Args:
        de: east-west position (easting) difference [m]
        lat: nominal latitude [rad]

    Returns:
        dlon: longitude difference [rad]
    """
    dlon = de * np.sqrt(1 - (E_EARTH * np.sin(lat))**2) / R_EARTH / np.cos(lat)
    return dlon

def dlat2dn(dlat: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert latitude difference to north-south position (northing) difference.

    Args:
        dlat: latitude difference [rad]
        lat: nominal latitude [rad]

    Returns:
        dn: north-south position (northing) difference [m]
    """
    dn = dlat / np.sqrt(1 - (E_EARTH * np.sin(lat))**2) * R_EARTH
    return dn

def dlon2de(dlon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert longitude difference to east-west position (easting) difference.

    Args:
        dlon: longitude difference [rad]
        lat: nominal latitude [rad]

    Returns:
        de: east-west position (easting) difference [m]
    """
    de = dlon / np.sqrt(1 - (E_EARTH * np.sin(lat))**2) * R_EARTH * np.cos(lat)
    return de