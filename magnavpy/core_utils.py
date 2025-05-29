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