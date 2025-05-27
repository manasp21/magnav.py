Navigation Algorithms
=====================

The following are key functions related to navigation algorithms.

MagNav Filter Model
-------------------

.. autofunction:: MagNavPy.src.model_functions.create_model

MagNav filter model internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: MagNavPy.src.model_functions.create_p0

.. autofunction:: MagNavPy.src.model_functions.create_qd

.. autofunction:: MagNavPy.src.model_functions.get_pinson

Cramér–Rao Lower Bound
----------------------

.. autofunction:: MagNavPy.src.ekf.crlb

Extended Kalman Filter
----------------------

.. autofunction:: MagNavPy.src.ekf.ekf

Run Filter (with additional options)
------------------------------------

.. autofunction:: MagNavPy.src.magnav.run_filt