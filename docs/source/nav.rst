Navigation Algorithms
=====================

The following are key functions related to navigation algorithms.

MagNav Filter Model
-------------------

.. autofunction:: magnavpy.model_functions.create_model

MagNav filter model internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: magnavpy.model_functions.create_P0

.. autofunction:: magnavpy.model_functions.create_Qd

.. autofunction:: magnavpy.model_functions.get_pinson

Cramér–Rao Lower Bound
----------------------

.. autofunction:: magnavpy.ekf.crlb

Extended Kalman Filter
----------------------

.. autofunction:: magnavpy.ekf.ekf

Run Filter (with additional options)
------------------------------------

.. autofunction:: magnavpy.magnav.run_filt