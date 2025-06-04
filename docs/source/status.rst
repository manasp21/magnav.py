Current Status and Known Issues
===============================

Ongoing Porting
---------------
The library is actively being ported from Julia. Not all features and functionalities of the original MagNav.jl may be fully implemented or tested.

Environment and Imports
-----------------------
Initial challenges with setting up the Python environment, particularly GDAL installation and resolving relative import paths within the ``magnavpy`` package, have been largely addressed.

Testing Coverage
----------------
The original task mentioned "pytest errors and otherwise." A comprehensive review and execution of the ``pytest`` test suite to ensure full functionality and identify remaining issues is still pending.

Manual Data Artifacts
---------------------
Data artifacts are currently handled manually by placing them in an ``artifacts_data`` directory at the root of the repository. There is no automated download or management system within this Python port. Refer to the original MagNav.jl project's ``Artifacts.toml`` for artifact sources.

Documentation
-------------
Documentation is being actively developed. While Sphinx documentation is available, it may not yet cover all aspects of the Python port comprehensively.