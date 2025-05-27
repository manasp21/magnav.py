API: Functions
==============

The following is a full listing of the public functions within the package.

In the Python version, these functions would typically be documented using Sphinx's autodoc directives. For example, to document a function, you would use:

.. code-block:: rst

   .. autofunction:: MagNavPy.src.module_name.function_name

Replace ``module_name.function_name`` with the actual path to the function within the ``MagNavPy.src`` package.
You might list several such directives here for all public functions.

For a more comprehensive approach, if all functions are within a specific module or submodule, you could use:

.. code-block:: rst

   .. automodule:: MagNavPy.src.some_module
      :members:
      :undoc-members:
      :show-inheritance:

This section will need to be populated with the specific ``autofunction`` or ``automodule`` directives corresponding to the public functions in the ``MagNavPy`` library.