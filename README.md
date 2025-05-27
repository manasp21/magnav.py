# MagNavPy: A Python Conversion of MagNav.jl

## 1. Project Overview

**MagNav.jl** is a Julia package designed for magnetic navigation research and development. It provides tools for simulating magnetic navigation scenarios, processing magnetometer data, compensating for aircraft magnetic noise, and implementing navigation filters.

The primary goal of the **MagNavPy** project is to convert the functionalities of `MagNav.jl` into Python, creating a comparable library that leverages the Python ecosystem. This allows for easier integration with other Python-based tools and libraries commonly used in scientific computing and aerospace engineering.

## 2. Conversion Process Summary

The conversion from `MagNav.jl` to `MagNavPy` involved several key steps:

1.  **Cloning the Original Repository**: The `MagNav.jl` repository was cloned to serve as the source for the conversion.
2.  **Analysis of Julia Project**: The structure, dependencies (e.g., `Project.toml`), source code, and test suites of `MagNav.jl` were analyzed to understand its components and their interrelations.
3.  **Python Project Setup**:
    *   A new directory, `MagNavPy/`, was created for the Python project.
    *   Initial dependency management files, `MagNavPy/requirements.txt` and `MagNavPy/pyproject.toml`, were set up.
4.  **Source Code Conversion**:
    *   Julia source files (`.jl`) located in `MagNav.jl/src/` were iteratively translated into Python files (`.py`) and placed in `MagNavPy/src/`. This involved manual and semi-automated translation of syntax, idioms, and library calls.
5.  **Test Suite Conversion**:
    *   Julia test files (`.jl`) from `MagNav.jl/test/` were converted to Python test files (`.py`) and placed in `MagNavPy/tests/`.
    *   The `pytest` framework was adopted for the Python test suite.
6.  **Documentation Migration**:
    *   A Sphinx documentation project was initialized in `MagNavPy/docs/`.
    *   Existing Markdown documentation from `MagNav.jl` was reviewed and content was adapted or converted to reStructuredText (`.rst`) for Sphinx.
    *   API reference stubs were created using Sphinx's `sphinx-apidoc` to automatically generate documentation from Python docstrings.
7.  **Documentation Build and Debugging**:
    *   Multiple attempts were made to build the Sphinx documentation.
    *   This involved debugging various issues, including import errors, docstring formatting problems, and `autodoc` configuration.

## 3. Current Status of `MagNavPy`

As of the latest update, the `MagNavPy` project has reached the following status:

*   **Source Code**: All identified core source modules from `MagNav.jl/src/` have been translated into their Python equivalents in `MagNavPy/src/`.
*   **Tests**: All identified test modules from `MagNav.jl/test/` have been translated into Python `pytest` modules in `MagNavPy/tests/`.
*   **Dependencies**:
    *   `MagNavPy/requirements.txt` and `MagNavPy/pyproject.toml` are in place to manage project dependencies.
    *   Key dependencies such as `numpy`, `scipy`, `pandas`, `matplotlib`, `toml`, `scikit-learn`, and `statsmodels` have been included.
    *   `Sphinx` and related extensions were added for documentation.
    *   **GDAL**: Due to its complex installation process across different operating systems, GDAL has been excluded from automated installation via `requirements.txt`. Users are required to install GDAL manually.
*   **Documentation**:
    *   A Sphinx documentation project is set up in `MagNavPy/docs/`.
    *   Content pages (e.g., for compensation, data handling, navigation, maps) and API reference stubs for the translated modules have been created.

## 4. Known Issues & Next Steps

Despite significant progress, several known issues need to be addressed, and further steps are required to complete and refine `MagNavPy`:

*   **Sphinx Documentation Build**: The Sphinx documentation build (`cd MagNavPy/docs && make html`) currently completes but with numerous warnings and errors. The primary causes include:
    *   **Persistent Circular Import Errors**: Issues such as `ImportError: cannot import name 'MapS' from partially initialized module 'MagNavPy.src.magnav'` (most likely due to a circular import) and similar errors involving other modules prevent `autodoc` from successfully importing and documenting several modules.
    *   **Docstring Formatting Error**: A stubborn error related to docstring formatting in `MagNavPy/src/compensation.py` for the `linear_fwd` function (`Unexpected section title.`) persists.
    *   **Autodoc Failures**: As a consequence of the import and formatting errors, `sphinx-autodoc` fails to import and generate documentation for many modules and their members.
    *   **Incomplete HTML Output**: The generated HTML documentation in `MagNavPy/docs/build/html/` is likely incomplete or missing significant portions of the API reference due to the build issues.
*   **GDAL Installation**: GDAL requires manual installation by the user. Clear instructions for different platforms should be provided or linked.
*   **Full Test Execution and Bug Fixing**: All tests in `MagNavPy/tests/` need to be systematically run. Any failing tests will indicate bugs or discrepancies in the ported code that must be identified and fixed.
*   **Code Review & Refinement**: The translated Python code would benefit from a thorough review for Pythonic best practices, logical correctness, performance, and resolution of any remaining subtle issues from the automated or manual translation (e.g., ensuring the SGL flight data implementation details are correctly ported, verifying the `expm_multiply` replacement logic).
*   **Complex Feature Validation**: Detailed validation of complex Julia features translated to Python (e.g., multiple dispatch patterns, macros, specific advanced algorithm implementations) need careful validation in their Python counterparts to ensure they behave as expected.

## 5. How to Use/Develop Further

To use or contribute to the `MagNavPy` project, follow these steps:

1.  **Set up the Environment**:
    *   Clone the `MagNavPy` repository (if not already done).
    *   **Install GDAL Manually**: Before proceeding, ensure that GDAL is installed on your system. Refer to the official GDAL documentation for installation instructions specific to your operating system.
    *   Create and activate a Python virtual environment (recommended):
        \`\`\`bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        \`\`\`
    *   Install the required Python packages:
        \`\`\`bash
        pip install -r MagNavPy/requirements.txt
        \`\`\`
        If you intend to modify the project and its dependencies, you might also want to install it in editable mode:
        \`\`\`bash
        pip install -e ./MagNavPy
        \`\`\`

2.  **Run Tests**:
    *   To execute the test suite, navigate to the root directory of the project (the one containing the `MagNavPy` directory) and run `pytest`:
        \`\`\`bash
        pytest MagNavPy/tests/
        \`\`\`

3.  **Build Documentation**:
    *   To build the Sphinx documentation:
        \`\`\`bash
        cd MagNavPy/docs
        make html
        \`\`\`
    *   The HTML output will be generated in `MagNavPy/docs/build/html/`. Open `MagNavPy/docs/build/html/index.html` in a web browser to view the documentation. Note the "Known Issues" regarding the documentation build.

Contributions, bug reports, and suggestions for improvement are welcome!