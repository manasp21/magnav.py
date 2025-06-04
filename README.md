# MagNavPy

[![PyPI version](https://img.shields.io/pypi/v/magnavpy.svg)](https://pypi.org/project/magnavpy/) <!-- Placeholder -->
[![Build Status](https://img.shields.io/travis/com/yourusername/magnavpy.svg)](https://travis-ci.com/yourusername/magnavpy) <!-- Placeholder -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MagNavPy is a Python library for magnetic navigation research and development. It is a port of the original [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) (Julia) package, providing tools for simulating magnetic navigation scenarios, processing magnetometer data, compensating for aircraft magnetic noise, and implementing navigation filters.

## Key Features

*   **Data Handling**: Load, process, and manage flight path, INS, and magnetometer data, including built-in datasets.
*   **Magnetic Anomaly Maps**: Utilities for loading, manipulating (e.g., upward continuation), and interpolating magnetic anomaly maps, with access to built-in global and regional maps.
*   **Aeromagnetic Compensation**: Implementations of classical methods like Tolles-Lawson and advanced Neural Network-based models for compensating aircraft magnetic noise.
*   **Navigation Algorithms**: Tools for magnetic navigation filtering, including Extended Kalman Filters (EKF) and the MagNav filter model, along with performance analysis using the Cramér–Rao Lower Bound (CRLB).
*   **Simulation & Analysis**: Simulate magnetic navigation scenarios and analyze performance.
*   **Data Visualization**: Plotting functions to visualize flight data, magnetic maps, and filter outputs.

## Core Concepts

MagNavPy utilizes several key data structures to organize and manage data:

*   [`Map`](magnavpy/common_types.py:9): Represents a magnetic anomaly map.
*   [`Traj`](magnavpy/magnav.py:40): Stores flight trajectory data.
*   [`INS`](magnavpy/magnav.py:43): Holds Inertial Navigation System data.
*   [`XYZ`](magnavpy/magnav.py:48): A general structure for flight data including position, time, and magnetic field measurements.
*   [`EKF_RT`](magnavpy/ekf.py:81): Represents the state of a Real-Time Extended Kalman Filter.
*   [`CompParams`](magnavpy/compensation.py:87), [`LinCompParams`](magnavpy/compensation.py:90), [`NNCompParams`](magnavpy/compensation.py:93): Structures for holding parameters for different compensation models.

## Original Project

This project is a Python conversion of the [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) library, originally developed by the MIT AI Accelerator. We acknowledge and thank the original authors for their foundational work.

## Installation

### Prerequisites

*   Python 3.9 or higher.
*   **GDAL**: This library depends on GDAL.
    *   **For Windows users:** It is strongly recommended to install GDAL using pre-compiled wheels from sources like Christoph Gohlke's Unofficial Windows Binaries for Python Extension Packages. Ensure you download the wheel compatible with your Python version (e.g., Python 3.13) and system architecture (e.g., `win_amd64`). Direct installation via `pip install gdal` can often lead to compilation issues on Windows. After downloading the appropriate `.whl` file, you can install it using pip (e.g., `pip install GDAL-3.9.0-cp313-cp313-win_amd64.whl`).
    *   **For other operating systems:** Please refer to the [official GDAL installation guide](https://gdal.org/download.html#binaries) for instructions.

### Project Dependencies

Beyond the prerequisites, MagNavPy relies on several Python packages for its functionality. All required packages are listed in the [`requirements.txt`](requirements.txt:0) file and can be installed as described in the installation steps. Key dependencies include:

*   **gdal**: For geospatial data operations (Python bindings, requires system-level GDAL).
*   **pandas**: For data manipulation and analysis.
*   **torch**: For deep learning models and tensor computations.
*   **matplotlib**: For plotting and visualization.
*   **h5py**: For interacting with HDF5 files.
*   **scipy**: For scientific and technical computing.
*   **jax**: For high-performance numerical computing and machine learning research.
*   **toml**: For parsing TOML configuration files.
*   **scikit-learn**: For machine learning tools.
*   **statsmodels**: For statistical modeling.
*   **pytest**: For running the test suite.

Please ensure GDAL is installed on your system *before* running `pip install -r requirements.txt`.

### Steps

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/yourusername/MagNavPy.git # Replace with actual URL
    ```
    Navigate into the cloned repository's root directory (where `requirements.txt` is located). All subsequent installation commands should be run from this directory.

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure your virtual environment is active and you are in the repository root directory. To install the required Python packages, run:
    ```bash
    pip install -r requirements.txt
    ```
    If you are developing the library, you might want to install it in editable mode:
    ```bash
    pip install -e .
    ```

## Data and Artifact Management

MagNavPy requires specific data artifacts, such as magnetic anomaly maps and flight datasets, to function correctly. Currently, these artifacts are not managed or downloaded automatically by this Python port.

**Manual Placement:**
Users are required to manually obtain these artifacts and place them in a directory named `artifacts_data`. This directory should be located at the **root of the `MagNavPy` repository** (e.g., if you cloned the repository into a folder named `MagNavPy`, the path would be `MagNavPy/artifacts_data/`).

**Obtaining Artifacts:**
Information regarding the required artifacts and their original sources can be found in the [`Artifacts.toml`](../MagNav.jl/Artifacts.toml:1) file of the original [MagNav.jl project](https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/main/Artifacts.toml). Please refer to this file to identify and download the necessary data.

## Usage

MagNavPy provides functions for various stages of magnetic navigation processing. Here are examples of key functions:

**Data Loading:**
Use functions like [`create_xyz0`](magnavpy/create_xyz.py:9), [`get_xyz20`](magnavpy/create_xyz.py:11), or [`get_XYZ`](magnavpy/create_xyz.py:13) to load flight data. Built-in datasets like `sgl_2020_train` and `sgl_2021_train` are also available.

**Map Handling:**
Load magnetic anomaly maps using [`get_map`](magnavpy/map_utils.py:9). Functions like [`upward_fft`](magnavpy/map_utils.py:25) are available for map manipulation.

**Aeromagnetic Compensation:**
Train and test compensation models using [`comp_train`](magnavpy/compensation.py:16) and [`comp_test`](magnavpy/compensation.py:21). The library supports classical Tolles-Lawson and various Neural Network-based models (e.g., `:m1`, `:m2a`, `:m2b`, `:m2c`, `:m2d`, `:m3s`, `:m3v`).

**Navigation Filtering:**
Run navigation filters, such as the Extended Kalman Filter, using the [`run_filt`](magnavpy/magnav.py:33) function.

For more detailed examples, please refer to the `examples/` directory (if available) or the test scripts in the `tests/` directory.

## Documentation

Full documentation, including API references and usage guides, is generated using Sphinx.

To build the documentation locally:
```bash
cd docs
make html
```
Then, open `docs/build/html/index.html` in your web browser.

The documentation may also be hosted online in the future.

## Testing

To run the test suite, navigate to the root directory of the project and use `pytest`:
```bash
pytest tests/
```
This will execute all tests defined in the `tests/` directory.

## Current Status and Known Issues

This Python port of MagNav.jl is an ongoing development effort. While significant progress has been made, users should be aware of the following:

*   **Ongoing Porting:** The library is actively being ported from Julia. Not all features and functionalities of the original MagNav.jl may be fully implemented or tested.
*   **Environment and Imports:** Initial challenges with setting up the Python environment, particularly GDAL installation and resolving relative import paths within the `magnavpy` package, have been largely addressed.
*   **Testing Coverage:** The original task mentioned "pytest errors and otherwise." A comprehensive review and execution of the `pytest` test suite to ensure full functionality and identify remaining issues is still pending.
*   **Manual Data Artifacts:** As detailed in the "Data and Artifact Management" section, data artifacts are currently handled manually. There is no automated download or management system within this Python port.
*   **Documentation:** Documentation is being actively developed. While Sphinx documentation is available, it may not yet cover all aspects of the Python port comprehensively.

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to open an issue to discuss your ideas or submit a pull request.
(More detailed contribution guidelines may be added to a `CONTRIBUTING.md` file in the future.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming a `LICENSE` file will be added, or state "MIT License" directly). The original MagNav.jl project is also licensed under the MIT License.

## Acknowledgements

*   The developers and contributors of the original [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl).
*   The broader open-source community for the tools and libraries that make this project possible.