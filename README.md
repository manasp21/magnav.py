# MagNavPy

[![PyPI version](https://img.shields.io/pypi/v/magnavpy.svg)](https://pypi.org/project/magnavpy/) <!-- Placeholder -->
[![Build Status](https://img.shields.io/travis/com/yourusername/magnavpy.svg)](https://travis-ci.com/yourusername/magnavpy) <!-- Placeholder -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MagNavPy is a Python library for magnetic navigation research and development. It is a port of the original [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) (Julia) package, providing tools for simulating magnetic navigation scenarios, processing magnetometer data, compensating for aircraft magnetic noise, and implementing navigation filters.

## Key Features

*   **Magnetic Navigation Simulation**: Tools to simulate and analyze magnetic navigation performance.
*   **Magnetometer Data Processing**: Utilities for handling and processing raw magnetometer data.
*   **Aircraft Magnetic Noise Compensation**: Implementation of algorithms like Tolles-Lawson for compensating magnetic interference.
*   **Navigation Filters**: Includes navigation filters such as Extended Kalman Filters (EKF) tailored for magnetic navigation.
*   **Magnetic Anomaly Map Utilities**: Functions for working with magnetic anomaly maps.
*   **Data Visualization**: Plotting functions to visualize flight data, magnetic maps, and filter outputs.

## Original Project

This project is a Python conversion of the [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) library, originally developed by the MIT AI Accelerator. We acknowledge and thank the original authors for their foundational work.

## Installation

### Prerequisites

*   Python 3.9 or higher.
*   **GDAL**: This library has a dependency on GDAL, which needs to be installed manually on your system due to its complex installation process. Please refer to the [official GDAL installation guide](https://gdal.org/download.html#binaries) for instructions specific to your operating system.

### Steps

1.  **Clone the repository (optional, if you want to install from source):**
    ```bash
    git clone https://github.com/yourusername/MagNavPy.git # Replace with actual URL
    cd MagNavPy
    ```

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    To install the required Python packages, run:
    ```bash
    pip install -r requirements.txt
    ```
    If you are developing the library, you might want to install it in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

Here's a basic example of how you might import and use a function from MagNavPy (this is a conceptual example):

```python
from magnavpy import compensation
from magnavpy.common_types import FlightData # Assuming FlightData is a relevant type

# Load or create your flight data
# flight_data = FlightData(...) # Placeholder for actual data loading/creation

# Example: Apply a compensation model (details depend on actual API)
# compensated_data = compensation.apply_tolles_lawson(flight_data, model_parameters)

# print("Compensation applied.")
```

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

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to open an issue to discuss your ideas or submit a pull request.
(More detailed contribution guidelines may be added to a `CONTRIBUTING.md` file in the future.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming a `LICENSE` file will be added, or state "MIT License" directly). The original MagNav.jl project is also licensed under the MIT License.

## Acknowledgements

*   The developers and contributors of the original [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl).
*   The broader open-source community for the tools and libraries that make this project possible.