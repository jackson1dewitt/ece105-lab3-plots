"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data using NumPy
and produces scatter, histogram, and box plot visualizations saved
as PNG files.

Usage
-----
    python generate_plots.py
"""

# Create a function generate_data(seed) that returns sensor_a, sensor_b,
# and timestamps arrays with the same parameters as in the notebook.
# Use NumPy-style docstring with Parameters and Returns sections.

import numpy as np


def generate_data(seed: int):
    """Generate synthetic sensor temperature data.

    The generated data matches the shapes and distribution parameters
    used in the accompanying Jupyter notebook (lab3_sensor_plots.ipynb).

    Parameters
    ----------
    seed : int
        Seed for NumPy's random number generator (passed to
        numpy.random.default_rng) to ensure reproducible outputs.

    Returns
    -------
    sensor_a : numpy.ndarray
        1-D array of shape (200,) with Sensor A temperature readings in
        degrees Celsius. Samples are drawn from a normal distribution
        with mean 25.0 and standard deviation 3.0 (dtype float64).

    sensor_b : numpy.ndarray
        1-D array of shape (200,) with Sensor B temperature readings in
        degrees Celsius. Samples are drawn from a normal distribution
        with mean 27.0 and standard deviation 4.5 (dtype float64).

    timestamps : numpy.ndarray
        1-D array of shape (200,) with timestamps in seconds (dtype
        float64). Values are sampled uniformly from the interval [0, 10)
        and sorted in ascending order to simulate measurements over time.

    Notes
    -----
    - Uses numpy.random.default_rng for modern, explicit RNG control.
    - The ordering of the returned tuple is (sensor_a, sensor_b, timestamps)
      to match the notebook's expectations.
    """
    rng = np.random.default_rng(seed)
    n = 200
    timestamps = np.sort(rng.uniform(0.0, 10.0, n))
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n).astype(np.float64)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n).astype(np.float64)

    return sensor_a, sensor_b, timestamps

# Create plot_scatter(sensor_a, sensor_b, timestamps, ax) that draws
# the scatter plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.