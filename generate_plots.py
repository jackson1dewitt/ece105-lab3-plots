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

def plot_scatter(sensor_a: np.ndarray, sensor_b: np.ndarray, timestamps: np.ndarray, ax) -> None:
    """Draw a scatter plot of two sensors' readings versus time onto an Axes.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        1-D array of Sensor A temperature readings (shape (200,), float64).
    sensor_b : numpy.ndarray
        1-D array of Sensor B temperature readings (shape (200,), float64).
    timestamps : numpy.ndarray
        1-D array of timestamps in seconds (shape (200,), float64). These are
        used for the x-axis values and are expected to be sorted or monotonic.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object to draw onto. The function modifies this
        Axes in place (labels, title, legend, grid) and returns None.

    Returns
    -------
    None

    Notes
    -----
    - Uses the same visual styling as the notebook: 'tab:blue' and
      'tab:orange' markers, alpha for slight transparency, a legend, and
      a light dashed grid for readability.
    """
    # Plot Sensor A and Sensor B on the provided axes
    ax.scatter(timestamps, sensor_a, color='tab:blue', s=30, alpha=0.7,
               label='Sensor A (25±3°C)')
    ax.scatter(timestamps, sensor_b, color='tab:orange', s=30, alpha=0.7,
               label='Sensor B (27±4.5°C)')

    # Labels, title, legend and grid
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Sensor temperature readings vs time (scatter)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    return None

# Intent: plot_histogram takes an ax, data array, title, and xlabel.
# It plots a frequency histogram of the data with labeled axes and a title.
# NumPy-style docstring. Modifies ax in place, returns None.

def plot_histogram(data: np.ndarray, ax, title: str = 'Histogram', xlabel: str = 'Value', bins: int = 30, *, color: str = 'tab:blue', alpha: float = 0.5) -> None:
    """Plot a frequency histogram of a single data array on the provided Axes.

    Parameters
    ----------
    data : numpy.ndarray
        1-D array of numeric observations to histogram.
    ax : matplotlib.axes.Axes
        Axes to draw the histogram on. Modified in place.
    title : str, optional
        Title for the axes (default: 'Histogram').
    xlabel : str, optional
        Label for the x-axis (default: 'Value').
    bins : int, optional
        Number of histogram bins (default: 30).
    color : str, optional
        Color used to fill the bars (default: 'tab:blue').
    alpha : float, optional
        Transparency for the bars (default: 0.5).

    Returns
    -------
    None

    Notes
    -----
    - Uses ax.hist so this can be embedded inside a larger figure (subplots).
    - Adds a vertical dashed line at the sample mean and annotates it.
    """
    # Draw histogram
    counts, bins_edges, patches = ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black')

    # Mean line and label
    mean_val = float(np.mean(data))
    ax.axvline(mean_val, color=color, linestyle='--', linewidth=1)
    ylim = ax.get_ylim()
    # place label slightly below top
    ax.text(mean_val, ylim[1] * 0.9, f'mean={mean_val:.2f}', color=color, ha='center', va='top')

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(False)

    return None

# Intent: plot_boxplot takes an ax, a list of data arrays, a list of group labels, title, and ylabel.
# It plots a side-by-side boxplot comparing distributions across groups with labeled axes and a title.
# NumPy-style docstring. Modifies ax in place, returns None.

def plot_boxplot(data_list, labels, ax, title: str = 'Box plots', ylabel: str = 'Value') -> None:
    """Create side-by-side box plots comparing multiple distributions.

    Parameters
    ----------
    data_list : Sequence[numpy.ndarray]
        Sequence (list/tuple) of 1-D numeric arrays to plot as boxes. Each
        element corresponds to one box in the plot.
    labels : Sequence[str]
        Labels for each group; length must match ``data_list``.
    ax : matplotlib.axes.Axes
        Axes to draw the boxplots on. Modified in place.
    title : str, optional
        Title for the axes (default: 'Box plots').
    ylabel : str, optional
        Label for the y-axis (default: 'Value').

    Returns
    -------
    None

    Notes
    -----
    - Uses notched boxplots (notch=True) and patch_artist=True so boxes can be
      colored to match the project's color scheme.
    """
    # Create the boxplot on the provided axes
    bp = ax.boxplot(data_list, notch=True, patch_artist=True, labels=labels, widths=0.6, showfliers=True)

    # Color boxes if number matches a common palette
    palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, box in enumerate(bp['boxes']):
        color = palette[i % len(palette)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    # Style medians
    for median in bp['medians']:
        median.set(color='black', linewidth=1)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    return None