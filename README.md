# Sensor Plots — Synthetic Temperature Data

A small utility that generates synthetic temperature readings for two sensors and produces three publication-quality plots (scatter, overlaid histogram, and box plots) saved to a single PNG.

## Installation

1. Activate the project Conda environment (recommended):

   conda activate ece105

2. Install dependencies with conda or mamba:

   conda install -n ece105 numpy matplotlib
   # or with mamba (faster):
   mamba install -n ece105 numpy matplotlib

(If you prefer pip and a virtual environment, install numpy and matplotlib into your chosen environment.)

## Usage

Run the script to generate example data and save a composite figure:

    python generate_plots.py

This writes `sensor_analysis.png` to the current directory by default.

## Example output

The generated PNG contains three side-by-side panels:

- Scatter: timestamped temperature readings from Sensor A (blue) and Sensor B (orange).
- Overlaid histogram: frequency histograms for both sensors with dashed vertical lines showing each sample mean.
- Box plots: side-by-side notched boxplots comparing the two sensors' distributions (median, IQR, and outliers).

## AI tools used and disclosure

Usage:

- Copilot in terminal to generate code
- Claude browser to assist with concept understanding and copilot prompt generation
