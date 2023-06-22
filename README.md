# CalciMetrics

Welcome to CalciMetrics, a Python library dedicated to the analysis and visualization of calcium imaging data. This tool is designed to assist with the processing of fluorescence microscopy data, specifically relating to the analysis of calcium indicators. It provides a collection of methods to perform data preprocessing, smoothing, detrending, and visualization.
Features

    Data Loading and Preprocessing: Load and preprocess data from different formats, normalizing and preparing it for further analysis.

    Heatmap Visualization: Generate a heatmap of your calcium imaging data. This provides a visual representation of the changes in fluorescence over time for each region of interest (ROI). You can also optionally mark stimulus start and end times on the heatmap.

    Gaussian Smoothing: Apply a Gaussian smoothing function to reduce noise in your data. This function includes a visualization tool that displays the raw and smoothed data side-by-side, and also shows the Gaussian kernel used for smoothing.

    Detrending: Apply either an exponential decay or a linear detrending method to correct for fluorescence decay or linear trends in the baseline. This function also includes a visualization tool to compare the raw, smoothed, and detrended data.

Features To Be Added In The Future

    Calcium Imaging Analysis for Astrocytes: In the future, CalciMetrics will provide a comprehensive suite of analysis tools for studying calcium imaging in astrocytes. This includes tools for identifying and quantifying astrocyte calcium events, as well as tools for analyzing the spatial and temporal dynamics of these events.

    Calcium Event Probability: This feature will estimate the likelihood of a calcium event occurring at any given time point, based on the observed data. This is a powerful tool for understanding the temporal dynamics of calcium signaling, and can provide insights into the underlying biological processes driving these events.

Installation

To install CalciMetrics, clone the repository and use Python to run the scripts. Please note that CalciMetrics depends on several Python libraries, including numpy, pandas, matplotlib, and scipy. Make sure these are installed in your environment. If you're using pip, you can install these dependencies with:

bash

pip install numpy pandas matplotlib scipy

After installing the dependencies, clone the repository:

bash

git clone https://github.com/<YourUsername>/CalciMetrics.git

Replace <YourUsername> with your GitHub username. You can then navigate to the cloned directory and import the library in your Python scripts.
Contributing

Contributions to CalciMetrics are welcome! Please read our contributing guidelines and code of conduct before you start.
License

CalciMetrics is licensed under the terms of the MIT license. See the LICENSE file for more details.

CalciMetrics makes it easier to work with calcium imaging data, offering a comprehensive toolset for preprocessing, analysis, and visualization. Enjoy using CalciMetrics for your research needs!
