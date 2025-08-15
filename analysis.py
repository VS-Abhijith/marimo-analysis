# analysis.py
# Author: Abhijith
# Email: 22ds3000188@ds.study.iitm.ac.in
# Description: Interactive Marimo notebook demonstrating relationship between variables

import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Cell 1: Create dataset ---
# We generate a synthetic dataset of X and Y values.
# The variable 'data' created here will be reused in later cells.
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 2, 100)
data = pd.DataFrame({"X": x, "Y": y})

# --- Cell 2: Define slider widget ---
# This slider allows us to choose how many points to sample from 'data'.
# The variable 'slider.value' will flow into Cell 3 and Cell 4.
slider = mo.ui.slider(10, 100, value=50)

# --- Cell 3: Sample dataset based on slider ---
# This cell depends on:
#   - 'data' from Cell 1
#   - 'slider.value' from Cell 2
# It produces 'sample', which will be used in Cell 5 to plot the data.
sample = data.sample(slider.value, random_state=42)

# --- Cell 4: Dynamic Markdown ---
# This Markdown cell depends directly on 'slider.value' from Cell 2.
# When the slider changes, this text automatically updates.
mo.md(f"""
### Interactive Relationship Demo  
Currently sampling **{slider.value}** points from the dataset.

🟢 Each change in the slider updates this text and the plot below.
""")

# --- Cell 5: Visualization ---
# This visualization depends on:
#   - 'sample' from Cell 3
# It shows how the sampled points (X, Y) are distributed.
fig, ax = plt.subplots()
ax.scatter(sample["X"], sample["Y"], alpha=0.7, label="Sampled data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Relationship with Noise")
ax.legend()
fig
