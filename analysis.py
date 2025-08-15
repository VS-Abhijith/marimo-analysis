# analysis.py
# Author: Abhijith
# Email: 22ds3000188@ds.study.iitm.ac.in
# Description: Interactive Marimo notebook demonstrating relationship between variables

import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Cell 1: Generate dataset ---
# Create synthetic linear dataset with noise
# Output: DataFrame 'data' with columns X, Y
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 2, 100)
data = pd.DataFrame({"X": x, "Y": y})

# --- Cell 2: Interactive slider ---
# Slider widget to select number of points to sample from 'data'
# Output: UI element 'slider'
slider = mo.ui.slider(10, 100, value=50)

# --- Cell 3: Sample dataset based on slider ---
# 'slider.value' determines how many rows to sample from 'data'
# Input: data (Cell 1), slider (Cell 2)
# Output: DataFrame 'sample'
sample = data.sample(slider.value, random_state=42)

# --- Cell 4: Dynamic Markdown explanation ---
# This Markdown updates when 'slider.value' changes
# Input: slider (Cell 2)
mo.md(f"""
### Interactive Relationship Demo  
Currently sampling **{slider.value}** points from the dataset.

🟢 Each increase in slider updates this plot and text in real time.
""")

# --- Cell 5: Visualization ---
# Scatter plot of sampled data
# Input: sample (Cell 3)
fig, ax = plt.subplots()
ax.scatter(sample["X"], sample["Y"], alpha=0.7, label="Sampled data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Relationship with Noise")
ax.legend()
fig
