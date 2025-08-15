# analysis.py
# Author: Abhijith
# Email: 22ds3000188@ds.study.iitm.ac.in
# Description: Interactive Marimo notebook demonstrating relationship between variables

import marimo as mo
import pandas as pd
import numpy as np

# --- Cell 1: Dataset creation ---
# Generate synthetic data (x, y with noise)
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 2, 100)
data = pd.DataFrame({"X": x, "Y": y})

# --- Cell 2: Interactive slider ---
# This slider controls the size of the sample
slider = mo.ui.slider(10, 100, value=50)

# --- Cell 3: Variable dependency on slider ---
# Sample depends on slider value
sample = data.sample(slider.value, random_state=42)

# --- Cell 4: Dynamic Markdown output ---
mo.md(f"""
### Interactive Relationship Demo  
Currently sampling **{slider.value}** points from the dataset.

🟢 Each increase in slider updates this plot in real time.
""")

# --- Cell 5: Visualization depending on slider ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(sample["X"], sample["Y"], alpha=0.7, label="Sampled data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Relationship with Noise")
ax.legend()
fig
