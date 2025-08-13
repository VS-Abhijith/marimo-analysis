# analysis.py
# Email: 22ds3000188@ds.study.iitm.ac.in

import marimo

__generated_with__ = "0.1.0"
app = marimo.App()


# Cell 1: Create and display interactive slider
@app.cell
def __(mo):
    # Interactive slider to control dataset sample size
    sample_slider = mo.ui.slider(start=10, stop=200, step=10, value=50, label="Sample Size")
    sample_slider  # Display widget
    return sample_slider,


# Cell 2: Dynamic markdown that changes with slider value
@app.cell
def __(mo, sample_slider):
    mo.md(f"### Current Sample Size: **{sample_slider.value}**")
    return


# Cell 3: Generate dataset based on slider value
@app.cell
def __(np, pd, sample_slider):
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    x = np.random.rand(sample_slider.value)
    y = 2 * x + np.random.normal(0, 0.1, sample_slider.value)

    df = pd.DataFrame({"X": x, "Y": y})
    df.head()
    return df, np, pd, x, y


# Cell 4: Plot scatter chart of dataset
@app.cell
def __(df, plt):
    import matplotlib.pyplot as plt

    plt.scatter(df["X"], df["Y"])
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Relationship between X and Y")
    plt.show()
    return plt,


# Cell 5: Calculate correlation
@app.cell
def __(df, mo):
    corr = df["X"].corr(df["Y"])
    mo.md(f"**Correlation between X and Y:** {corr:.4f}")
    return corr,


if __name__ == "__main__":
    app.run()
