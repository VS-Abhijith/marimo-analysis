# analysis.py
# Email: 22ds3000188@ds.study.iitm.ac.in

import marimo

__generated_with__ = "0.1.0"
app = marimo.App()


@app.cell
def __(mo):
    # Create an interactive slider widget
    # This slider controls the sample size for the random dataset
    sample_slider = mo.ui.slider(start=10, stop=200, step=10, value=50, label="Sample Size")
    sample_slider
    return sample_slider,


@app.cell
def __(sample_slider):
    # Display dynamic markdown based on slider state
    mo.md(f"### Current Sample Size: **{sample_slider.value}**")
    return


@app.cell
def __(sample_slider):
    import numpy as np
    import pandas as pd

    # Generate random dataset based on slider sample size
    np.random.seed(42)
    x = np.random.rand(sample_slider.value)
    y = 2 * x + np.random.normal(0, 0.1, sample_slider.value)

    df = pd.DataFrame({"X": x, "Y": y})
    df.head()
    return df, np, pd, x, y


@app.cell
def __(df):
    import matplotlib.pyplot as plt

    # Scatter plot of the generated dataset
    plt.scatter(df["X"], df["Y"])
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Relationship between X and Y")
    plt.show()
    return plt,
    

@app.cell
def __(df):
    # Calculate correlation between X and Y
    corr = df["X"].corr(df["Y"])
    mo.md(f"**Correlation between X and Y:** {corr:.4f}")
    return corr,
    

if __name__ == "__main__":
    app.run()
