# 22ds3000188@ds.study.iitm.ac.in
# Marimo notebook style file using # %% cell markers
# Purpose: interactive analysis demo showing relationship between variables
# Requirements satisfied:
# - Email included as comment (top of file)
# - At least two cells with variable dependencies
# - Interactive slider (ipywidgets)
# - Dynamic markdown output responding to widget state
# - Comments documenting data flow between cells

# %% [cell1]
# Cell 1: Imports and Data Load
# Data produced here (df_raw) will be consumed by downstream cells.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets

# Set plotting style
sns.set(style="whitegrid")

# Load an example dataset (Iris). This is our canonical df_raw.
df_raw = sns.load_dataset("iris")  # columns: sepal_length, sepal_width, petal_length, petal_width, species

# Quick sanity-check output
print("Loaded dataset with shape:", df_raw.shape)
df_raw.head()

# %% [cell2]
# Cell 2: Derived variables & helper functions
# This cell depends on df_raw from cell1.
# We create functions that compute filtered DF and statistics.
def filter_by_threshold(df: pd.DataFrame, feature: str, threshold: float) -> pd.DataFrame:
    """
    Returns a filtered dataframe keeping rows where df[feature] >= threshold.
    Dependent on df_raw loaded in cell1.
    """
    return df[df[feature] >= threshold].reset_index(drop=True)


def compute_correlation(df: pd.DataFrame, x: str, y: str) -> float:
    """
    Compute Pearson correlation between columns x and y.
    """
    if df.shape[0] < 2:
        return float("nan")
    return float(df[x].corr(df[y]))


def summary_table(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Return simple summary stats for selected features.
    """
    return df[features].describe().T[["mean", "std", "min", "max"]]


# Define default columns to examine
x_col = "sepal_length"
y_col = "petal_length"

# Provide a baseline filtered df for immediate inspection
df_filtered_default = filter_by_threshold(df_raw, x_col, df_raw[x_col].min())

# %% [cell3]
# Cell 3: Interactive Widgets & Dynamic Markdown Output
# This cell uses functions from cell2 and data from cell1.
# It defines an interactive slider that drives filtering threshold,
# dynamically updates markdown summary and redrawing of scatter plot.

# Create slider for threshold on x_col
min_val = float(df_raw[x_col].min())
max_val = float(df_raw[x_col].max())
step = (max_val - min_val) / 100.0

threshold_slider = widgets.FloatSlider(
    value=min_val,
    min=min_val,
    max=max_val,
    step=step,
    description=f"{x_col} ≥",
    continuous_update=True,
    readout_format=".2f",
    layout=widgets.Layout(width="70%")
)

# Dropdowns to allow selecting which columns to compare (extra interactivity)
x_dropdown = widgets.Dropdown(
    options=list(df_raw.select_dtypes(include=[np.number]).columns),
    value=x_col,
    description="X:",
)
y_dropdown = widgets.Dropdown(
    options=list(df_raw.select_dtypes(include=[np.number]).columns),
    value=y_col,
    description="Y:",
)

# Output area to update plot and markdown
out = widgets.Output()

def update_plot_and_text(change=None):
    """
    Main handler: uses current widget state to filter df, compute correlation,
    and update both the markdown summary and the scatter plot.
    This function depends on:
      - df_raw (from cell1)
      - filter_by_threshold, compute_correlation, summary_table (from cell2)
      - the widget values threshold_slider.value, x_dropdown.value, y_dropdown.value
    """
    # Read widget state
    threshold = float(threshold_slider.value)
    x_feature = x_dropdown.value
    y_feature = y_dropdown.value

    # Data flow: df_raw -> filter_by_threshold -> df_filtered
    df_filtered = filter_by_threshold(df_raw, x_feature, threshold)

    # Compute metrics: correlation and counts
    corr = compute_correlation(df_filtered, x_feature, y_feature)
    n_rows = df_filtered.shape[0]

    # Dynamic markdown text
    md_text = f"### Dynamic Summary\n\n"
    md_text += f"- **Filter:** keep rows where `{x_feature} >= {threshold:.2f}`\n"
    md_text += f"- **Rows after filter:** **{n_rows}**\n"
    if np.isnan(corr):
        md_text += "- **Pearson correlation:** Not enough data\n"
    else:
        md_text += f"- **Pearson correlation ({x_feature}, {y_feature}):** **{corr:.3f}**\n"

    # Add a small summary table for the filtered data
    if n_rows > 0:
        stats = summary_table(df_filtered, [x_feature, y_feature])
        md_text += "\n**Summary stats (filtered):**\n\n"
        # convert summary table to markdown table
        md_table = stats.reset_index().rename(columns={"index": "feature"})
        md_text += md_table.to_markdown(index=False)

    # Update the output area atomically
    with out:
        clear_output(wait=True)
        display(Markdown(md_text))

        # Draw scatter plot under the markdown
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = df_filtered["species"].astype('category').cat.codes if "species" in df_filtered.columns else None
        scatter = ax.scatter(df_filtered[x_feature], df_filtered[y_feature],
                             c=colors, cmap="viridis", alpha=0.9, s=45, edgecolors="k")
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{y_feature} vs {x_feature} (filtered: {n_rows} rows)")
        # If species exists, add legend
        if "species" in df_filtered.columns and n_rows > 0:
            # create a legend mapping codes -> species labels
            species_labels = list(df_filtered["species"].astype('category').cat.categories)
            handles = []
            for i, lab in enumerate(species_labels):
                handles.append(plt.Line2D([], [], marker="o", color=scatter.cmap(i / max(1, len(species_labels)-1)),
                                          linestyle="", markersize=6, markeredgecolor="k"))
            ax.legend(handles, species_labels, title="species", loc="best", fontsize="small")
        plt.show()

# Wire up observers to widgets to call update on change
threshold_slider.observe(update_plot_and_text, names="value")
x_dropdown.observe(update_plot_and_text, names="value")
y_dropdown.observe(update_plot_and_text, names="value")

# Initial display
display(widgets.VBox([widgets.HBox([x_dropdown, y_dropdown]), threshold_slider, out]))
# Call once to render initial view
update_plot_and_text()

# %% [cell4]
# Cell 4: Notes on data flow and reproducibility
# This cell documents how data flows between cells and how to extend the notebook.
#
# Data flow summary:
# - Cell 1: loads raw data into `df_raw`.
# - Cell 2: defines reusable transformations and metrics:
#     * filter_by_threshold(df, feature, threshold)
#     * compute_correlation(df, x, y)
#     * summary_table(df, features)
# - Cell 3: builds interactive widgets and uses the functions above to produce:
#     * a filtered dataframe (df_filtered)
#     * computed metrics (corr, n_rows)
#     * dynamic markdown summary and a plot
#
# Reproducibility tips:
# - Seed random steps (if you add sampling) with np.random.seed(42)
# - For large datasets, perform server-side aggregation before visualizing
# - Export the current filtered view using: df_filtered.to_csv("filtered_view.csv", index=False)
#
# End of notebook.
