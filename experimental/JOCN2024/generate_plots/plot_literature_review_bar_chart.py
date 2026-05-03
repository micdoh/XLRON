"""Stacked bar chart of RL-RSA literature review publications by year and category.

Extracted from rsa_literature_review_bar_chart.ipynb.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import configure_style

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Color scheme for publication categories
CATEGORY_COLORS = {
    'Grooming': '#FFA500',
    'Defragmentation': '#FF4500',
    'Survivability': '#32CD32',
    'Toolkit': '#FFD700',
    'Multicast': '#8A2BE2',
    'Multicore': '#FF69B4',
    'Multiband': '#20B2AA',
    'Routing': '#87CEFA',
    'RWA': '#4682B4',
    'RMSA': '#191970',
    'RSA': '#0000CD',
    'Other': '#808080',
}


def categorize_paper(tags):
    """Classify a paper into RWA/RSA/RMSA/Other based on manual tags."""
    categories = ['RWA', 'RSA', 'RMSA']
    try:
        tags_lower = tags.lower()
    except Exception:
        return 'Other'
    for category in categories:
        if category.lower() in tags_lower:
            return category
    return 'Other'


def plot_literature_review(csv_path):
    df = pd.read_csv(csv_path)

    # Convert 'Publication Year' to numeric, dropping any non-numeric values
    df['Publication Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')
    df = df.dropna(subset=['Publication Year'])

    # Group years 2018 and earlier into 'pre-2019'
    df['Publication Year'] = df['Publication Year'].apply(
        lambda x: 'pre-2019' if x <= 2018 else x
    )

    # Apply the categorization
    df['Manual Tags'] = df['Manual Tags'].fillna('Other')
    df['Category'] = df['Manual Tags'].apply(categorize_paper)

    # Group by year and category, and count the papers
    grouped = df.groupby(['Publication Year', 'Category']).size().unstack(fill_value=0)

    # Sort the columns in the desired order
    column_order = ['RWA', 'RSA', 'RMSA', 'Other']
    grouped = grouped.reindex(columns=column_order, fill_value=0)

    # Sort the index (years) to ensure 'pre-2019' comes first
    year_order = ['pre-2019'] + sorted(
        [year for year in grouped.index if year != 'pre-2019']
    )
    grouped = grouped.reindex(year_order)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('white')

    grouped.plot(
        kind='bar', stacked=True, ax=ax,
        color=[CATEGORY_COLORS.get(x, '#000000') for x in grouped.columns],
    )

    plt.xticks(rotation=0, ha='center')
    plt.xlabel('Year of Publication', labelpad=10)
    plt.ylabel('Publication Count')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'RL_RSA_litreview_barchart.png')


def main():
    configure_style(font_size=20, axes_label_size=22, tick_size=20, legend_size=20)
    csv_path = Path(__file__).resolve().parents[1] / 'results' / 'RL RSA.csv'
    plot_literature_review(csv_path)


if __name__ == '__main__':
    main()
