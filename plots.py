import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(df, labels):
    df['label_name'] = df['label'].map(labels)
    label_counts = df['label_name'].value_counts()
    label_counts = df['label_name'].value_counts()

    sns.set_style("whitegrid")

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title('Class Distribution', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Annotate bars with counts and percentages
    total = float(len(df))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5, '{:.0f} ({:.1f}%)'.format(height, (height / total) * 100),
                ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()
