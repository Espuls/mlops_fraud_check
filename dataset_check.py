import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

# Constantes
DATA_FILE_PATH = 'dataset/creditcard.csv'
OUTPUT_PLOT_FILE = 'class.html'
BAR_COLOR = "Red"


# Carrega dataset
def load_data(file_path):
    return pd.read_csv(file_path)


# Inspeciona dataset
def inspect_data(data_frame):
    print(f"Dataset Info - Rows: {data_frame.shape[0]}, Columns: {data_frame.shape[1]}")
    print(data_frame.head())
    print(data_frame.describe())
    missing_data = data_frame.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_data / data_frame.isnull().count() * 100).sort_values(ascending=False)
    return pd.concat([missing_data, missing_percent], axis=1, keys=['Total', 'Percent'])


# Plota a distribuição de classes
def visualize_class_distribution(data_frame, output_file, bar_color):
    class_counts = data_frame["Class"].value_counts()
    class_data = pd.DataFrame({'Class': class_counts.index, 'values': class_counts.values})
    trace = go.Bar(
        x=class_data['Class'],
        y=class_data['values'],
        name="Credit Card Fraud Class Distribution",
        marker=dict(color=bar_color),
        text=class_data['values']
    )
    layout = dict(
        title='Credit Card Fraud Class Distribution (Not fraud = 0, Fraud = 1)',
        xaxis=dict(title='Class', showticklabels=True),
        yaxis=dict(title='Number of Transactions'),
        hovermode='closest',
        width=600
    )
    fig = dict(data=[trace], layout=layout)
    plot(fig, filename=output_file, auto_open=True)


def main():
    # Load data
    data = load_data(DATA_FILE_PATH)

    # Inspect data
    inspect_data(data)

    # Visualize class distribution
    visualize_class_distribution(data, OUTPUT_PLOT_FILE, BAR_COLOR)


if __name__ == "__main__":
    main()