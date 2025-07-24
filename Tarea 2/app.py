import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import webbrowser
import threading
import os

# Cargar CSV con nombre de columnas en minúsculas
csv_path = os.path.join(os.path.dirname(__file__), 'recipe_site_traffic_2212_clustered.csv')
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()  # Asegura nombres en minúsculas

# Inicializar app
app = Dash(__name__)
app.title = "Dashboard de Recetas Agrupadas"

# Layout
app.layout = html.Div([
    html.H1("Dashboard de Recetas Agrupadas", style={
        'textAlign': 'center',
        'fontFamily': 'Segoe UI',
        'color': '#6a1b9a',
        'marginTop': '2rem'
    }),

    html.Div([
        html.Label("Selecciona un Clúster:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='cluster-selector',
            options=[{'label': f"Clúster {c}", 'value': c} for c in sorted(df['cluster'].unique())],
            value=0,
            style={'marginBottom': '1rem'}
        )
    ], style={'width': '50%', 'margin': '0 auto'}),

    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram'),

    html.H3("Recetas del Clúster Seleccionado", style={
        'textAlign': 'center',
        'marginTop': '2rem',
        'color': '#4a148c'
    }),
    html.Div(id='tabla', style={'width': '80%', 'margin': '0 auto'})
])

# Callback para actualizar los gráficos y la tabla
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('tabla', 'children'),
    Output('histogram', 'figure'),
    Input('cluster-selector', 'value')
)
def update_dashboard(cluster_value):
    df_filtered = df[df['cluster'] == cluster_value]

    fig = px.scatter(
        df_filtered,
        x='calories',
        y='carbohydrate',
        color='cluster',
        title='Visualización de Clústeres (Calorías vs Carbohidratos)',
        hover_data=['category', 'protein', 'sugar'],
        color_discrete_sequence=px.colors.sequential.Plasma
    )

    hist = px.histogram(
        df_filtered,
        x='category',
        title='Distribución de Categorías',
        color='category',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    table = html.Table(
        [html.Thead(html.Tr([html.Th(col) for col in df_filtered.columns]))] +
        [html.Tbody([
            html.Tr([html.Td(df_filtered.iloc[i][col]) for col in df_filtered.columns])
            for i in range(min(len(df_filtered), 10))
        ])],
        style={'width': '100%', 'border': '1px solid gray'}
    )

    return fig, table, hist

# Ejecutar servidor con navegador automático
if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
