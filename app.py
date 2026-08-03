import dash
from dash import html, dcc

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
app.title = "GNN Diagnostic Hub"

server = app.server

app.layout = html.Div([
    # Global Top Navigation Header
    html.Div([
        html.H2("🩺 GNN Medical Diagnostic Suite", style={'margin': '0', 'color': '#2C3E50'}),
        html.Div([
            dcc.Link("🏠 Home Overview", href="/", style={'marginRight': '20px', 'fontWeight': 'bold', 'color': '#34495E', 'textDecoration': 'none'}),
            dcc.Link("⚡ Predict Engine", href="/predict", style={'fontWeight': 'bold', 'color': '#34495E', 'textDecoration': 'none'}),
        ], style={'marginTop': '10px'})
    ], style={'padding': '15px 25px', 'backgroundColor': '#FFFFFF', 'borderBottom': '1px solid #E0E0E0', 'marginBottom': '20px'}),

    # Dynamic Page Container (Dash automatically loads pages here)
    dcc.Store(id='prediction-store', storage_type='memory'),
    dash.page_container
], style={'fontFamily': 'sans-serif', 'backgroundColor': '#F8F9FA', 'minHeight': '100vh', 'margin': '0'})

if __name__ == '__main__':
    app.run(debug=True, port=8051)