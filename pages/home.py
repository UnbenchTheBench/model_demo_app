import dash
from dash import html, dcc

# Register page as the root path '/'
dash.register_page(__name__, path='/', name='Home')

layout = html.Div([
    html.Div([
        html.H2("Welcome to the Diagnostic Model Sandbox", style={'color': '#2C3E50'}),
        html.P("This workspace allows you to test local medical prediction models across three modalities: Text, Image, and Multimodal."),
        html.Hr(),
        dcc.Link("🚀 Launch Predict Engine", href="/predict", style={
            'display': 'inline-block', 'padding': '10px 20px', 'backgroundColor': '#3498DB',
            'color': 'white', 'fontWeight': 'bold', 'borderRadius': '4px', 'textDecoration': 'none'
        })
    ], style={'padding': '30px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})
], style={'padding': '0 25px'})