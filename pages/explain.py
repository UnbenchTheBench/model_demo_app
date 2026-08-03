import dash
from dash import html, dcc

dash.register_page(__name__, path='/explain', name='Explainability')

layout = html.Div([
    html.Div([
        dcc.Link("← Back to Predict Workbench", href="/predict", style={'color': '#3498DB', 'textDecoration': 'none', 'fontWeight': 'bold'}),
        html.H3("🧠 Model Decision & Grad-CAM Interpretability", style={'marginTop': '15px', 'color': '#2C3E50'}),
        html.P("Visualizing convolutional layer activations and gradient feature attribution.", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
    ], style={'marginBottom': '20px'}),

    html.Div([
        # Explanation Card
        html.Div([
            html.H4("Grad-CAM (Gradient-Weighted Class Activation Mapping)", style={'marginTop': '0', 'color': '#2C3E50'}),
            html.P("Grad-CAM highlights regions in the input image that were most influential in generating the model's confidence prediction."),
            
            html.Div([
                html.H5("🔥 How to Interpret the Activation Map:", style={'color': '#2980B9', 'marginTop': '0'}),
                html.Ul([
                    html.Li([html.Strong("🔴 Red / Warm Areas: "), "High gradient activation. Regions the neural network focused on most."]),
                    html.Li([html.Strong("🔵 Blue / Cool Areas: "), "Low gradient impact. Neutral background regions with little effect."]),
                    html.Li([html.Strong("🎯 Focus Alignment: "), "Ensures the model focuses on biological structures rather than background artifacts."])
                ], style={'marginBottom': '0'})
            ], style={'backgroundColor': '#EBF5FB', 'padding': '15px', 'borderRadius': '6px', 'borderLeft': '4px solid #3498DB', 'marginBottom': '20px'}),

            html.H5("Feature Layer Analysis", style={'color': '#34495E'}),
            html.P("1. Spatial Gradients computed from final Conv2D bottlenecks.\n2. Non-Maximum Suppression (NMS) applied to isolate key features.")

        ], style={'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})
    ])
], style={'padding': '0 25px'})