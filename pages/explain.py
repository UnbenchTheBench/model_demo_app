import dash
from dash import html, dcc

# Register new page route
dash.register_page(__name__, path='/explain', name='Explainability')

layout = html.Div([
    # Navigation header
    html.Div([
        dcc.Link("← Back to Predict Workbench", href="/predict", style={'color': '#3498DB', 'textDecoration': 'none', 'fontWeight': 'bold'}),
        html.H3("🧠 Model Decision & Interpretability Hub", style={'marginTop': '15px', 'color': '#2C3E50'}),
        html.P("Breakdown of features, confidence distributions, and decision rationale.", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
    ], style={'marginBottom': '20px'}),

    # Main Explainability Panel
    html.Div([
        html.Div([
            html.H4("Explainable AI (XAI) Breakdown", style={'marginTop': '0', 'color': '#34495E'}),
            html.P("This dashboard helps clinicians and users inspect how neural networks process diagnostic inputs."),
            
            html.Div([
                html.H5("🎯 How YOLO Segmentation Works:", style={'color': '#2980B9'}),
                html.Ul([
                    html.Li("Bounding Box Anchors: Scans candidate regions across spatial grid layers."),
                    html.Li("Non-Maximum Suppression (NMS): Filters overlapping candidate masks."),
                    html.Li("Confidence Thresholding: Highlights target regions exceeding the activation barrier.")
                ])
            ], style={'backgroundColor': '#EBF5FB', 'padding': '15px', 'borderRadius': '6px', 'borderLeft': '4px solid #3498DB', 'marginBottom': '20px'}),

            html.Div([
                html.P("💡 Tip: To share prediction data across pages (e.g. Grad-CAM heatmaps or bounding box data), you can save state using Dash's dcc.Store component in app.py.", style={'fontSize': '13px', 'color': '#7F8C8D'})
            ])

        ], style={'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})
    ])
], style={'padding': '0 25px'})