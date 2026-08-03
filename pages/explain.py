import dash
from dash import html, dcc, Input, Output, callback

dash.register_page(__name__, path='/explain', name='Explainability')

layout = html.Div([
    html.Div([
        dcc.Link("← Back to Predict Workbench", href="/predict", style={'color': '#3498DB', 'textDecoration': 'none', 'fontWeight': 'bold'}),
        html.H3("🧠 Model Decision & Grad-CAM Interpretability", style={'marginTop': '15px', 'color': '#2C3E50'}),
        html.P("Visualizing convolutional layer activations and feature attribution.", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
    ], style={'marginBottom': '20px'}),

    html.Div([
        # Grad-CAM Visualization Area
        html.Div(id='explain-content-display', style={'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})
    ])
], style={'padding': '0 25px'})


@callback(
    Output('explain-content-display', 'children'),
    Input('prediction-store', 'data')
)
def render_explanation(store_data):
    if not store_data or not isinstance(store_data, dict):
        return html.Div([
            html.H4("No Prediction Data Found", style={'color': '#7F8C8D'}),
            html.P("Please run a prediction on the Predict workbench page first.")
        ])

    gradcam_img = store_data.get("gradcam_image", None)
    annotated_img = store_data.get("annotated_image", None)
    finding = store_data.get("finding", "N/A")
    risk_score = store_data.get("risk_score", 0)

    elements = [
        html.H4("Grad-CAM Activation Heatmap Analysis", style={'marginTop': '0', 'color': '#2C3E50'}),
        html.P([html.Strong("Prediction Finding: "), finding]),
        html.P([html.Strong("Confidence Risk Score: "), f"{risk_score}%"]),
        html.Hr()
    ]

    if gradcam_img:
        elements.append(
            html.Div([
                html.H5("🔥 Grad-CAM Attention Heatmap", style={'color': '#C0392B'}),
                html.P("Warmer colors (Red/Yellow) highlight exact image regions that contributed most to the model's prediction decision."),
                html.Img(src=gradcam_img, style={'maxWidth': '100%', 'maxHeight': '450px', 'borderRadius': '6px', 'border': '1px solid #BDC3C7', 'marginBottom': '20px'})
            ])
        )
    else:
        elements.append(
            html.Div([
                html.P("⚠️ No Grad-CAM heatmap was generated for this prediction. (This can happen if no objects/segmentation masks exceeded the threshold).", style={'color': '#D35400'})
            ])
        )

    return html.Div(elements)