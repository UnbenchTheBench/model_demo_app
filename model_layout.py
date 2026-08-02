import dash
from dash import html, dcc, Input, Output, State
import json

layout = html.Div([
    html.H2("🧪 GNN Model Interaction Sandbox", style={'fontFamily': 'sans-serif', 'color': '#2C3E50'}),
    html.P("Test inference, inspect feature payloads, and validate raw model predictions.", 
           style={'color': '#7F8C8D', 'fontFamily': 'sans-serif'}),
    
    html.Hr(),

    html.Div([
        # Left Panel: Controls & Input Data Payload
        html.Div([
            html.H3("1. Model Input Config"),
            
            html.Label("Target Model / Checkpoint:"),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'GNN Node Classifier v1', 'value': 'gnn_v1'},
                    {'label': 'Link Prediction Engine v2', 'value': 'link_v2'},
                    {'label': 'Anomaly Detector Baseline', 'value': 'anomaly_base'}
                ],
                value='gnn_v1',
                clearable=False,
                style={'marginBottom': '15px'}
            ),

            html.Label("Sample Feature Payload (JSON):"),
            dcc.Textarea(
                id='model-input-json',
                value=json.dumps({
                    "node_id": "test_node_101",
                    "features": [0.45, 1.2, 0.0, 8.5],
                    "neighbors": ["node_12", "node_88"]
                }, indent=2),
                style={'width': '100%', 'height': '180px', 'fontFamily': 'monospace', 'fontSize': '12px'}
            ),

            html.Button("⚡ Run Model Inference", id="btn-run-inference", style={
                'marginTop': '15px', 'backgroundColor': '#3498DB', 'color': 'white',
                'border': 'none', 'padding': '10px 20px', 'fontWeight': 'bold',
                'borderRadius': '4px', 'cursor': 'pointer', 'width': '100%'
            }, n_clicks=0)
        ], style={'width': '45%', 'padding': '15px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'}),

        # Right Panel: Output Predictions & Diagnostic Logs
        html.Div([
            html.H3("2. Raw Prediction Output"),
            dcc.Loading(
                id="loading-model-output",
                type="circle",
                children=html.Pre(
                    id='model-output-display',
                    children="Awaiting inference trigger...",
                    style={
                        'backgroundColor': '#2C3E50', 'color': '#2ECC71',
                        'padding': '15px', 'borderRadius': '6px',
                        'height': '280px', 'overflowY': 'auto',
                        'fontFamily': 'monospace', 'fontSize': '13px'
                    }
                )
            )
        ], style={'width': '48%', 'padding': '15px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})

    ], style={'display': 'flex', 'justifyContent': 'space-between', 'fontFamily': 'sans-serif'})
])


def register_callbacks(app):
    @app.callback(
        Output('model-output-display', 'children'),
        Input('btn-run-inference', 'n_clicks'),
        State('model-selector', 'value'),
        State('model-input-json', 'value'),
        prevent_initial_call=True
    )
    def simulate_model_inference(n_clicks, selected_model, raw_json):
        try:
            # 1. Parse incoming test payload
            input_data = json.loads(raw_json)
            
            # 2. [PLACEHOLDER] Call your PyTorch / Scikit-learn / TensorFlow model inference here
            # example: output = my_gnn_model.predict(input_data)
            
            mock_response = {
                "status": "SUCCESS",
                "model_used": selected_model,
                "input_node": input_data.get("node_id"),
                "predicted_class": "High Risk",
                "confidence_score": 0.9421,
                "node_embeddings": [0.12, -0.44, 0.89, 0.02],
                "computed_styles": {
                    "shape": "diamond",
                    "color": "#E74C3C"
                }
            }
            return json.dumps(mock_response, indent=2)

        except Exception as e:
            return f"❌ ERROR executing inference:\n{str(e)}"