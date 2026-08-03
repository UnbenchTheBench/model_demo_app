import dash
from dash import html, dcc, Input, Output, State, callback

# Import backend prediction engines from predictors directory
from predictors.predict_image import run_image_model
from predictors.predict_text import run_text_model
from predictors.predict_multimodal import run_multimodal_model

# Register page at route '/predict'
dash.register_page(__name__, path='/predict', name='Predict')

layout = html.Div([
    html.Div([
        html.H3("Unified Prediction Hub", style={'margin': '0', 'color': '#2C3E50'}),
        html.P("Select a modality, supply inputs, and trigger modular backend predictors.", style={'margin': '5px 0 0 0', 'color': '#7F8C8D'})
    ], style={'marginBottom': '20px'}),

    html.Div([
        # Left Panel: Modality Selection & Inputs
        html.Div([
            html.Label("1. Select Diagnostic Modality:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='modality-selector',
                options=[
                    {'label': ' 📝 Text Only', 'value': 'text'},
                    {'label': ' 🖼️ Image Only', 'value': 'image'},
                    {'label': ' 🧬 Multimodal (Text + Image)', 'value': 'multimodal'}
                ],
                value='text',
                labelStyle={'display': 'block', 'margin': '8px 0', 'cursor': 'pointer'}
            ),
            html.Hr(),

            # Text Input Container
            html.Div(id='text-input-container', children=[
                html.Label("Clinical Notes / Patient History:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                dcc.Textarea(
                    id='input-clinical-text',
                    placeholder='Type patient symptoms or history...',
                    style={'width': '95%', 'height': '110px', 'borderRadius': '4px', 'marginBottom': '15px'}
                )
            ]),

            # Image Input Container
            html.Div(id='image-input-container', children=[
                html.Label("Diagnostic Image Upload:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                dcc.Upload(
                    id='input-image-file',
                    children=html.Div(['Drag & Drop or ', html.A('Select Medical Image')]),
                    style={
                        'width': '95%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='image-upload-filename', style={'fontSize': '12px', 'color': '#27AE60', 'marginBottom': '15px'})
            ]),
            
            html.Button("⚡ Run Model Prediction", id="btn-run-predict", style={
                'marginTop': '20px', 'backgroundColor': '#2ECC71', 'color': 'white',
                'border': 'none', 'padding': '12px', 'fontWeight': 'bold',
                'borderRadius': '4px', 'cursor': 'pointer', 'width': '100%'
            }, n_clicks=0)

        ], style={'width': '40%', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'}),

        # Right Panel: Output Payload & Visualization
        html.Div([
            html.H3("📊 Prediction Results", style={'marginTop': '0', 'color': '#34495E'}),
            dcc.Loading(
                id='loading-predictions',
                type='circle',
                children=html.Div(
                    id='prediction-output-display',
                    children="Select input modality and click 'Run Model Prediction'.",
                    style={
                        'backgroundColor': '#F8F9FA', 'color': '#2C3E50',
                        'padding': '20px', 'borderRadius': '6px', 'minHeight': '380px',
                        'overflowY': 'auto', 'border': '1px solid #E2E8F0'
                    }
                )
            )
        ], style={'width': '52%', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderRadius': '6px', 'border': '1px solid #E0E0E0'})

    ], style={'display': 'flex', 'justify': 'space-between'})
], style={'padding': '0 25px'})


# Toggle Input Fields based on Modality
@dash.callback(
    Output('text-input-container', 'style'),
    Output('image-input-container', 'style'),
    Input('modality-selector', 'value')
)
def toggle_input_fields(selected_modality):
    if selected_modality == 'text':
        return {'display': 'block'}, {'display': 'none'}
    elif selected_modality == 'image':
        return {'display': 'none'}, {'display': 'block'}
    else:  # Multimodal
        return {'display': 'block'}, {'display': 'block'}


# Show Uploaded Filename Status
@dash.callback(
    Output('image-upload-filename', 'children'),
    Input('input-image-file', 'filename')
)
def show_filename(filename):
    return f"📁 Uploaded: {filename}" if filename else ""


# Dispatch Prediction Callback
@dash.callback(
    Output('prediction-output-display', 'children'),
    Input('btn-run-predict', 'n_clicks'),
    State('modality-selector', 'value'),
    State('input-clinical-text', 'value'),
    State('input-image-file', 'contents'),
    State('input-image-file', 'filename'),
    prevent_initial_call=True
)
def dispatch_prediction(n_clicks, modality, text_val, image_contents, filename):
    if n_clicks == 0:
        return dash.no_update

    try:
        # Route request to target model predictor function
        if modality == 'text':
            results = run_text_model(text_val)
        elif modality == 'image':
            results = run_image_model(image_contents, image_filename=filename)
        elif modality == 'multimodal':
            results = run_multimodal_model(text_val, image_contents, image_filename=filename)
        else:
            return html.Div("Unknown modality selected.", style={'color': '#E74C3C'})

        # Extract values from return dict
        finding = results.get("finding", "Analysis complete.")
        risk_score = results.get("risk_score", 0)
        model_used = results.get("model_used", "Standard Model")
        annotated_image = results.get("annotated_image", None)

        border_color = "#2ECC71" if risk_score < 30 else "#E67E22"
        status_color = "#27AE60" if risk_score < 30 else "#D35400"

        # Build UI Result Layout
        output_content = [
            html.Div([
                html.H4("Analysis Complete", style={'color': status_color, 'marginTop': '0', 'marginBottom': '10px'}),
                html.P([html.Strong("Engine Used: "), model_used], style={'margin': '5px 0'}),
                html.P([html.Strong("Primary Finding: "), finding], style={'margin': '5px 0'}),
                
                # Risk Bar
                html.Div([
                    html.Span(f"Estimated Risk Score: {risk_score}%", style={'fontWeight': 'bold', 'color': status_color, 'fontSize': '14px'}),
                    html.Div(
                        html.Div(style={'width': f'{risk_score}%', 'backgroundColor': border_color, 'height': '100%', 'borderRadius': '4px'}),
                        style={'backgroundColor': '#E0E0E0', 'height': '12px', 'borderRadius': '4px', 'marginTop': '5px'}
                    )
                ], style={'marginTop': '15px'})
            ])
        ]

        # Render Segmentation Image Overlay if returned by run_image_model
        if annotated_image:
            output_content.append(
                html.Div([
                    html.Hr(style={'margin': '15px 0'}),
                    html.Label("🖼️ Predicted Segmentation Mask:", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '8px'}),
                    html.Img(
                        src=annotated_image,
                        style={
                            'width': '100%',
                            'maxHeight': '320px',
                            'objectFit': 'contain',
                            'borderRadius': '6px',
                            'border': '1px solid #BDC3C7'
                        }
                    )
                ])
            )

            # --- NEW: Add Explainability Button ---
            output_content.append(
                html.Div([
                    html.Hr(style={'margin': '15px 0'}),
                    dcc.Link(
                        html.Button("🔍 How Did the Model Decide?", style={
                            'backgroundColor': '#3498DB', 'color': 'white',
                            'border': 'none', 'padding': '10px 15px', 'fontWeight': 'bold',
                            'borderRadius': '4px', 'cursor': 'pointer', 'width': '100%'
                        }),
                        href='/explain'
                    )
                ])
            )

        return html.Div(output_content)

    except Exception as e:
        return html.Div([
            html.H4("⚠️ Model Execution Error", style={'color': '#C0392B', 'marginTop': '0'}),
            html.P(f"Error Details: {str(e)}", style={'color': '#922B21', 'fontFamily': 'monospace', 'fontSize': '12px'})
        ])