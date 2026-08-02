import dash
from model_layout import layout, register_callbacks

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "GNN Model Sandbox"

# Server instance exposed for Render deployment
server = app.server

app.layout = layout
register_callbacks(app)

if __name__ == '__main__':
    # Running on port 8051 so it doesn't collide with your graph app on 8050
    app.run(debug=True, port=8051)