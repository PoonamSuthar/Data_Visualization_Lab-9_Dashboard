import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the stroke prediction dataset
app = dash.Dash(__name__)
app.title = "Stroke Prediction Analytics Dashboard"

# Read data
df = pd.read_csv("C:/Users/Radhe-Radhe/Downloads/healthcare-dataset-stroke-data.csv")
df = df.drop('id', axis=1)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Convert categorical variables to numerical
cat_vars = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for var in cat_vars:
    cat_list = 'var' + '_' + var

# Register all departments for callbacks
gender_list = df['gender'].unique().tolist()
work_type_list = df['work_type'].unique().tolist()

scaler = StandardScaler()
joblib.dump(scaler, 'scaler.joblib')




def description_card():
    return html.Div(
        id="description-card",
        style={"textAlign": "center", "paddingTop": "25px", "paddingBottom": "25px"},
        children=[
            html.H1("Stroke Prediction"),
        ],        
    )

def generate_control_card():
    return html.Div(
        id="control-card",
        style={"display": "flex", "flexDirection": "column", "justifyContent": "flex-start", "alignItems": "center",
               "paddingTop": "25px", "paddingBottom": "25px", "paddingLeft": "50px", "paddingRight": "50px"},
        children=[
            html.Div([
                html.Div([
                    html.P("Select Gender:"),
                    dcc.Dropdown(
                        id="gender-select",
                        options=[{"label": i, "value": i} for i in gender_list],
                        value=gender_list[0],
                    ),
                ], className='six columns'),
                html.Div([
                    html.P("Select Work Type:"),
                    dcc.Dropdown(
                        id="work-type-select",
                        options=[{"label": i, "value": i} for i in work_type_list],
                        value=work_type_list[0],
                    ),
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.P("Heart Disease:"),
                    dcc.Dropdown(
                        id="admit-select",
                        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        value=0
                    ),
                ], className='six columns'),
                html.Div([
                    html.P("Hypertension:"),
                    dcc.Dropdown(
                        id="hypertension-select",
                        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        value=0
                    ),
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.P("Age:"),
                    dcc.RangeSlider(
                        id='age-slider',
                        min=0,
                        max=100,
                        step=1,
                        value=[0, 100],
                        marks={
                            0: {'label': '0'},
                            20: {'label': '20'},
                            40: {'label': '40'},
                            60: {'label': '60'},
                            80: {'label': '80'},
                            100: {'label': '100'}
                        },
                    ),
                ], className='six columns'),
                                html.Div([
                    html.P("BMI:"),
                    dcc.RangeSlider(
                        id='bmi-slider',
                        min=10,
                        max=60,
                        step=1,
                        value=[10, 60],
                        marks={
                            10: {'label': '10'},
                            20: {'label': '20'},
                            30: {'label': '30'},
                            40: {'label': '40'},
                            50: {'label': '50'},
                            60: {'label': '60'}
                        },
                    ),
                ], className='six columns'),
            ], className='row'),
        ],
    )

def generate_charts_card():
    return html.Div(
        id="charts-card",
        style={"display": "flex", "flexDirection": "row", "justifyContent": "space-around",
               "alignItems": "center", "paddingTop": "25px", "paddingBottom": "25px"},
        children=[
            html.Div([
                dcc.Graph(id='gender-pie-chart')
            ], className='six columns'),

            html.Div([
                dcc.Graph(id='age-bar-chart')
            ], className='six columns'),
        ],
    )

def generate_histogram_card():
    return html.Div(
        id="histogram-card",
        style={"paddingTop": "25px", "paddingBottom": "25px", "paddingLeft": "50px", "paddingRight": "50px"},
        children=[
            html.Div([
                dcc.Graph(id='bmi-histogram')
            ]),
        ],
    )

def generate_prediction_card():
    return html.Div(
        id="prediction-card",
        style={"paddingTop": "25px", "paddingBottom": "25px", "paddingLeft": "50px", "paddingRight": "50px"},
        children=[
            html.Div([
                html.Button('Predict Stroke', id='predict-button', n_clicks=0, style={"width": "100%"}),
            ], className='twelve columns'),
            html.Br(),
            html.Div([
                html.H3("Prediction Result: ", id='prediction-result', style={"margin": "10px"}),
            ], className='twelve columns'),
        ],
    )


def generate_heatmap(df, gender, admit, hypertension):
    filtered_df = df[(df["gender"] == gender) & (df["heart_disease"] == admit) & (df["hypertension"] == hypertension)]
    fig = px.density_heatmap(filtered_df, x="age", y="bmi",nbinsx=20, nbinsy=20,color_continuous_scale=[[0, 'white'], [1, 'blue']])
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))
    return fig

def generate_scatter_plot(df, gender, admit, hypertension):
    filtered_df = df[(df["gender"] == gender) & (df["heart_disease"] == admit) & (df["hypertension"] == hypertension)]
    fig = px.scatter(filtered_df, x="age", y="avg_glucose_level", color="stroke")
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))
    return fig



# Define the app layout
app.layout = html.Div(
    id="big-app-container",
    children=[
        # Define the header
        html.Div(
            id="header",
            children=[
                description_card(),
            ],
        ),
        # Define the main content
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="controls-container",
                    children=[
                        generate_control_card(),
                        generate_histogram_card(),
                    ],
                    style={"display": "flex", "flexDirection": "row", "justifyContent": "center", "alignItems": "center"}
                ),
                html.Div(
                    id="charts-container",
                    children=[
                        generate_charts_card(),
                        generate_prediction_card(),
                    ],
                    style={"display": "flex", "flexDirection": "row", "justifyContent": "center", "alignItems": "center"}
                ),
                html.Div(
                    children=[
                        dcc.Graph(id="heatmap"),
                        dcc.Graph(id="scatter-plot"),
                    ],
                    style={"display": "flex", "flexDirection": "row", "justifyContent": "center", "alignItems": "center"}
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "alignItems": "center"}

        )

    ],
    style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "alignItems": "center"}

)


@app.callback(
    Output("heatmap", "figure"),
    [Input("gender-select", "value"),
    Input("admit-select", "value"),
    Input("hypertension-select", "value"),],
)

def update_heatmap(gender, admit, hypertension):
    fig = generate_heatmap(df, gender, admit, hypertension)
    return fig

@app.callback(
    Output("scatter-plot", "figure"),
    [Input("gender-select", "value"),
    Input("admit-select", "value"),
    Input("hypertension-select", "value"),],
)

def update_scatter_plot(gender, admit, hypertension):
    fig = generate_scatter_plot(df, gender, admit, hypertension)
    return fig



# Define the callback functions
@app.callback(
    Output('gender-pie-chart', 'figure'),
    [Input('gender-select', 'value')]
)
def update_gender_pie_chart(gender):
    gender_counts = df[df['gender'] == gender]['stroke'].value_counts()
    labels = ['No Stroke', 'Stroke']
    values = [gender_counts[0], gender_counts[1]]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title='Stroke by Gender')
    return fig

@app.callback(
    Output('age-bar-chart', 'figure'),
    [Input('gender-select', 'value'),
     Input('work-type-select', 'value'),
     Input('admit-select', 'value'),
     Input('hypertension-select', 'value'),
     Input('age-slider', 'value')]
)
def update_age_bar_chart(selected_gender, selected_work_type, selected_admit, selected_hypertension, selected_age):
    age_df = df[(df['gender'] == selected_gender) & (df['work_type'] == selected_work_type) & (df['heart_disease'] == selected_admit) & (df['hypertension'] == selected_hypertension) & (df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1])]
    age_count = age_df['age'].value_counts().sort_index()
    fig = go.Figure(
        data=[go.Bar(x=age_count.index, y=age_count.values)]
    )
    fig.update_layout(
        title_text="Age Count",
        xaxis_title="Age",
        yaxis_title="Count",
        title_x=0.5,
    )
    return fig

# Define the callback for BMI histogram
@app.callback(
    Output('bmi-histogram', 'figure'),
    [Input('gender-select', 'value'),
     Input('work-type-select', 'value'),
     Input('admit-select', 'value'),
     Input('hypertension-select', 'value'),
     Input('age-slider', 'value'),
     Input('bmi-slider', 'value')]
)
def update_bmi_histogram(selected_gender, selected_work_type, selected_admit, selected_hypertension, selected_age, selected_bmi):
    bmi_df = df[(df['gender'] == selected_gender) & (df['work_type'] == selected_work_type) & (df['heart_disease'] == selected_admit) & (df['hypertension'] == selected_hypertension) & (df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1]) & (df['bmi'] >= selected_bmi[0]) & (df['bmi'] <= selected_bmi[1])]
    fig = px.histogram(bmi_df, x="bmi", nbins=20)
    fig.update_layout(
        title_text="BMI Distribution",
        xaxis_title="BMI",
        yaxis_title="Count",
        title_x=0.5,
    )
    return fig
@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("gender-select", "value"),
    State("age-slider", "value"),
    State("work-type-select", "value"),
    State("admit-select", "value"),
    State("hypertension-select", "value"),
    State("bmi-slider", "value")
    )
def predict_stroke(n_clicks, gender, age_range, work_type, admit, hypertension, bmi_range):
    if n_clicks > 0:
# Filter the dataframe based on the user's selections
        filtered_df = df[(df["gender"] == gender) & 
                         (df["work_type"] == work_type) & 
                         (df["ever_married"] == admit) & 
                         (df["hypertension"] == hypertension)]

        # Convert age and bmi to numerical values
        age = (age_range[0] + age_range[1]) / 2
        bmi = (bmi_range[0] + bmi_range[1]) / 2

        # Scale the numerical features
        scaled_features = scaler.transform([[age, bmi]])

        # Make the prediction
        prediction = model.predict_proba(scaled_features)[0][1]
        prediction_text = f"The probability of having a stroke is {prediction:.2f}"
        return prediction_text
    else:
        return ""
    

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)

