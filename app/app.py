
# Import required libraries
import joblib
import numpy as np
import pandas as pd
import os
import requests
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from sklearn.model_selection import KFold
from A2modelandprep import LinearRegression, ManualPreprocessor, NoRegularization, LassoPenalty, RidgePenalty
import mlflow
from LoadA3model import a3_model

# Make these classes available in __main__ namespace for pickle compatibility
NoRegularization = NoRegularization
LassoPenalty = LassoPenalty  
RidgePenalty = RidgePenalty
LinearRegression = LinearRegression
ManualPreprocessor = ManualPreprocessor

# Load the trained model from multiple possible paths
MODEL_PATHS = [
    "./car_price.model",  # For Docker deployment
    "./model/car_price.model",  # For root directory local development
    "../model/car_price.model",  # For app directory local development
    "./app/model/car_price.model",  # For root directory with app folder
]

model = None
for MODEL_PATH in MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            break
        except Exception as e:
            print(f"Failed to load model from {MODEL_PATH}. Error: {e}")
            continue

if model is None:
    raise RuntimeError("No valid model found in any of the expected paths")

# Load A2 model package from multiple possible paths
A2_MODEL_PATHS = [
    "./best_model.pkl",  # For Docker deployment
    "./model/best_model.pkl",  # For root directory local development
    "../model/best_model.pkl",  # For app directory local development
    "./app/model/best_model.pkl",  # For root directory with app folder
]

# Try to load A2 model package
a2_model_package = None
a2_model = None
a2_preprocessor = None
a2_features = None

for MODEL_PATH in A2_MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            a2_model_package = joblib.load(MODEL_PATH)
            a2_model = a2_model_package['model']
            a2_preprocessor = a2_model_package['preprocessor']
            a2_features = a2_model_package['features']
            print(f"A2 Model package loaded successfully from {MODEL_PATH}")
            print(f"A2 Package contains: {list(a2_model_package.keys())}")
            break
        except Exception as e:
            print(f"Failed to load A2 model package from {MODEL_PATH}. Error: {e}")
            continue

if a2_model_package is None:
    print("Warning: A2 model package not found or has import issues. A2 page will use original model as fallback.")
    print("This is normal if the model was created in a notebook environment with different imports.")

# Initialize Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for gunicorn if needed later

# Add CSS styling for better UI appearance
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Georgia:wght@400;700&display=swap');
            body {
                font-family: 'Georgia', serif !important;
                margin: 0;
                background-color: #fafafa;
            }
            .nav-bar {
                background-color: #2c3e50;
                padding: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .nav-container {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0 20px;
            }
            .nav-logo {
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-decoration: none;
            }
            .nav-links {
                display: flex;
                gap: 30px;
            }
            .nav-link {
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .nav-link:hover {
                background-color: #34495e;
                text-decoration: none;
                color: white;
            }
            .nav-link.active {
                background-color: #3498db;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            .instruction-card {
                background: white;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create navigation bar component
def create_navbar(current_page):
    return html.Div([
        html.Div([
            html.A("Car Price Predictor", href="/", className="nav-logo"),
            html.Div([
                html.A("Instructions", 
                      href="/", 
                      className=f"nav-link {'active' if current_page == 'home' else ''}"),
                html.A("Predict Price", 
                      href="/predict", 
                      className=f"nav-link {'active' if current_page == 'predict' else ''}"),
                html.A("A2 - Predict Price", 
                      href="/a2-predict", 
                      className=f"nav-link {'active' if current_page == 'a2-predict' else ''}"),
                html.A("A3 - Price Class", 
                      href="/a3-predict", 
                      className=f"nav-link {'active' if current_page == 'a3-predict' else ''}"),
            ], className="nav-links")
        ], className="nav-container")
    ], className="nav-bar")

# Create instructions page layout
def instructions_layout():
    return html.Div([
        create_navbar('home'),
        html.Div([
            html.Div([
                html.H1("Car Price Prediction Assignment System", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
                html.H2("by Dechathon Niamsa-ard [st126235]",
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                
                html.H3("Comparison: Assignment 1 (XGBoost) vs. Assignment 2 (Custom Linear Regression) vs. Assignment 3 (Classification)", 
                       style={'color': '#34495e', 'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("A1 – XGBoost Model", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Strengths:", style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Captures complex, nonlinear relationships between features."),
                            html.Li("Generally achieves higher accuracy (lower MSE, higher R²) on car price prediction."),
                            html.Li("Handles interactions between features automatically.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Limitations:", style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("More of a \"black box\" cause harder to interpret feature effects."),
                            html.Li("Heavier computational cost for training and tuning."),
                            html.Li("Requires more system resources for deployment.")
                        ], style={'color': '#555'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'border': '1px solid #dee2e6'})
                ]),
                
                html.Div([
                    html.H4("A2 – Custom Linear Regression Model", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Strengths:", style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Fully implemented from scratch with gradient descent, regularization options (Ridge, Lasso), and polynomial extensions."),
                            html.Li("Much more interpretable: coefficients directly show how each feature influences car price."),
                            html.Li("Lightweight and efficient — requires less memory and faster to deploy."),
                            html.Li("Includes consistent preprocessing pipeline (imputation, scaling, encoding) and MLflow logging, making it more production-friendly.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Limitations:", style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Linear assumption: struggles with nonlinear patterns in car data."),
                            html.Li("Predictive accuracy may be lower than XGBoost when the dataset has complex feature interactions.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Best Model Configuration (from experiments):", style={'fontWeight': 'bold', 'color': '#8e44ad', 'marginBottom': '8px'}),
                        html.Div([
                            html.P("LinearRegression-method-sto-lr-0.001-init_method-zeros-momentum-False", 
                                   style={'fontFamily': 'monospace', 'backgroundColor': '#f1f3f4', 'padding': '10px', 'borderRadius': '5px', 'color': '#2c3e50', 'marginBottom': '5px'}),
                            html.P("This configuration uses Stochastic Gradient Descent (SGD) with learning rate 0.001, zero initialization, and no momentum.", 
                                   style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
                        ], style={'backgroundColor': '#f8f4ff', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #d1c4e9'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'border': '1px solid #dee2e6'})
                ]),
                
                html.Div([
                    html.H4("A3 – Custom Logistic Regression (Classification)", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Strengths:", style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Converts car price prediction into a classification problem using price quartiles (4 classes: 0-3)."),
                            html.Li("Implements multinomial logistic regression with Ridge regularization from scratch."),
                            html.Li("Uses MLflow for experiment tracking and model deployment."),
                            html.Li("Provides probability-based predictions for price ranges rather than exact values."),
                            html.Li("Same preprocessing pipeline as A2 for consistency.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Limitations:", style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Provides price ranges rather than exact price predictions."),
                            html.Li("Classification boundaries may not perfectly capture price distribution."),
                            html.Li("Less granular than regression approaches for pricing decisions.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Best Model Configuration:", style={'fontWeight': 'bold', 'color': '#8e44ad', 'marginBottom': '8px'}),
                        html.Div([
                            html.P("Simple Logistic Regression with minibatch gradient descent with learning rate 0.001", 
                                   style={'fontFamily': 'monospace', 'backgroundColor': '#f1f3f4', 'padding': '10px', 'borderRadius': '5px', 'color': '#2c3e50', 'marginBottom': '5px'}),
                            html.P("Deployed via MLflow Model Registry for consistent access.", 
                                   style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
                        ], style={'backgroundColor': '#f8f4ff', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #d1c4e9'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '30px', 'border': '1px solid #dee2e6'})
                ]),
                
                html.H3("How the Assignment System Works", style={'color': '#34495e', 'marginBottom': '20px'}),
                html.P([
                    "This assignment system demonstrates three different approaches to car price analysis. "
                    "Assignment 1 uses XGBoost (advanced machine learning), Assignment 2 implements custom linear regression from scratch, "
                    "and Assignment 3 converts the problem to classification using custom logistic regression. "
                    "All models analyze car specifications, condition, and market trends, but provide different types of insights."
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Step-by-Step Instructions", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Ol([
                    html.Li("Navigate to 'Predict Price' (Assignment 1 - XGBoost), 'A2 - Predict Price' (Assignment 2 - Custom Linear Regression), or 'A3 - Price Class' (Assignment 3 - Classification) using the navigation bar above"),
                    html.Li("Fill in the car details you know in the input form"),
                    html.Li("Don't worry if you don't have all information - you can skip any field"),
                    html.Li("For missing fields, each model uses its own imputation techniques learned during training"),
                    html.Li("Click the 'Predict Price' or 'Predict Price Class' button to submit your data"),
                    html.Li("Compare predictions: A1 & A2 give exact prices, A3 gives price class (quartile-based ranges)"),
                    html.Li("The predicted result will appear below the form within moments")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Missing Data Handling", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.P([
                    "Both assignment models intelligently handle missing information using their respective preprocessing strategies learned from market data:"
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                html.Ul([
                    html.Li("Assignment 1 (XGBoost): Uses built-in XGBoost preprocessing with median/mean imputation"),
                    html.Li("Assignment 2 (Custom Linear Regression): Uses custom ManualPreprocessor with median for numerical fields and mean for mileage"),
                    html.Li("Assignment 3 (Custom Logistic Regression): Uses same preprocessing as A2 for consistency"),
                    html.Li("Numerical fields (Year, Kilometers, Owner, Engine, Power): Uses median from training data"),
                    html.Li("Mileage: Uses mean from training data"),
                    html.Li("Categorical fields (Fuel Type, Transmission, Brand): Uses most frequent from training data"),
                    html.Li("All imputation values are learned from the actual car market data during model training")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Important Notes", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Div([
                    html.P("• All models are trained specifically on Petrol and Diesel vehicles", 
                           style={'color': '#e74c3c', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("• A1 & A2 predictions are price estimates based on historical market data", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• A3 predicts price class (0=lowest quartile, 3=highest quartile) rather than exact price", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• The more accurate information you provide, the better the prediction", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• Assignment 1 (XGBoost) typically shows highest accuracy but is less interpretable", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• Assignment 2 & 3 (Custom implementations) are more interpretable but may have different accuracy", 
                           style={'color': '#555'})
                ]),
                
                html.Div([
                    html.A("Try Assignment 1 (XGBoost) →", 
                           href="/predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#3498db',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px',
                               'marginRight': '15px'
                           }),
                    html.A("Try Assignment 2 (Custom Linear Regression) →", 
                           href="/a2-predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#e67e22',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px',
                               'marginRight': '15px'
                           }),
                    html.A("Try Assignment 3 (Classification) →", 
                           href="/a3-predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#9b59b6',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px'
                           })
                ], style={'textAlign': 'center'})
                
            ], className="instruction-card")
        ], className="container")
    ])

# Create helper functions for form elements
def labeled_input(label, id_, type_="number", placeholder="", **kwargs):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Input(
            id=id_, 
            type=type_, 
            placeholder=placeholder, 
            style={
                "width": "100%", 
                "padding": "12px", 
                "border": "2px solid #bdc3c7",
                "borderRadius": "5px",
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            },
            **kwargs
        )
    ], style={"marginBottom": "20px"})

def labeled_dropdown(label, id_, options, value=None):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Dropdown(
            id=id_, 
            options=[{"label": o, "value": o} for o in options], 
            value=value, 
            clearable=True,
            placeholder="Select or leave blank for pipeline imputation",
            style={
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            }
        )
    ], style={"marginBottom": "20px"})

# Create prediction page layout
def prediction_layout():
    return html.Div([
        create_navbar('predict'),
        html.Div([
            html.Div([
                html.H1("Car Price Prediction", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. Leave fields blank if you don't have the information.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px', 'fontSize': '16px'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "year", placeholder="e.g., 2016 (leave blank for median from data)", min=1, step=1),
                        labeled_input("Kilometers Driven", "km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Fuel Type", "fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "engine", placeholder="e.g., 1197 (leave blank for median from data)", min=0, step=1),
                        labeled_input("Max Power (bhp)", "power", placeholder="e.g., 82 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Brand", "brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price", 
                           id="predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#4a9d5b",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Create A2 prediction page layout (perfect clone of prediction_layout)
def a2_prediction_layout():
    return html.Div([
        create_navbar('a2-predict'),
        html.Div([
            html.Div([
                html.H1("A2 - Car Price Prediction", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. Leave fields blank if you don't have the information.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px', 'fontSize': '16px'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "a2-year", placeholder="e.g., 2016 (leave blank for median from data)", min=1, step=1),
                        labeled_input("Kilometers Driven", "a2-km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Fuel Type", "a2-fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "a2-transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "a2-owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "a2-mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "a2-engine", placeholder="e.g., 1197 (leave blank for median from data)", min=0, step=1),
                        labeled_input("Max Power (bhp)", "a2-power", placeholder="e.g., 82 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Brand", "a2-brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price", 
                           id="a2-predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#4a9d5b",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="a2-result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Create A3 prediction page layout for classification
def a3_prediction_layout():
    return html.Div([
        create_navbar('a3-predict'),
        html.Div([
            html.Div([
                html.H1("A3 - Car Price Classification", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. This model predicts price class (0-3) based on quartiles.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px', 'fontSize': '16px'}),
                html.Div([
                    html.P("Price Classes:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#2c3e50'}),
                    html.P("• Class 0: Lowest 25% (Budget cars)", style={'marginBottom': '3px', 'color': '#555'}),
                    html.P("• Class 1: 25%-50% (Economy cars)", style={'marginBottom': '3px', 'color': '#555'}),
                    html.P("• Class 2: 50%-75% (Mid-range cars)", style={'marginBottom': '3px', 'color': '#555'}),
                    html.P("• Class 3: Top 25% (Premium cars)", style={'marginBottom': '15px', 'color': '#555'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px', 'border': '1px solid #dee2e6'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "a3-year", placeholder="e.g., 2016 (leave blank for median from data)", min=1, step=1),
                        labeled_input("Kilometers Driven", "a3-km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Fuel Type", "a3-fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "a3-transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "a3-owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "a3-mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "a3-engine", placeholder="e.g., 1197 (leave blank for median from data)", min=0, step=1),
                        labeled_input("Max Power (bhp)", "a3-power", placeholder="e.g., 82 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Brand", "a3-brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price Class", 
                           id="a3-predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#9b59b6",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="a3-result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Set up main app layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Handle URL routing to display correct page
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/predict':
        return prediction_layout()
    elif pathname == '/a2-predict':
        return a2_prediction_layout()
    elif pathname == '/a3-predict':
        return a3_prediction_layout()
    else:  # Default to instructions page
        return instructions_layout()

# Handle price prediction when button is clicked
@app.callback(
    Output("result-section", "children"),
    Input("predict", "n_clicks"),
    State("year", "value"),
    State("km", "value"),
    State("fuel", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("power", "value"),
    State("brand", "value"),
)
def predict_price(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price' to get your estimate.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for model prediction
    # Leave missing values as NaN/None - the pipeline will handle imputation
    row = pd.DataFrame([{
        "year": float(year) if year is not None else np.nan,
        "km_driven": float(km) if km is not None else np.nan,
        "fuel": str(fuel) if fuel is not None else None,
        "transmission": str(transmission) if transmission is not None else None,
        "owner": float(owner_num) if owner_num is not None else np.nan,  # Model expects numeric owner
        "engine": float(engine) if engine is not None else np.nan,  # Model expects numeric engine
        "max_power": float(power) if power is not None else np.nan,  # Model expects numeric max_power
        "brand": str(brand) if brand is not None else None,
        "mileage": float(mileage) if mileage is not None else np.nan,  # Model expects numeric mileage
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in row.columns:
        if pd.isna(row.at[0, col]) or row.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction and convert from log scale to price
    try:
        # Debug information for troubleshooting
        print("Input row shape:", row.shape)
        print("Input row columns:", list(row.columns))
        print("Input row values:", row.iloc[0].to_dict())
        print("Imputed fields:", imputed_fields)
        
        pred_log = float(model.predict(row)[0])
        price = float(np.exp(pred_log))
        
        print(f"Predicted log price: {pred_log:.4f}")
        print(f"Predicted price: {price:.2f}")
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Estimated Price: {price:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '32px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by Pipeline", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the trained pipeline's imputation strategy:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → pipeline default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': '3px solid #27ae60'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Model type: {type(model)}")
        return html.Div([
            html.H3("Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Handle A2 price prediction when button is clicked
@app.callback(
    Output("a2-result-section", "children"),
    Input("a2-predict", "n_clicks"),
    State("a2-year", "value"),
    State("a2-km", "value"),
    State("a2-fuel", "value"),
    State("a2-transmission", "value"),
    State("a2-owner", "value"),
    State("a2-mileage", "value"),
    State("a2-engine", "value"),
    State("a2-power", "value"),
    State("a2-brand", "value"),
)
def a2_predict_price(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price' to get your estimate.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for A2 model prediction using the model package approach
    sample_data = pd.DataFrame([{
        'year': float(year) if year is not None else np.nan,
        'km_driven': float(km) if km is not None else np.nan,
        'fuel': str(fuel) if fuel is not None else None,
        'transmission': str(transmission) if transmission is not None else None,
        'brand': str(brand) if brand is not None else None,
        'owner': float(owner_num) if owner_num is not None else np.nan,
        'engine': float(engine) if engine is not None else np.nan,
        'power': float(power) if power is not None else np.nan,
        'mileage': float(mileage) if mileage is not None else np.nan,
        'max_power': float(power) if power is not None else np.nan,  # Note: power and max_power are the same
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in sample_data.columns:
        if pd.isna(sample_data.at[0, col]) or sample_data.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction using the A2 model package approach
    try:
        # Debug information for troubleshooting
        print("A2 Input data shape:", sample_data.shape)
        print("A2 Input data columns:", list(sample_data.columns))
        print("A2 Input data values:", sample_data.iloc[0].to_dict())
        print("A2 Imputed fields:", imputed_fields)
        
        # Use A2 model if available, otherwise fall back to original model
        if a2_model is not None and a2_preprocessor is not None and a2_features is not None:
            print("A2 Features expected by preprocessor:", a2_features)
            
            # Preprocess the sample data using the A2 preprocessor
            sample_data_transformed = a2_preprocessor.transform(sample_data[a2_features])
            print("A2 Transformed data shape:", sample_data_transformed.shape)
            
            # Make prediction using the A2 model
            try:
                # Try with is_polynomial=False first (as in notebook)
                predicted_log_price = a2_model.predict(sample_data_transformed, is_polynomial=False)
            except TypeError:
                try:
                    # Try with is_polynomial=True
                    predicted_log_price = a2_model.predict(sample_data_transformed, is_polynomial=True)
                except TypeError:
                    # Try without parameter
                    predicted_log_price = a2_model.predict(sample_data_transformed)
            
            predicted_price = np.expm1(predicted_log_price)  # Inverse of log1p transformation
            price = float(predicted_price[0])
            
            print(f"A2 Predicted log price: {predicted_log_price[0]:.4f}")
            print(f"A2 Predicted price: {price:.2f}")
        else:
            # Fallback to original model approach
            print("A2 Model package not available, using original model")
            pred_log = float(model.predict(sample_data)[0])
            price = float(np.exp(pred_log))
            
            print(f"A2 Fallback predicted log price: {pred_log:.4f}")
            print(f"A2 Fallback predicted price: {price:.2f}")
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Estimated Price: {price:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '32px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "power": "Max Power → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by A2 Preprocessor", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the A2 model's preprocessor imputation strategy:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → A2 preprocessor default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': '3px solid #27ae60'
        })
        
    except Exception as e:
        print(f"A2 Prediction error: {e}")
        print(f"A2 Model type: {type(a2_model) if a2_model is not None else 'None'}")
        print(f"A2 Preprocessor type: {type(a2_preprocessor) if a2_preprocessor is not None else 'None'}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H3("A2 Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Handle A3 price classification when button is clicked
@app.callback(
    Output("a3-result-section", "children"),
    Input("a3-predict", "n_clicks"),
    State("a3-year", "value"),
    State("a3-km", "value"),
    State("a3-fuel", "value"),
    State("a3-transmission", "value"),
    State("a3-owner", "value"),
    State("a3-mileage", "value"),
    State("a3-engine", "value"),
    State("a3-power", "value"),
    State("a3-brand", "value"),
)
def a3_predict_price_class(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price Class' to get your classification.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for A3 model prediction
    sample_data = pd.DataFrame([{
        'year': float(year) if year is not None else np.nan,
        'km_driven': float(km) if km is not None else np.nan,
        'fuel': str(fuel) if fuel is not None else None,
        'transmission': str(transmission) if transmission is not None else None,
        'brand': str(brand) if brand is not None else None,
        'owner': float(owner_num) if owner_num is not None else np.nan,
        'engine': float(engine) if engine is not None else np.nan,
        'max_power': float(power) if power is not None else np.nan,
        'mileage': float(mileage) if mileage is not None else np.nan,
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in sample_data.columns:
        if pd.isna(sample_data.at[0, col]) or sample_data.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction using the A3 model
    try:
        # Debug information
        print("A3 Input data shape:", sample_data.shape)
        print("A3 Input data columns:", list(sample_data.columns))
        print("A3 Input data values:", sample_data.iloc[0].to_dict())
        print("A3 Imputed fields:", imputed_fields)
        
        if a3_model is not None and a2_preprocessor is not None and a2_features is not None:
            # Use A2 features and preprocessor for A3 model (same preprocessing pipeline)
            print("Using A2 preprocessor and features for A3 model")
            
            # Preprocess the sample data using the A2 preprocessor (same as A3)
            sample_data_transformed = a2_preprocessor.transform(sample_data[a2_features])
            print("A3 Transformed data shape:", sample_data_transformed.shape)
            
            # Make prediction using A3 model from MLflow
            predicted_class = a3_model.predict(sample_data_transformed)
            predicted_class = int(predicted_class[0])
            
            print(f"A3 Predicted class: {predicted_class}")
            
            # Map class to description
            class_descriptions = {
                0: "Budget Cars (Lowest 25%)",
                1: "Economy Cars (25%-50%)",
                2: "Mid-range Cars (50%-75%)",
                3: "Premium Cars (Top 25%)"
            }
            
            class_colors = {
                0: "#e74c3c",  # Red
                1: "#f39c12",  # Orange
                2: "#3498db",  # Blue
                3: "#27ae60"   # Green
            }
            
            class_desc = class_descriptions.get(predicted_class, "Unknown")
            class_color = class_colors.get(predicted_class, "#555")
            
        else:
            # Fallback when A2 preprocessor is not available
            if a3_model is None:
                error_msg = "A3 model could not be loaded from MLflow."
                details = "Please check MLflow connection and model availability."
            else:
                error_msg = "A2 preprocessor not available for A3 model."
                details = "A3 model requires the same preprocessor as A2. Please ensure A2 model is properly loaded."
                
            return html.Div([
                html.H3("A3 Model Not Available", style={'color': '#e74c3c', 'textAlign': 'center'}),
                html.P(error_msg, style={'color': '#7f8c8d', 'textAlign': 'center'}),
                html.P(details, style={'color': '#7f8c8d', 'textAlign': 'center'})
            ], style={
                'backgroundColor': '#fdf2f2', 
                'padding': '20px', 
                'borderRadius': '5px',
                'border': '2px solid #e74c3c',
                'marginTop': '20px'
            })
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Predicted Price Class: {predicted_class}", 
                   style={'textAlign': 'center', 'color': class_color, 'fontSize': '32px', 'marginBottom': '10px'}),
            html.H3(class_desc, 
                   style={'textAlign': 'center', 'color': class_color, 'fontSize': '24px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by A3 Preprocessor", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the A3 model's preprocessor:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → A3 preprocessor default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Classification based on quartiles of car price data.", 
                      style={'color': '#7f8c8d', 'textAlign': 'center', 'fontSize': '14px'}),
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': f'3px solid {class_color}'
        })
        
    except Exception as e:
        print(f"A3 Prediction error: {e}")
        print(f"A3 Model type: {type(a3_model) if a3_model is not None else 'None'}")
        print(f"A2 Preprocessor type (used for A3): {type(a2_preprocessor) if a2_preprocessor is not None else 'None'}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H3("A3 Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Start the application
if __name__ == "__main__":
    # Get configuration from environment variables
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    print(f"Starting Car Price Prediction App on port {port}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
