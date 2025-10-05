# Car Price Prediction Web Application 🚗💰

**Student ID:** st126235  
**Student Name:** Dechathon Niamsa-ard
**Assignment:** Machine Learning Assignment 3 - Predicting Car Price (Classification)
**Github Link:** https://github.com/dniamsaard4codework/A3-Predicting-Car-Price.git

**Link to the website:** https://st126235.ml.brain.cs.ait.ac.th/

---

### Introduction

The goal of this assignment is to predict car prices by turning a regression problem into a classification task. The preprocessor from assignment 2 is reused in this assignment. This experiment compares Multinomial Logistic Regression with and without Ridge regularization, and logs the model for deployment with MLFlow. Moreover, I set up CI/CD in GitHub Actions.

---

### Task

- Convert the `selling_price` variable into 4 classes (0–3) based on quantiles.  
- Build a Logistic Regression model from scratch.  
- Implement evaluation metrics (Accuracy, Precision, Recall, F1, Macro, Weighted) manually.  
- Extend the model with Ridge (L2) regularization.  
- Log and compare different experiments using MLflow.  
- Prepare for deployment with CI/CD via GitHub Actions.

---

### Tech Stack

- **Python 3.12**: Core programming language
- **Dash/Plotly**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Model preprocessing utilities
- **MLflow**: Experiment tracking and model management
- **Docker**: Containerization for deployment
- **GitHub Actions**: CI/CD pipeline automation
- **uv**: Fast Python package installer and dependency manager
- **Traefik**: Load balancer and SSL termination

---

### Testing

The project implements comprehensive testing strategies:

#### Unit Tests
- **Model Shape Tests** (`test_model_staging.py`): Verify model input/output shape compatibility
- **Prediction Tests**: Ensure model produces valid predictions within expected ranges
- **Data Validation**: Test preprocessing pipeline and data transformations

#### Integration Tests  
- **App Callback Tests** (`test_app_callbacks.py`): Validate Dash application behavior
- **End-to-End Tests**: Test complete prediction workflow from input to output

#### Testing Framework
- **pytest**: Testing framework with fixtures and parameterized tests
- **Docker Testing**: All tests run inside containerized environment
- **Automated Testing**: Triggered on every push and pull request
- **Test Coverage**: Comprehensive coverage of critical application components

---

### Key Components

#### 1. **Custom Logistic Regression Implementation** (`A3model.py`)
- From-scratch implementation of multinomial logistic regression
- Support for batch, mini-batch, and stochastic gradient descent
- Ridge (L2) regularization implementation
- Manual evaluation metrics calculation

#### 2. **Data Preprocessing Pipeline** (`A2modelandprep.py`)
- Reuses preprocessor from Assignment 2 for consistency
- Feature scaling and encoding for categorical variables
- Price classification into 4 quantile-based categories

#### 3. **Web Application** (`app.py`)
- Interactive Dash application for car price prediction
- Real-time prediction with user input validation
- Multiple model comparison and visualization
- Responsive UI with professional styling

#### 4. **Model Management** (`LoadA3model.py`)
- MLflow integration for model versioning
- Automated model loading with fallback paths
- Model transition from staging to production

#### 5. **CI/CD Pipeline**
- **Build Stage**: Multi-stage Docker builds with optimization
- **Test Stage**: Automated testing with pytest
- **Deploy Stage**: Zero-downtime deployment with health checks

---

### Experimental Results

The project conducted extensive experiments comparing different configurations:

- **Models**: Simple Logistic Regression vs. Ridge Regularized Logistic Regression
- **Optimization Methods**: Batch, Mini-batch, and Stochastic Gradient Descent  
- **Learning Rates**: 0.0001, 0.001, 0.01
- **Best Performance**: Ridge Regression with Mini-batch GD (α=0.001)

**MLflow Experiment Tracking**: [View Results](https://mlflow.ml.brain.cs.ait.ac.th/#/experiments/607305997044080535)

All experiments are logged with metrics, parameters, and model artifacts for reproducibility.

---

### CI/CD Pipeline

#### Continuous Integration
- **GitHub Actions**: Automated workflows on every push/PR
- **Build Testing** (`build-test.yml`): Docker build, test execution, optional deployment
- **Model Staging** (`test-model-staging.yml`): Model validation and shape testing
- **Quality Gates**: Deployment only proceeds if all tests pass

#### Continuous Deployment
- **Multi-Stage Deployment**: 
  - **Development**: Local testing environment
  - **Staging**: Automated deployment for integration testing  
  - **Production**: Traefik-managed deployment with SSL
- **Zero-Downtime Deployment**: Health checks and gradual rollout
- **Docker Hub Integration**: Automated image building and pushing

#### MLOps Integration
- **MLflow Model Registry**: Centralized model management
- **Model Versioning**: Automatic versioning with experiment tracking
- **Model Promotion**: Automated staging to production promotion via `transition.py`
- **Model Monitoring**: Performance tracking and drift detection

#### Security & Infrastructure
- **Multi-stage Docker builds**: Optimized, secure containers
- **Non-root user execution**: Enhanced security posture
- **SSL/TLS encryption**: Traefik-managed certificates
- **Secrets management**: GitHub secrets for sensitive configuration
- **Network isolation**: Containerized environments with controlled access

---

### File Structure

```
├── app/                          # Main application directory
│   ├── A2modelandprep.py        # Preprocessor from Assignment 2
│   ├── A3model.py               # Custom logistic regression implementation
│   ├── app.py                   # Dash web application
│   ├── LoadA3model.py           # Model loading utilities
│   ├── transition.py            # MLflow model transition script
│   └── model/                   # Trained models directory
│       ├── best_model.pkl       # A2 preprocessor model
│       └── car_price.model      # A3 classification model
├── data/                        # Dataset directory
│   └── Cars.csv                # Car price dataset
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest configuration
│   ├── test_app_callbacks.py   # Application callback tests
│   └── test_model_staging.py   # Model validation tests
├── notebook/                    # Jupyter notebooks
│   ├── st126235_Assignment_3.ipynb  # Main analysis notebook
│   └── figures/                 # Experiment visualizations
├── .github/workflows/           # CI/CD pipeline definitions
├── docker-compose.yml          # Local development setup
├── docker-compose-deploy.yml   # Production deployment setup
├── Dockerfile                  # Container definition
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Local Development & Testing 

### Quick Setup
```bash
# Clone repository
git clone https://github.com/dniamsaard4codework/A3-Predicting-Car-Price.git
cd A3-Predicting-Car-Price

# Install dependencies
uv sync  # or pip install -r requirements.txt

# Run application
python app/app.py
```

**Access**: http://localhost:8050

### Testing
```bash
# Run all tests
pytest tests/ -v

# Specific tests
pytest tests/test_model_staging.py -v      # Model tests
pytest tests/test_app_callbacks.py -v     # App tests
```

### Docker (Alternative)
```bash
docker-compose up --build
```

---