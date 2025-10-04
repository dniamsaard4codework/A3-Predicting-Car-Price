"""
Pytest configuration file for handling imports and setup
"""
import sys
import os

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Setup the required classes for pickle compatibility
try:
    from A2modelandprep import LinearRegression, ManualPreprocessor, NoRegularization, LassoPenalty, RidgePenalty
    import __main__
    __main__.LinearRegression = LinearRegression
    __main__.ManualPreprocessor = ManualPreprocessor
    __main__.NoRegularization = NoRegularization
    __main__.LassoPenalty = LassoPenalty
    __main__.RidgePenalty = RidgePenalty
except ImportError:
    pass