import os

def setup_environment():
    """
    Utility function to ensure all necessary output directories exist.
    """
    directories = ['outputs', 'outputs/plots', 'outputs/reports']
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print("Utility: Environment setup complete. Output directories are ready.")