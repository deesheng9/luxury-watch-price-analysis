import os
import logging

def setup_environment():
    """
    Utility function to ensure all necessary output directories exist 
    and to configure the project logger.
    """
    # 1. Create directories (added 'logs' folder)
    directories = ['outputs', 'outputs/plots', 'outputs/reports', 'logs']
    for d in directories:
        os.makedirs(d, exist_ok=True)
    
    # 2. Configure the Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/project_execution.log", mode='w'), # Saves to a file
            logging.StreamHandler() # Prints to the VS Code terminal
        ]
    )
    
    logging.info("Utility: Environment setup complete. Output and log directories are ready.")