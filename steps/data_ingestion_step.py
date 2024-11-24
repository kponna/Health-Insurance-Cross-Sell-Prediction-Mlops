import os
import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step
import logging

@step
def data_ingestion_step(file_path: str, extract_to: str) -> pd.DataFrame:
    """Ingest data from a file using the appropriate DataIngestor and return the DataFrame. 
    Args:
        file_path (str): The file path of the source file (e.g., a ZIP archive) to be ingested.
        extract_to (str): The directory where the file contents should be extracted.

    Returns:
        pd.DataFrame: A DataFrame containing the ingested data from the target CSV file.
        """
    try:
        # Determine the file extension dynamically
        file_extension = os.path.splitext(file_path)[1]

        # Get the appropriate DataIngestor
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

        # Ingest the data (extract CSV files)
        data_ingestor.ingest(file_path, extract_to)
 
        target_csv_file = 'train.csv'
        target_csv_path = os.path.join(extract_to, target_csv_file)
         
        if not os.path.exists(target_csv_path):
            raise FileNotFoundError(f"{target_csv_file} not found in {extract_to}")
         
        df = pd.read_csv(target_csv_path)
        
        return df
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise e