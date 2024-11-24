import os
import zipfile
from abc import ABC, abstractmethod 
import logging
 
# Abstract Base Class for Data Ingestor
# ------------------------------------
# This class defines a common interface for different data ingestion strategies.
# Subclasses must implement the ingest method.
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str, extract_to: str) -> None:
        """
        Abstract method to ingest data from a given file.

        Parameters:
        file_path (str): The path to the file to ingest.
        extract_to (str): The directory to extract or save data to. 
        """
        pass



# Concrete Class for ZIP Data Ingestion
# --------------------------------------
# This class implements the DataIngestor interface for handling .zip files. 
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str, extract_to: str) -> None:
        """
        Extracts CSV files from a .zip archive and saves them to the specified directory.

        Parameters:
        file_path (str): The path to the .zip file.
        extract_to (str): The directory where extracted CSV files will be saved.

        Raises:
        ValueError: If the provided file is not a .zip file.
        """
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Create the specified directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # Extract only CSV files
            for file_info in zip_ref.infolist(): 
                if file_info.filename.endswith(".csv"): 
                    file_name = os.path.basename(file_info.filename) 
                    target_path = os.path.join(extract_to, file_name)
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        logging.info(f"CSV files extracted to /data/artifacts.")


# Factory Class to Create Data Ingestors
# -------------------------------------
# This factory class creates the appropriate DataIngestor based on the file extension.
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on the file extension.

        Parameters:
        file_extension (str): The extension of the file to be ingested.

        Returns:
        DataIngestor: An instance of the appropriate DataIngestor class. 
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")
 
# Example usage:
if __name__ == "__main__":
    # # Specify the file path to the .zip and the folder to extract to
    # file_path = "/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/archive.zip"
    # extract_to = "/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/extracted_files"  # Target extraction folder

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and extract the CSV files to the specified folder
    # data_ingestor.ingest(file_path, extract_to)

    # # Now you can read the individual CSV files manually from the specified directory if needed
    # for csv_file in os.listdir(extract_to):
    #     if csv_file.endswith(".csv"):
    #         df = pd.read_csv(os.path.join(extract_to, csv_file))
    #         print(f"Data from {csv_file}:")
    #         print(df.head())  
    pass