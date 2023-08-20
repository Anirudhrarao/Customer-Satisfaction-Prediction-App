import logging
import pandas as pd 



class IngestData:
    """Ingests data from a CSV file.
    Args:
        data_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The data from the CSV file.
    """
    def __init__(self,data_path: str) -> pd.DataFrame:
        """Initializes the class.
        Args:
            data_path (str): The path to the CSV file.
        Returns:
            None.
        """
        self.data_path = data_path
    
    def get_data(self):
        """Ingests data from the CSV file.
        Returns:
            pd.DataFrame: The data from the CSV file.
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)


def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.info(f"Error while ingesting data: {e}")
        raise e


