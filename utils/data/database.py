import sqlite3
import pandas as pd
import os


class Database:
    """
    The Database class provides methods for interacting with a SQLite database.
    It includes methods for updating the database with data from a CSV file and retrieving data based on a file path.
    """

    def __init__(self):
        """
        Initializes the Database object. This constructor does not take any parameters.
        """
        pass

    @staticmethod
    def update_database(table):
        """
        Updates the database with data from a CSV file. The table to be updated and the CSV file are determined based on the table parameter.

        Parameters:
        table (str): The name of the table to be updated. Currently, only 'neural_networks' is supported.

        Raises:
        ValueError: If the table parameter is not 'neural_networks'.

        Returns:
        None
        """
        if table == 'neural_networks':
            filepath = 'Models/neural_networks/compiled_nn_results.csv'
        else:
            raise ValueError("Invalid table name")

        conn = sqlite3.connect('Models/database.db')
        cursor = conn.cursor()

        # drop table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()

        df = pd.read_csv(filepath)
        df.to_sql(f'{table}', conn, if_exists='replace', index=False)

        conn.commit()
        conn.close()

    @staticmethod
    def get_by_filepath(filepath):
        """
        Retrieves data from the database based on a file path. The table to retrieve data from is determined based on the file path.

        Parameters:
        filepath (str): The file path to use for retrieving data. The table name is extracted from the file path.

        Returns:
        DataFrame: A DataFrame containing the retrieved data.
        """
        table = filepath.split('/')[1]

        conn = sqlite3.connect('Models/database.db')
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE model_path LIKE '{filepath}%'")
        cols = [description[0] for description in cursor.description]
        result = pd.DataFrame(cursor.fetchall(), columns=cols)
        conn.close()
        return result