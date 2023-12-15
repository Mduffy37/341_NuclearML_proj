import pickle
import os
import json
import pandas as pd


def separate_uniform_models(df):
    """
    This function separates a DataFrame into multiple DataFrames based on the 'hidden_layer_1' column.
    Each unique value in the 'hidden_layer_1' column forms a separate DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame to be separated.

    Returns:
    list: A list of DataFrames, each containing rows with the same 'hidden_layer_1' value.
    """
    # Initialize an empty dictionary to store the separated DataFrames
    dfs = {}

    # Iterate over each row in the input DataFrame
    for index, row in df.iterrows():
        # Get the value in the 'hidden_layer_1' column
        layer_size = row['hidden_layer_1']

        # If this value is not already a key in the dictionary, add it with an empty DataFrame as the value
        if layer_size not in dfs:
            dfs[layer_size] = pd.DataFrame()

        # Add the current row to the DataFrame corresponding to its 'hidden_layer_1' value
        dfs[layer_size] = pd.concat([dfs[layer_size], row.to_frame().transpose()], ignore_index=True)

    # Sort the keys in the dictionary
    sorted_keys = sorted(dfs.keys())

    # Return a list of DataFrames in the order of the sorted keys
    return [dfs[key] for key in sorted_keys]


def load_model(filepath):
    """
    This function loads a pickled model from a file.

    Parameters:
    filepath (str): The path to the file containing the pickled model.

    Returns:
    The loaded model.
    """
    # Open the file in binary mode
    with open(filepath, 'rb') as file:
        # Load the model from the file using pickle
        model = pickle.load(file)

    return model


def amend_json_vals(path=None):
    """
    This function traverses a directory structure, finds all JSON files, and modifies them according to specific rules.
    It is a recursive function that calls itself when it encounters a subdirectory.

    Parameters:
    path (str): The directory path to start the traversal from. If None, it defaults to the current directory.

    Returns:
    None
    """

    # Iterate over each item in the directory
    for item in sorted(os.listdir(path)):
        # Construct the full path to the item
        item_path = os.path.join(path, item)

        # If the item is a directory, call this function recursively with the new directory as the path
        if os.path.isdir(item_path):
            amend_json_vals(path=item_path)

        # If the item is a JSON file, open the file and load the JSON data into a Python dictionary
        elif item.endswith('data.json'):
            with open(item_path, 'r') as file:
                data = json.load(file)

                # Create a new dictionary to store the modified model properties
                new_model_properties = {}

                # If the 'model_properties' field contains a 'model_name' field, replace it with a 'model_type' field
                # with the same value
                if 'model_name' in data['model_properties']:
                    new_model_properties['model_type'] = data['model_properties']['model_name']
                    del data['model_properties']['model_name']

                # Set the 'model_path' field to the path of a file named 'model.pkl' in the current directory
                new_model_properties['model_path'] = f"{path}/model.pkl"

                # Set the 'model_id' field to the base name of the current directory
                new_model_properties['model_id'] = os.path.basename(path)

                # Update the 'model_properties' field in the data dictionary with the new model properties
                data['model_properties'].update(new_model_properties)

                # Write the modified data back to the JSON file, overwriting the original file
                with open(item_path, 'w') as output_file:
                    json.dump(data, output_file, indent=4)

        # If the item is neither a directory nor a JSON file, ignore it and move on to the next item
        else:
            pass


class BaseCSVCompiler:
    """
    BaseCSVCompiler is a base class that provides methods for compiling data from JSON files into CSV files.
    It is designed to be subclassed, with subclasses providing a specific implementation of the `filter_data` method.
    """

    filepath = None  # The default directory path for the CSV files
    cvs_name = None  # The default name for the CSV files

    @classmethod
    def filter_data(cls, data):
        """
        Method to filter the data from the JSON files. This method should be implemented by subclasses.

        Parameters:
        data (dict): The data from the JSON file.

        Returns:
        dict: The filtered data.
        """
        raise NotImplementedError

    @classmethod
    def auto_compile_unit_results_into_csv(cls, path=None):
        """
        Method to automatically compile unit results from JSON files into a CSV file.

        Parameters:
        path (str): The directory path to start the compilation from. If None, it defaults to `cls.filepath`.

        Returns:
        None
            """
        # If the path is not provided, use the default filepath
        if path is None:
            path = cls.filepath

        # Iterate over each item in the directory
        for item in sorted(os.listdir(path)):
            # Construct the full path to the item
            item_path = os.path.join(path, item)

            # If the item is a directory, recursively call this function
            if os.path.isdir(item_path):
                cls.auto_compile_unit_results_into_csv(path=item_path)

            # If the item is a JSON file, compile unit results into a CSV file
            elif item.endswith('data.json'):
                # Get the parent directory of the current path
                parent = os.path.dirname(path)
                # Call the method to compile unit results into a CSV file
                cls.compile_unit_results_into_csv(parent)

            # If the item is neither a directory nor a JSON file, ignore it
            else:
                pass

    @classmethod
    def compile_unit_results_into_csv(cls, filepath, csv_name=None):
        """
        Method to compile unit results from JSON files in a specific directory into a CSV file.

        Parameters:
        filepath (str): The directory path where the JSON files are located.
        csv_name (str): The name of the CSV file. If None, it defaults to `cls.cvs_name`.

        Returns:
        None
        """
        # If the CSV name is not provided, use the default CSV name
        if csv_name is None:
            csv_name = cls.cvs_name

        # Initialize an empty list to store the data from each JSON file
        compiled_data = []

        # Iterate over each directory in the specified filepath
        for dirct in sorted(os.listdir(filepath)):
            # If the item is not a directory, skip it
            if not os.path.isdir(os.path.join(filepath, dirct)):
                continue

            # Construct the full path to the JSON file in the directory
            data_path = os.path.join(filepath, dirct, "data.json")

            # Open the JSON file and load the data into a Python dictionary
            with open(data_path, 'r') as file:
                data = json.load(file)

            # Append the data to the list
            compiled_data.append(data)

        # Initialize an empty list to store the filtered data from each JSON file
        rows = []

        # Iterate over each data in the compiled data
        for data in compiled_data:
            # Filter the data using the filter_data method and append it to the list
            rows.append(cls.filter_data(data))

        # Convert the list of filtered data into a DataFrame
        df = pd.DataFrame(rows)

        # Write the DataFrame to a CSV file in the specified filepath
        df.to_csv(f"{filepath}/{csv_name}", index=False)

    @classmethod
    def del_csv(cls, path=None, csv_name=None):
        """
        Method to delete a specific CSV file from a directory and its subdirectories.

        Parameters:
        path (str): The directory path to start the deletion from. If None, it defaults to `cls.filepath`.
        csv_name (str): The name of the CSV file to delete. If None, it defaults to `cls.cvs_name`.

        Returns:
        None
        """
        # If the path is not provided, use the default filepath
        if path is None:
            path = cls.filepath

        # If the CSV name is not provided, use the default CSV name
        if csv_name is None:
            csv_name = cls.cvs_name

        # Iterate over each item in the directory
        for item in sorted(os.listdir(path)):
            # Construct the full path to the item
            item_path = os.path.join(path, item)

            # If the item is a directory, recursively call this function
            if os.path.isdir(item_path):
                cls.del_csv(path=item_path)

            # If the item is a CSV file, delete it
            elif item.endswith(csv_name):
                os.remove(item_path)

            # If the item is neither a directory nor a CSV file, ignore it
            else:
                pass

    @classmethod
    def auto_compile_all_results_into_csv(cls, path=None, csv_name=None):
        """
        Method to automatically compile all results from csv files below it into a CSV file for the directory.

        Parameters:
        path (str): The directory path to start the compilation from. If None, it defaults to `cls.filepath`.
        csv_name (str): The name of the CSV file. If None, it defaults to `cls.cvs_name`.

        Returns:
        bool: True if all files have a combined CSV file, False otherwise.
        """
        # If the path is not provided, use the default filepath and delete the existing CSV file
        if path is None:
            path = cls.filepath
            cls.del_csv()

        # If the CSV name is not provided, use the default CSV name
        if csv_name is None:
            csv_name = cls.cvs_name

        # Compile unit results into CSV files for all subdirectories
        cls.auto_compile_unit_results_into_csv()

        # Initialize a flag to check if all files have a combined CSV file
        all_files_have_combined_csv = True

        # Iterate over each item in the directory
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)

            # If the item is a directory, recursively call this function
            if os.path.isdir(item_path):
                all_files_have_combined_csv &= cls.auto_compile_all_results_into_csv(path=item_path)

            # If the item is a CSV file, check if it exists
            elif item.endswith(csv_name):
                csv = os.path.join(path, csv_name)
                all_files_have_combined_csv &= os.path.exists(csv)

            # If the item is neither a directory nor a CSV file, ignore it
            else:
                pass

        # If all files have a combined CSV file, compile all results into a single CSV file
        if all_files_have_combined_csv:
            cls.compile_all_results_into_csv(path)

        # Return the flag indicating if all files have a combined CSV file
        return all_files_have_combined_csv

    @classmethod
    def compile_all_results_into_csv(cls, filepath, csv_name=None):
        """
        Method to compile all results from CSV files in a specific directory into a single CSV file.

        Parameters:
        filepath (str): The directory path where the CSV files are located.
        csv_name (str): The name of the resulting CSV file. If None, it defaults to `cls.cvs_name`.

        Returns:
        None
        """
        # If the CSV name is not provided, use the default CSV name
        if csv_name is None:
            csv_name = cls.cvs_name

        # Initialize an empty list to store the data from each CSV file
        compiled_data = []

        # Iterate over each directory in the specified filepath
        for dirct in sorted(os.listdir(filepath)):
            # Construct the full path to the directory
            dir_path = os.path.join(filepath, dirct)

            # If the item is not a directory, skip it
            if not os.path.isdir(dir_path):
                continue

            # Construct the full path to the CSV file in the directory
            csv = os.path.join(dir_path, csv_name)

            # If the CSV file exists, read it into a DataFrame and append it to the list
            if os.path.exists(csv):
                df = pd.read_csv(csv)
                compiled_data.append(df)

        # If no CSV files were found, return None
        if len(compiled_data) == 0:
            return

        # Concatenate all the DataFrames in the list into a single DataFrame
        df = pd.concat(compiled_data)

        # Write the DataFrame to a CSV file in the specified filepath
        df.to_csv(f"{filepath}/{csv_name}", index=False)


class NNResultsCSVCompiler(BaseCSVCompiler):
    """
    NNResultsCSVCompiler is a subclass of BaseCSVCompiler provides specific implementation of the `filter_data` method.
    This class is used to compile data from JSON files into CSV files for neural network models.
    """

    # The directory path for the CSV files
    filepath = 'Models/neural_networks'
    # The name for the CSV files
    cvs_name = 'compiled_nn_results.csv'

    @classmethod
    def filter_data(cls, data):
        """
        Method to filter the data from the JSON files for neural network models.
        This method extracts specific fields from the data and formats them into a dictionary
        that can be converted into a row in a CSV file.

        Parameters:
        data (dict): The data from the JSON file.

        Returns:
        dict: The filtered data.
        """
        # Extract the model properties from the data
        model_properties = data['model_properties']
        # Extract specific fields from the model properties
        model_type = model_properties['model_type']
        model_id = model_properties['model_id']
        model_path = model_properties['model_path']
        hidden_layer_sizes = model_properties['model_hyperparameters']['hidden_layer_sizes'] + [0] * (
                9 - len(model_properties['model_hyperparameters']['hidden_layer_sizes']))
        layer_depth = len(model_properties['model_hyperparameters']['hidden_layer_sizes'])

        # Extract the training properties from the data
        training_properties = {k: v for k, v in data['training_properties'].items() if isinstance(v, (float, int))}
        # Extract the performance metrics from the data
        performance_metrics = data['performance_metrics']

        # Create a dictionary to store the row data
        row_data = {'time_created': data['training_properties']['time_created'], 'model_type': model_type,
                    'model_id': model_id, 'model_path': model_path,
                    'dataset': data['training_properties']['dataset'], 'layer_depth': layer_depth}
        # Add the hidden layer sizes to the row data
        for i in range(9):
            row_data[f'hidden_layer_{i + 1}'] = hidden_layer_sizes[i]

        # Add the training properties and performance metrics to the row data
        row_data.update(training_properties)
        row_data.update(performance_metrics)

        # Return the row data
        return row_data
