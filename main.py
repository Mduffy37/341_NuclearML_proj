from utils.data.database import Database
from utils.data.data_manager import NNResultsCSVCompiler
from utils.data.plotter_classes import ModelPlotter, ResultsPlotter, MultiResultsPlotter
from utils.model_creation import CustomMLPRegressor, AutoNN
import pandas as pd
import matplotlib.pyplot as plt

# todo: add error handling for finding no models in the db

if __name__ == '__main__':

    compiler = NNResultsCSVCompiler()
    compiler.auto_compile_all_results_into_csv()

    db = Database()
    db.update_database('neural_networks')





