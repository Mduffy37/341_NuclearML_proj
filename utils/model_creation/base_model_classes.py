from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import time
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    max_error,
    )
import numpy as np
import json
from config import DATASETS
from utils.data.plotter_classes import ModelPlotter


class Dataset:
    def __init__(self, dataset_key):
        self.dataset_key = dataset_key
        self.filepath = DATASETS[dataset_key]
        self.df = pd.read_csv(self.filepath)

        if self.dataset_key == 'ds1' or self.dataset_key == 'ds2':
            self.features = self.df.drop(['Keff', 'Std_dev'], axis=1)

        elif self.dataset_key == 'ds1fe1':
            self.df['vmvf'] = (self.df['Pitch'] * self.df['Pitch'] / (
                        np.pi * self.df['FuelRadius'] * self.df['FuelRadius'])) - 1 - (
                                      0.12 / self.df['FuelRadius']) - (
                                          0.0036 / (self.df['FuelRadius'] * self.df['FuelRadius']))

            self.features = self.df.drop(['Keff', 'Std_dev', 'WaterDensity'], axis=1)

        self.labels = self.df['Keff']

class BaseModel:
    def __init__(self, model, model_name, dataset_key):

        self.dataset = Dataset(dataset_key)
        self.features = self.dataset.features
        self.labels = self.dataset.labels

        self.model = model
        self.model_name = model_name
        self.model_type = model.__class__.__name__
        self.model_hyperparameters = model.get_params()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.labels,
                                                                                test_size=0.2, random_state=9320)
        self.training_features = self.X_train.columns

        self.training_time = self.train()
        self.loss = self.model.loss_
        self.predictions = self.model.predict(self.X_test)

    def train(self):
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        end = time.time()
        training_time = round(end - start, 3)
        return training_time

    def save_model(self, parent_dir, dir_name, data, plots):
        if not os.path.exists(f"{parent_dir}"):
            os.makedirs(f"{parent_dir}", exist_ok=True)
        if not os.path.exists(f"{parent_dir}/{dir_name}"):
            os.mkdir(f"{parent_dir}/{dir_name}")
        if not os.path.exists(f"{parent_dir}/{dir_name}/Figures"):
            os.mkdir(f"{parent_dir}/{dir_name}/Figures")

        txt_filename = f"{parent_dir}/{dir_name}/data.json"
        with open(txt_filename, 'w') as file:
            file.write(json.dumps(data, indent=4))

        pkl_filename = f"{parent_dir}/{dir_name}/model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)

        for data in plots:
            plot, name = data
            plot.savefig(f"{parent_dir}/{dir_name}/Figures/{name}.png")




class AutoNN(BaseModel):
    def __init__(self, model, parent_dir, dir_name, dataset_key):
        super().__init__(model, dir_name, dataset_key)

        self.parent_dir = parent_dir
        self.dir_name = dir_name
        self.dataset_key = dataset_key

        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = np.sqrt(self.mse)

        self.mae = mean_absolute_error(self.y_test, self.predictions)
        self.mae_percentage = mean_absolute_percentage_error(self.y_test, self.predictions) * 100000
        self.max_error = max_error(self.y_test, self.predictions) * 100000

        self.r2 = r2_score(self.y_test, self.predictions)

        data = self.create_data_dict()
        plotter = ModelPlotter(self)
        plots = [plotter.plot_pred_vs_actual(), plotter.plot_residuals_histogram(), plotter.plot_residuals_vs_predictions()]
        self.save_model(parent_dir, dir_name, data, plots)
        self.serialize()

    def create_data_dict(self):
        model_properties = {
                "model_type": self.model_type,
                "model_path": f"{self.parent_dir}/{self.dir_name}/model.pkl",
                'model_id': self.dir_name,
                "model_hyperparameters": self.model_hyperparameters,
            }

        training_properties = {
            "dataset": self.dataset_key,
            "training_features": self.training_features.tolist(),
            "training_time": self.training_time,
            "loss": self.loss,
            "iterations": self.model.n_iter_,
            "time_created": time.strftime("%d/%m/%Y %H:%M:%S"),
        }

        performance_metrics = {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mae_percentage": self.mae_percentage,
            "max_error": self.max_error,
            "r2": self.r2,
        }
        data = {
            "model_properties": model_properties,
            "training_properties": training_properties,
            "performance_metrics": performance_metrics,
        }
        return data

    def serialize(self):
        pkl_filename = f"{self.parent_dir}/{self.dir_name}/model_obj.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self, file)





