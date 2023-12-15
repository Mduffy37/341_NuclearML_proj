from sklearn.neural_network import MLPRegressor
from utils.model_creation import AutoNN


class CustomMLPRegressor:
    def __init__(self, dataset_key):
        self.dataset_key = dataset_key

    @staticmethod
    def create_model(**kwargs):
        return MLPRegressor(hidden_layer_sizes=kwargs.get('layer_sizes', [25]),
                            activation=kwargs.get('activation', 'logistic'),
                            batch_size=100,
                            learning_rate='constant',
                            learning_rate_init=kwargs.get('learning_rate', 0.001),
                            early_stopping=True,
                            tol=1e-6,
                            verbose=True,
                            n_iter_no_change=50,
                            max_iter=kwargs.get('max_iter', 50000),
                            )

    # LEARNING RATE TESTING

    def learning_rate_init_test(self, learning_rates, layer_size=25):
        if isinstance(layer_size, list):
            layer_label = f"{layer_size[0]}_x{len(layer_size)}"
        else:
            layer_label = layer_size
        for learning_rate in learning_rates:
            model = self.create_model(layer_sizes=layer_size, learning_rate=learning_rate)
            obj = AutoNN(model, f"Models/neural_networks/learning_rate_testing",
                         f"MLPRegressor_{layer_label}_lr{learning_rate}_{self.dataset_key}", self.dataset_key)
            print(f"Completed {obj.dir_name} -> Training Time: {obj.training_time}")

    # ACTIVATION FUNC TESTING

    def activation_func_test(self, activations, layer_size=25):
        if isinstance(layer_size, list):
            layer_label = f"{layer_size[0]}_x{len(layer_size)}"
        else:
            layer_label = layer_size
        for activation in activations:
            model = self.create_model((layer_size,), activation=activation)
            obj = AutoNN(model, f"Models/neural_networks/activation_testing",
                         f"MLPRegressor_{layer_label}_{activation}_{self.dataset_key}", self.dataset_key)
            print(f"Completed {obj.dir_name} -> Training Time: {obj.training_time}")

    # DEPTH TESTING

    def inc_uniform_layer_depth_unit(self, layer_size, depth):
        parent_dir = f"Models/neural_networks/layer_testing/uniform/size_{layer_size}"
        for i in range(depth):
            layers = (tuple([layer_size] * (i + 1)))
            model = self.create_model(layers)
            obj = AutoNN(model, parent_dir,
                         f"MLPRegressor_{layer_size}_x{len(layers)}_{self.dataset_key}", self.dataset_key)
            print(f"Completed {obj.dir_name} -> Training Time: {obj.training_time}")

    def inc_layer_size_inc_depth_unit(self, increment, depth, reverse=False):
        for i in range(depth):
            if reverse:
                layers = tuple(reversed([increment * (j + 1) for j in range(i + 1)]))
                model = self.create_model(layers)
                parent_dir = f"Models/neural_networks/layer_testing/decremental/-{increment}"
                obj = AutoNN(model, parent_dir,
                             f"MLPRegressor_-{increment}_x{len(layers)}_{self.dataset_key}", self.dataset_key)
            else:
                layers = tuple([increment * (j + 1) for j in range(i + 1)])
                model = self.create_model(layers)
                parent_dir = f"Models/neural_networks/layer_testing/incremental/+{increment}"
                obj = AutoNN(model, parent_dir,
                             f"MLPRegressor_+{increment}_x{len(layers)}_{self.dataset_key}", self.dataset_key)
            print(f"Completed {obj.dir_name} -> Training Time: {obj.training_time}")
