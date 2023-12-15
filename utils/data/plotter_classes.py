import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
import os


class ModelPlotter:
    def __init__(self, model):
        self.model = model

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.model.X_train, self.model.y_train,
                                                                cv=5)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        plt.plot(train_sizes, train_scores_mean, label="Training score")
        plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)

        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def plot_pred_vs_actual(self):
        fig, ax = plt.subplots()
        ax.set_title(f"Predictions vs Actual - {self.model.model_name}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.scatter(self.model.y_test, self.model.predictions, s=1)
        p1 = max(max(self.model.predictions), max(self.model.y_test))
        p2 = min(min(self.model.predictions), min(self.model.y_test))
        ax.plot([p1, p2], [p1, p2], 'b-')
        ax.axis("equal")
        ax.grid()
        return fig, "pred_vs_actual"

    def plot_residuals_histogram(self):
        fig, ax = plt.subplots()
        ax.set_title(f"Residuals Histogram - {self.model.model_name}")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-0.5, 0.5)
        ax.hist(self.model.y_test - self.model.predictions, bins=100)
        ax.grid()
        return fig, "resid_hist"

    import numpy as np

    def plot_residuals_vs_predictions(self):
        fig, ax = plt.subplots()
        ax.set_title(f"Residuals vs Predictions - {self.model.model_name}")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Residuals")

        residuals = self.model.y_test - self.model.predictions
        ax.scatter(self.model.predictions, residuals, s=1)
        ax.plot(ax.get_xlim(), [0, 0], 'k--', lw=1)
        ax.grid()
        return fig, "resid_vs_predict"


class MultiResultsPlotter:
    def __init__(self, dataframes, name=None, directory=None, id_to_label=None, remove_ds=True):
        if name is not None:
            self.name = name
        else:
            raise ValueError("Must provide a name for the results")

        if directory is None:
            self.directory = 'Figures'
        else:
            self.directory = directory

        self.id_to_label = id_to_label
        self.remove_ds = remove_ds

        self.dfs = dataframes
        self.columns = self.dfs[0].columns.tolist()

        self.modify_model_id()
        self.zero_df = self.join_dfs_with_zeros()

    def modify_model_id(self):
        for idx, df in enumerate(self.dfs):
            for idx, row in df.iterrows():
                if self.remove_ds:
                    end = "_" + row['model_id'].split('_')[-1]
                    df.at[idx, 'model_id'] = row['model_id'][:-(len(end))]

                if self.id_to_label is not None:
                    split = row['model_id'].split('_')
                    new_id = ""
                    for i in reversed(range(1, self.id_to_label + 1)):
                        if i == 1:
                            new_id += split[-i]
                        else:
                            new_id += split[-i] + "_"
                    df.at[idx, 'model_id'] = new_id

    def assign_dir_name(self, dir_name):
        self.directory = dir_name

    def join_dfs_with_zeros(self):
        new_df = []
        for idx, df in enumerate(self.dfs):
            if idx == 0:
                empty = pd.DataFrame([[0] * len(self.columns)], columns=self.columns)
                empty['model_id'] = f""
                new_df.append(empty)

            df_copy = df.copy()

            zero_df = pd.DataFrame([[0] * len(df_copy.columns.to_list())], columns=df_copy.columns.to_list())
            empty_spaces = "  " * (idx + 1)
            zero_df['model_id'] = f"{empty_spaces}"

            new_df.append(df_copy)
            new_df.append(zero_df)

        return pd.concat(new_df, ignore_index=True)

    def plot_bar(self, y, hue=None, limit=0.95, palette='viridis', save=True, remove_legend=False):
        if y not in self.columns:
            raise ValueError(f"Invalid y value \n Columns are: {str(self.columns)}")

        if hue is not None:
            if hue not in self.columns:
                raise ValueError(f"Invalid hue value \n Columns are: {str(self.columns)}")

        concat_df = self.zero_df.copy()

        sns.set(style="darkgrid")
        palette_colors = sns.color_palette(palette, n_colors=len(concat_df[hue].unique()))
        g = sns.catplot(x="model_id", y=y, data=concat_df, kind="bar", height=6, aspect=2, hue=hue, palette=palette_colors)
        g.set_xticklabels(rotation=90)
        plt.tight_layout()
        if remove_legend:
            g._legend.remove()

        if y == 'r2':
            if limit > 1:
                lim = concat_df[concat_df[y] != 0][y].min()
                lim = lim * 0.99
            else:
                lim = concat_df[concat_df[y] != 0][y].quantile(1 - limit)
            g.set(ylim=(lim, 1))
        else:
            if limit > 1:
                limit = 1
            lim = concat_df[concat_df[y] != 0][y].quantile(limit)
            lim = concat_df[concat_df[y] < lim * 1.5][y].quantile(limit)
            g.set(ylim=(0, lim * 1.1))

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        g.fig.suptitle(f"{self.name} - {y}")
        if save:
            plt.savefig(f"{self.directory}/{self.name}_bar_{y}.png")
        else:
            plt.show()

    def plot_generic_x_y(self, x, y, kind):
        if x not in self.columns:
            raise ValueError(f"Invalid x value \n Columns are: {str(self.columns)}")
        if y not in self.columns:
            raise ValueError(f"Invalid y value \n Columns are: {str(self.columns)}")

        sns.set(style="darkgrid")
        g = sns.FacetGrid(height=5, data=pd.DataFrame())
        g.map(plt.scatter, x, y, s=1)

        for idx, df in enumerate(self.dfs):
            color = plt.cm.get_cmap('viridis')(idx / len(self.dfs))
            sns.lineplot(x=x, y=y, data=df, ax=g.ax, label=df['model_id'].iloc[0], color=color)

        g.fig.subplots_adjust(top=0.9)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        g.fig.suptitle(f"{self.name} - {y} vs {x}")
        plt.legend()
        plt.savefig(f"{self.directory}/{self.name}_{y}vs{x}.png")



class ResultsPlotter:
    def __init__(self, data, name=None, directory=None):
        if isinstance(data, str) and data.endswith('.csv'):
            self.df = pd.read_csv(data)
            if name is None:
                self.name = data.split('/')[-1].split('.')[0]
            else:
                self.name = name
        elif isinstance(data, pd.DataFrame):
            self.df = data
            if name is not None:
                self.name = name
            else:
                raise ValueError("Must provide a name for the dataset results")
        else:
            raise TypeError("Invalid data type. Must be a filepath to a csv or a pandas DataFrame")

        self.columns = self.df.columns.tolist()

        if directory is None:
            self.directory = 'Figures'
        else:
            self.directory = directory

    def assign_dir_name(self, dir_name):
        self.directory = dir_name

    def plot_bar(self, y):
        if y not in self.columns:
                raise ValueError(f"Invalid y value \n Columns are: {str(self.columns)}")

        sns.set(style="darkgrid")
        g = sns.catplot(x="model_id", y=y, data=self.df, kind="bar", height=5, aspect=2)
        g.set_xticklabels(rotation=90)
        g.set(ylim=(0.95, 1))

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        g.fig.suptitle(f"{self.name} - {y}")
        plt.show()
        plt.savefig(f"{self.directory}/{self.name}bar_{y}.png")

    def plot_generic_x_y(self, x, y_plot, kind):
        if x not in self.columns:
            raise ValueError(f"Invalid x value \n Columns are: {str(self.columns)}")

        if isinstance(y_plot, str):
            if y_plot not in self.columns:
                raise ValueError(f"Invalid y_plot value \n Columns are: {str(self.columns)}")
        elif isinstance(y_plot, list):
            for y in y_plot:
                if y not in self.columns:
                    raise ValueError(f"Invalid y_plot value \n Columns are: {str(self.columns)}")
        else:
            raise ValueError("Invalid y_plot format. It should be a string or a list of strings.")

        sns.set(style="darkgrid")
        g = sns.FacetGrid(height=5, data=pd.DataFrame())

        if isinstance(y_plot, str):
            sns.relplot(x=x, y=y_plot, kind=kind, data=self.df)
        elif len(y_plot) > 1 and isinstance(y_plot, list):
            for idx, y in enumerate(y_plot[0:]):
                color = plt.cm.get_cmap('viridis')(idx / len(y_plot))
                sns.lineplot(x=x, y=y, data=self.df, ax=g.ax, label=y, color=color)

        g.fig.subplots_adjust(top=0.9)
        g.set_ylabels("")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if isinstance(y_plot, str):
            g.fig.suptitle(f"{self.name} - {y_plot} vs {x}")
            plt.savefig(f"{self.directory}/{self.name}_{y_plot}vs{x}.png")
        else:
            g.fig.suptitle(f"{self.name} plots")
            plt.legend()
            plt.savefig(f"{self.directory}/{self.name}_plotsvs{x}.png")

    def create_table(self, columns):
        for column in columns:
            if column not in self.columns:
                raise ValueError(f"Invalid column value ({column}) \n Columns are: {str(self.columns)}")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis('off')
        table = plt.table(cellText=self.df[columns].values, colLabels=columns, loc='center', )
        table.scale(2, 2)
        table.set_fontsize(20)
        plt.show()

    def plot_keff_boxplot(self):
        plt.boxplot(self.df['Keff'], vert=False, whis=10)

        plt.yticks([])
        plt.xlabel('Keff')
        plt.title('Boxplot of Keff for dataset 1')

        q1 = round(self.df['Keff'].quantile(0.25), 3)
        q3 = round(self.df['Keff'].quantile(0.75), 3)
        iqr = round(q3 - q1, 3)

        plt.text(q1, 1.15, f'Q1: {q1}', fontsize=10, horizontalalignment='center', verticalalignment='center')
        plt.text(q3, 1.15, f'Q3: {q3}', fontsize=10, horizontalalignment='center', verticalalignment='center')
        plt.text((q1 + q3) / 2, 0.85, f'IQR: {iqr}', fontsize=10, horizontalalignment='center',
                 verticalalignment='center')

        plt.tight_layout()

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        plt.savefig(f"{self.directory}/{self.name}_boxplot.png")
