from iggy_metaflow_base import IggyFlow, LoadedDataset
from metaflow import FlowSpec, step
import matplotlib.pyplot as plt


class IggyBaselineFlowMLP(FlowSpec, IggyFlow):
    @step
    def start(self):
        # Load Data
        self.file_prefix = "baseline_mlp"
        data, scaled_features = self.load_data(drop_cols=True)
        self.dataset = LoadedDataset(data, scaled_features)
        self.next(self.feature_selection)

    @step
    def feature_selection(self):
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self.dataset[0]

        # Feature Selection
        (X_train, X_val, X_test), selected_features = self.select_features(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        self.selected_features = selected_features
        self.dataset = LoadedDataset(
            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
            ),
            self.dataset[1],
        )
        self.next(self.train_model)

    @step
    def train_model(self):
        import pandas as pd
        from metaflow import S3

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self.dataset.data
        scaled_features = self.dataset.scaled_features
        # Train model
        # model = self.train(X_train, y_train, X_val, y_val)
        self.model = self.train_mlp(X_train, y_train, X_val, y_val)

        # Eval Model
        mean, std = scaled_features[self.label_col]
        self.eval_result = self.eval(self.model, X_test, y_test, mean, std)
        print(f"Test result: {self.eval_result}")

        # Observed vs Predicted values plot
        self.regress_obs_vs_pred(
            self.model, X_test, y_test,
            f"op/{self.file_prefix}/op_scaled_transformed.png",
            False
        )
        self.regress_obs_vs_pred(
            self.model, X_test, y_test,
            f"op/{self.file_prefix}/op_unscaled_untransformed.png",
            False, mean, std
        )

        # Cache stats
        stats = pd.DataFrame(self.eval_result, index=[0])
        stats.to_csv(f"op/{self.file_prefix}/stats.csv", index=False)

        # Visualize loss
        # print(self.model.score(X_train, y_train))
        plt.plot(self.model.loss_curve_)
        try:
            plt.plot(self.model.validation_scores_)
        except:
            print("Validation scores not available...")
        plt.savefig(f"op/{self.file_prefix}/loss.png")

        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")


if __name__ == "__main__":
    IggyBaselineFlowMLP()
