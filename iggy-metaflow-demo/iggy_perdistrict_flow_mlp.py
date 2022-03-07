from iggy_metaflow_base import IggyFlow, LoadedDataset
from metaflow import FlowSpec, step, catch
import matplotlib.pyplot as plt
import pandas as pd


class IggyPerDistrictFlowMLP(FlowSpec, IggyFlow):
    @step
    def start(self):
        import pandas as pd
        from metaflow import S3

        # Load Data
        self.file_prefix = "perdistrict_mlp"
        data, scaled_features = self.load_data(drop_cols=False)
        self.dataset = LoadedDataset(data, scaled_features)
        self.next(self.enrich)

    @step
    def enrich(self):
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self.dataset.data

        # Run Iggy Feature Enrichment
        X_train, X_val, X_test = self.iggy_enrich(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        self.dataset = LoadedDataset(
            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
            ),
            self.dataset.scaled_features,
        )
        self.next(self.segment)

    @step
    def segment(self):
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self.dataset.data
        # Extract features according to districts
        tax_col = "current_tax_district_dscr_"
        self.train_data = self.segment_df(X_train, y_train, tax_col)
        self.val_data = self.segment_df(X_val, y_val, tax_col)
        self.test_data = self.segment_df(X_test, y_test, tax_col)
        self.keep_districts = [
            k for k, v in self.train_data.items() if v[0].shape[0] >= 850
        ]
        self.scaled_features = self.dataset.scaled_features
        self.next(self.feature_selection_and_train_model, foreach="keep_districts")

    @catch(var="exception")
    @step
    def feature_selection_and_train_model(self):
        # Train Many Models

        self.exception = None
        tax_dst, self.tax_district = self.input, self.input

        X_train, y_train = self.train_data[tax_dst]
        X_val, y_val = self.val_data[tax_dst]
        X_test, y_test = self.test_data[tax_dst]

        # feature selection
        (X_train, X_val, X_test), selected_features = self.select_features(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        # train model
        self.model = self.train_mlp(X_train, y_train, X_val, y_val)

        # eval
        mean, std = self.scaled_features[self.label_col]
        self.eval_result = self.eval(self.model, X_test, y_test, mean, std)
        print(f"** Results for tax district {tax_dst} **")
        print(self.eval_result)

        # Observed vs Predicted values plot
        self.regress_obs_vs_pred(
            self.model, X_test, y_test,
            f"op/{self.file_prefix}/{tax_dst}/op_scaled_transformed.png",
            False
        )
        self.regress_obs_vs_pred(
            self.model, X_test, y_test,
            f"op/{self.file_prefix}/{tax_dst}/op_unscaled_untransformed.png",
            False, mean, std
        )

        # Cache stats
        stats = pd.DataFrame(self.eval_result, index=[0])
        stats.to_csv(f"op/{self.file_prefix}/{tax_dst}/stats.csv", index=False)

        # Visualize loss
        # print(self.model.score(X_train, y_train))
        plt.plot(self.model.loss_curve_)
        try:
            plt.plot(self.model.validation_scores_)
        except:
            print("Validation scores not available...")
        plt.savefig(f"op/{self.file_prefix}/{tax_dst}/loss.png")

        self.next(self.join)

    @step
    def join(self, inputs):
        import pandas as pd
        from metaflow import S3
        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")


if __name__ == "__main__":
    IggyPerDistrictFlowMLP()
