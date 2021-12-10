import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def compute_performance(
    name_output_dir="performance_results",
    dataset_dir=DATASET_DIRECTORY,
    dataset_experiment=["adult", "artificial_10", "bank", "compas", "german", "heart"],
    n_times=5,
    min_supports=[
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
    ],
):
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path
    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer

    from import_datasets import discretize

    def estimate_performance(
        X_discretized,
        class_map,
        min_supports,
        dataset_name="",
        out_dir="",
        n_times=5,
        true_class="class",
        predicted_class="predicted",
    ):
        times = {}
        infos = {}
        import time

        print(dataset_name)
        for sup in min_supports:
            print(sup, end=" ")
            times[sup] = {}
            # We repeat the process of frequent pattern (FP) divergence extract n times (n=5 in the paper)
            for i in range(0, n_times):

                # We estimate the time to extract FP divergence
                s_time = time.time()
                fp_diver = FP_DivergenceExplorer(
                    X_discretized,
                    true_class,
                    predicted_class,
                    class_map=class_map,
                    dataset_name=dataset_name,
                )

                info_col = ["support"]
                FP_fm = fp_diver.getFrequentPatternDivergence(min_support=sup)

                times[sup][i] = time.time() - s_time

                # Time to extract frequent pattern divergence
                print(f"- time: {round(time.time()-s_time, 4)}")

                infos[sup] = FP_fm[info_col].describe()

        time_df = pd.DataFrame.from_dict(times).describe()
        for sup in min_supports:
            infos[sup]["time"] = time_df[sup]
            filename = os.path.join(out_dir, f"{dataset_name}_{sup}.csv")
            infos[sup].to_csv(filename)

        return infos

    output_result_dir = os.path.join(os.path.curdir, name_output_dir)

    print(f"Output performance results in directory {output_result_dir}")

    Path(output_result_dir).mkdir(parents=True, exist_ok=True)

    if "adult" in dataset_experiment:
        dataset_name = "adult"

        processed_dataset_dir = os.path.join(dataset_dir, "processed")

        adult_processed_file = os.path.join(
            processed_dataset_dir, "adult_discretized_RF.csv"
        )

        if os.path.isfile(adult_processed_file):
            X_discretized = pd.read_csv(adult_processed_file)
            class_map = {"N": "<=50K", "P": ">50K"}
        else:

            from import_datasets import import_process_adult

            dfI, class_map = import_process_adult(inputDir=dataset_dir)
            dfI.reset_index(drop=True, inplace=True)

            # Train and predict

            # Cross validation
            from import_datasets import train_predict

            (
                X_FP,
                y_FP,
                y_predicted,
                y_predict_prob,
                encoders,
                indexes_FP,
            ) = train_predict(dfI, type_cl="RF", labelEncoding=True, validation="cv")
            attributes = dfI.columns.drop("class")

            X_discretized = discretize(
                dfI,
                attributes=attributes,
                indexes_FP=indexes_FP,
                dataset_name=dataset_name,
            )

            X_discretized["class"] = y_FP["class"]
            X_discretized["predicted"] = y_predicted

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            X_discretized,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )

    if "artificial_10" in dataset_experiment:
        dataset_name = "artificial_10"

        features_3 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        data = np.concatenate([np.full((25000, 10), 0), np.full((25000, 10), 1)])
        np.random.seed(7)
        for i in range(data.shape[1]):
            np.random.shuffle(data[:, i])
        df_artificial_3 = pd.DataFrame(data=data, columns=features_3)
        df_artificial_3["class"] = 0
        indexes = df_artificial_3.loc[
            (
                (df_artificial_3["a"] == df_artificial_3["b"])
                & (df_artificial_3["a"] == df_artificial_3["c"])
            )
        ].index
        df_artificial_3.loc[indexes, "class"] = 1

        import random

        indexes_class1 = list(indexes)
        random.Random(7).shuffle(indexes_class1)
        df_artificial_3["predicted"] = df_artificial_3["class"]
        df_artificial_3.loc[indexes_class1[: int(len(indexes_class1) / 2)], "class"] = 0

        class_map = {"N": 0, "P": 1}

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            df_artificial_3,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )

    if "compas" in dataset_experiment:
        dataset_name = "compas"

        risk_class_type = True

        # Probublica analysis https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm

        from import_datasets import import_process_compas

        dfI, class_map = import_process_compas(risk_class=risk_class_type)
        dfI.reset_index(drop=True, inplace=True)

        dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})

        attributes = dfI.columns.drop(["class", "predicted"])
        X_FP = dfI[attributes]
        y_FP = dfI[["class"]]
        y_predict_prob = None
        y_predicted = np.asarray(dfI["predicted"])

        X_FP = dfI[attributes].copy()
        X_discretized = discretize(
            dfI, attributes=attributes, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            X_discretized,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )

    if "german" in dataset_experiment:

        dataset_name = "german"

        from import_datasets import import_process_german

        dfI, class_map = import_process_german(inputDir=dataset_dir)

        from import_datasets import train_predict

        X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP = train_predict(
            dfI, type_cl="RF", labelEncoding=True, validation="cv"
        )

        attributes = dfI.columns.drop("class")

        X_discretized = discretize(
            dfI, attributes=attributes, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            X_discretized,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )

    if "heart" in dataset_experiment:

        dataset_name = "heart"

        from import_datasets import import_process_heart

        dfI, class_map = import_process_heart(inputDir=dataset_dir)

        from import_datasets import train_predict

        X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP = train_predict(
            dfI, type_cl="RF", labelEncoding=True, validation="cv"
        )
        attributes = dfI.columns.drop("class")

        X_discretized = discretize(
            dfI, attributes=attributes, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            X_discretized,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )

    if "bank" in dataset_experiment:

        dataset_name = "bank"

        from import_datasets import import_process_bank

        dfI, class_map = import_process_bank(inputDir=dataset_dir)

        from import_datasets import train_predict

        X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP = train_predict(
            dfI, type_cl="RF", labelEncoding=True, validation="cv"
        )

        attributes = dfI.columns.drop("class")

        X_discretized = discretize(
            dfI, attributes=attributes, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

        output_dir_dataset = os.path.join(output_result_dir, dataset_name)
        Path(output_dir_dataset).mkdir(parents=True, exist_ok=True)

        info = estimate_performance(
            X_discretized,
            class_map,
            min_supports,
            dataset_name=dataset_name,
            out_dir=output_dir_dataset,
            n_times=n_times,
        )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="performance_results",
        help="specify the name of the output folder",
    )
    parser.add_argument(
        "--dataset_experiment",
        nargs="*",
        type=str,
        default=["adult", "artificial_10", "bank", "compas", "german", "heart"],
        help='specify a list of dataset among ["adult", "artificial_10", "bank", "compas", "german", "heart"]',
    )
    parser.add_argument(
        "--min_supports",
        nargs="*",
        type=float,
        default=[
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
        ],
        help="specify a list of min supports of interest, with values from 0 to 1",
    )
    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="specify the number of times to repeat the estimation, 5 as default",
    )

    args = parser.parse_args()

    compute_performance(
        name_output_dir=args.name_output_dir,
        min_supports=args.min_supports,
        n_times=args.n_times,
        dataset_dir=args.dataset_dir,
        dataset_experiment=args.dataset_experiment,
    )
