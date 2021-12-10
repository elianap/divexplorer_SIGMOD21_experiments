import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def redundancy_experiments(
    name_output_dir="output",
    dataset_dir=DATASET_DIRECTORY,
    show_figures=True,
):

    # Derive Frequent Pattern divergence and compute the number of extracted patterns for a given support and redundancy pruning threshold
    # Varying input supports
    # Varying edundancy pruning threshold

    def compute_redundancy_stats(
        X_discretized,
        class_map,
        min_supports,
        dataset_name="",
        true_class="class",
        predicted_class="predicted",
        th_redundancy=[None, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
    ):
        from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer

        red_info = {}
        for sup in min_supports:
            fp_diver = FP_DivergenceExplorer(
                X_discretized,
                true_class,
                predicted_class,
                class_map=class_map,
                dataset_name=dataset_name,
            )
            metric = "d_fpr"
            FP_fm = fp_diver.getFrequentPatternDivergence(min_support=sup)

            for th_red in th_redundancy:

                th_red_str = "None" if th_red is None else th_red

                if th_red_str not in red_info:
                    red_info[th_red_str] = {}

                from divexplorer.FP_Divergence import FP_Divergence

                fp_div = FP_Divergence(FP_fm, metric)
                df_red = fp_div.getDivergenceMetricNotRedundant(
                    th_redundancy=th_red, sortV=False
                )[[fp_div.t_value_col]]
                red_info[th_red_str][sup] = len(df_red)

        return red_info

    import numpy as np
    import os
    import pandas as pd
    from pathlib import Path

    from import_datasets import discretize

    from pathlib import Path

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)

    print(f"Output results in directory {main_output_dir}")

    Path(main_output_dir).mkdir(parents=True, exist_ok=True)

    min_supports = [
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
    ]

    #########################################################################
    # Computation for the adult dataset
    dataset_name = "adult"

    processed_dataset_dir = os.path.join(os.path.curdir, "datasets", "processed")

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

        X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP = train_predict(
            dfI, type_cl="RF", labelEncoding=True, validation="cv"
        )
        attributes = dfI.columns.drop("class")

        X_discretized = discretize(
            dfI, attributes=attributes, indexes_FP=indexes_FP, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

    red_info_adult = compute_redundancy_stats(
        X_discretized,
        class_map,
        min_supports,
        dataset_name=dataset_name,
    )

    ##############################################################################################
    # Computation for the compas dataset

    dataset_name = "compas"

    risk_class_type = True

    # Probublica analysis https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm

    from import_datasets import import_process_compas

    dfI, class_map = import_process_compas(
        risk_class=risk_class_type, inputDir=dataset_dir
    )
    dfI.reset_index(drop=True, inplace=True)

    dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})

    attributes = dfI.columns.drop(["class", "predicted"])
    X_FP = dfI[attributes]
    y_FP = dfI[["class"]]
    y_predict_prob = None
    y_predicted = np.asarray(dfI["predicted"])

    X_FP = dfI[attributes].copy()
    X_discretized = discretize(dfI, attributes=attributes, dataset_name=dataset_name)

    X_discretized["class"] = y_FP["class"]
    X_discretized["predicted"] = y_predicted

    red_info_compas = compute_redundancy_stats(
        X_discretized, class_map, min_supports, dataset_name=dataset_name
    )

    ##############################################################################################
    # Plot the results

    info_dicts = red_info_compas
    info_dicts2 = red_info_adult

    sizeFig = (6, 4)
    markersize = 3
    linewidth = 1
    xlabel = "Minimum support s"

    import matplotlib.pyplot as plt

    m_i = 0
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "*",
        "8",
        "s",
        "p",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for label, info_dict in info_dicts.items():
        ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=markers[m_i],
            linewidth=linewidth,
            markersize=markersize,
        )
        ax2.plot(
            list(info_dicts2[label].keys()),
            list(info_dicts2[label].values()),
            label=label,
            marker=markers[m_i],
            linewidth=linewidth,
            markersize=markersize,
        )
        m_i = m_i + 1
    import cycler

    plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=plt.get_cmap("tab10").colors)

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    ax1.set_xlabel(f"{xlabel}\n\n(a)")
    ax2.set_xlabel(f"{xlabel}\n\n(b)")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_title("COMPAS")
    ax2.set_title("adult")
    ax1.set_ylabel("#FP")

    fig.tight_layout(pad=0.5)
    ax1.legend(title="ε", fontsize=10)

    output_fig_dir = os.path.join(os.path.curdir, "output", "figures")

    Path(output_fig_dir).mkdir(parents=True, exist_ok=True)

    figure_name = os.path.join(output_fig_dir, "figure_10.pdf")

    plt.savefig(figure_name, bbox_inches="tight")

    if show_figures:
        plt.show()
        plt.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    redundancy_experiments(
        name_output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )
