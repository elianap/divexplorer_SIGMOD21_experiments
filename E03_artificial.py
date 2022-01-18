import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def artificial_experiments(
    name_output_dir="output",
    compute_results=["figure_4"],
    show_figures=True,
):

    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
    from divexplorer.FP_Divergence import FP_Divergence
    from divexplorer.shapley_value_FPx import compareShapleyValues, normalizeMax

    # Check input
    mex = []

    supported_values = [
        "figure_4",
    ]

    for compute_result in compute_results:

        if compute_result not in supported_values:
            mex.append(compute_result)
        if mex != []:
            mex = f"{' '.join(mex)} are not possible results, select one or more among {supported_values}"
            raise ValueError(mex)

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)

    print(f"Output results in directory {main_output_dir}")

    # # Dataset

    # Create the artificial dataset

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

    # We add FP errors
    import random

    indexes_class1 = list(indexes)
    random.Random(7).shuffle(indexes_class1)
    df_artificial_3["predicted"] = df_artificial_3["class"]
    df_artificial_3.loc[indexes_class1[: int(len(indexes_class1) / 2)], "class"] = 0

    class_map = {"N": 0, "P": 1}

    # # Extract divergence

    # Input:
    # - discretized dataframe
    # - true class column name
    # - predicted class column name

    # Parameters: minimum support of the extracted patterns
    min_sup = 0.01

    fp_diver = FP_DivergenceExplorer(
        df_artificial_3,
        "class",
        "predicted",
        class_map=class_map,
        dataset_name=dataset_name,
    )
    FP_fm = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup, metrics=["d_fpr"]
    )

    # Frequent pattern divergence extraction
    fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

    if "figure_4" in compute_results:

        output_dir_figs = os.path.join(main_output_dir, "figures")

        Path(output_dir_figs).mkdir(parents=True, exist_ok=True)

        output_file_name = os.path.join(output_dir_figs, "figure_4")

        print("-----------------------------------------------------------------------")

        global_shapley_fpr = fp_divergence_fpr.computeGlobalShapleyValue()
        compareShapleyValues(
            normalizeMax(global_shapley_fpr),
            normalizeMax(fp_divergence_fpr.getFItemsetsDivergence()[1]),
            title=[r"$\tilde{\Delta}^g_{FPR}$", "$\Delta_{FPR}$"],
            labelsize=8,
            height=0.5,
            sizeFig=(6, 5),
            saveFig=True,
            subcaption=True,
            nameFig=output_file_name,
            show_figure=show_figures,
        )
        caption_str = r"Figure 4: Relative magnitudes of $\tilde{\Delta}^g(\cdot, s)$ and individual item divergence, for false-positive rate in the artificial dataset. The attributes a, b, c give raise to divergence when appearing together: global divergence captures this."
        print(caption_str)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output",
        help="specify the name of the output folder",
    )
    parser.add_argument(
        "--compute_results",
        nargs="*",
        type=str,
        default=["figure_4"],
        help='specify the figures and tables to compute, specify one or more among ["figure_4"]',
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    artificial_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        show_figures=args.no_show_figs,
    )
