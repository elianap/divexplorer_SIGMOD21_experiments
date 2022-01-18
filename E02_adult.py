import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def adult_experiments(
    name_output_dir="output",
    compute_results=["table_5", "figure_8", "figure_9", "table_6", "figure_11"],
    dataset_dir=DATASET_DIRECTORY,
    show_figures=True,
    use_processed=True,
):

    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path
    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer

    from import_datasets import import_process_adult, discretize, train_predict

    from utils_print import printable, printableAll

    # Check input
    mex = []

    supported_values = ["table_5", "figure_8", "figure_9", "table_6", "figure_11"]

    for compute_result in compute_results:

        if compute_result not in supported_values:
            mex.append(compute_result)
        if mex != []:
            mex = f"{' '.join(mex)} are not possible results, select one or more among {supported_values}"
            raise ValueError(mex)

    main_output_dir = os.path.join(os.path.curdir, name_output_dir)

    # # Dataset

    dataset_name = "adult"

    # Abbreviation as in the paper, just for vizualization (and space constraint) purposes

    abbreviations = {
        "marital-status": "status",
        "High School grad": "HS",
        "education": "edu",
        "relationship": "relation",
        "occupation": "occup",
        "hours-per-week": "hoursXW",
        "workclass": "workcl",
        "Never-Married": "Unmarried",
        "=<=": "≤",
        "Self-emp-not-inc": "self-not-inc",
        "Female": "F",
        "Male": "M",
        "capital-": "",
        "Professional": "Prof",
    }

    from import_datasets import check_dataset_availability

    processed_dataset_dir = os.path.join(dataset_dir, "processed")

    check_dataset_availability(
        "adult_discretized_RF.csv", inputDir=processed_dataset_dir
    )

    adult_processed_file = os.path.join(
        processed_dataset_dir, "adult_discretized_RF.csv"
    )

    # We use the processed file, if available
    # It avoids having small differences in the results due to the process of training and labeling
    # - cross validation
    # - we train a RF model and we label the data --> depending on the library version, we can have small differences on the final labels and hence on our result
    # On the other hand, if the labels are fixed, our results are fixed (no randomicity in our process)

    # If already available, we use the processed dataset, which is already discretized
    if use_processed and os.path.isfile(adult_processed_file):
        X_discretized = pd.read_csv(adult_processed_file)
        class_map = {"N": "<=50K", "P": ">50K"}
    else:
        # Otherwise we do the entire process of importing the original data and process them

        dfI, class_map = import_process_adult(inputDir=DATASET_DIRECTORY)
        dfI.reset_index(drop=True, inplace=True)

        # Train and predict using random forest classifier and cross validation

        type_cl = "RF"
        labelEncoding = True

        # Cross validation
        X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP = train_predict(
            dfI, type_cl=type_cl, labelEncoding=labelEncoding, validation="cv"
        )
        attributes = dfI.columns.drop("class")

        # We discretize the data
        X_discretized = discretize(
            dfI, attributes=attributes, indexes_FP=indexes_FP, dataset_name=dataset_name
        )

        X_discretized["class"] = y_FP["class"]
        X_discretized["predicted"] = y_predicted

    # # Extract divergence

    # Input:
    # - discretized dataframe
    # - true class column name
    # - predicted class column name

    # Parameters: minimum support of the extracted patterns

    min_sup = 0.05

    fp_diver = FP_DivergenceExplorer(
        X_discretized,
        "class",
        "predicted",
        class_map=class_map,
        dataset_name=dataset_name,
    )

    if "table_5" in compute_results:

        # Frequent pattern divergence extraction

        FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_sup)

        n_rows = 3

        from divexplorer.FP_Divergence import FP_Divergence

        # Derive the divergence w.r.t. the FPR

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        div_fpr = fp_divergence_fpr.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_fpr.metric,
                fp_divergence_fpr.t_value_col,
            ]
        ]

        # The printable function derives a vizualization ready dataframe, to visualize the results in the paper: rounded, same order, itemsets are abbreviated for vizualization purposes
        div_pr_fpr = printable(div_fpr.head(n_rows), abbreviations=abbreviations)

        # Derive the divergence w.r.t. the FNR

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")

        div_fnr = fp_divergence_fnr.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_fnr.metric,
                fp_divergence_fnr.t_value_col,
            ]
        ]
        div_pr_fnr = printable(div_fnr.head(n_rows), abbreviations=abbreviations)

        # We visualize the results in single table, as in the paper

        dfs = [div_pr_fpr, div_pr_fnr]

        div_all = printableAll(dfs)

        print("-----------------------------------------------------------------------")
        print(div_all)

        # print(div_all.to_latex(index=False))

        caption_str = f"Table 5: Top-3 divergent itemsets for FPR and FNR. adult dataset, s = {min_sup}."
        print(caption_str)
        outputdir = os.path.join(main_output_dir, "tables")
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outputdir, "table_5.csv")
        div_all.to_csv(filename, index=False)
    if "figure_8" in compute_results:
        from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
        from divexplorer.FP_Divergence import abbreviateDict

        from divexplorer.shapley_value_FPx import plotComparisonShapleyValues

        # Derive the divergence w.r.t. the FPR
        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # Get the itemset with highest divergence for the FPR
        itemset_top_fpr = list(
            fp_divergence_fpr.getDivergenceTopK(K=1, th_redundancy=0).keys()
        )[0]

        # Compute the LOCAL Shapley value for the itemset with highest divergence for the FPR
        shap_itemset_top_fpr = abbreviateDict(
            fp_divergence_fpr.computeShapleyValue(itemset_top_fpr), abbreviations
        )

        # Derive the divergence w.r.t. the FNR
        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")

        # Get the itemset with highest divergence for the FNR
        itemset_top_fnr = list(
            fp_divergence_fnr.getDivergenceTopK(K=1, th_redundancy=0).keys()
        )[0]

        # Compute the LOCAL Shapley value for the itemset with highest divergence for the FNR
        shap_itemset_top_fnr = abbreviateDict(
            fp_divergence_fnr.computeShapleyValue(itemset_top_fnr), abbreviations
        )

        t1, t2 = (
            f"$Δ_{{{fp_divergence_fpr.metric_name}}}$(α|I)",
            f"$Δ_{{{fp_divergence_fnr.metric_name}}}$(α|I)",
        )

        output_dir_shap = os.path.join(main_output_dir, "figures")
        Path(output_dir_shap).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(output_dir_shap, "figure_8")

        print("-----------------------------------------------------------------------")

        # Plot the two LOCAL Shapley values for the frequent patterns having greatest FPR and FNR divergence
        plotComparisonShapleyValues(
            shap_itemset_top_fpr,
            shap_itemset_top_fnr,
            height=0.4,
            sharedAxis=False,
            sizeFig=(5.2, 2.2),
            title=[t1, t2],
            saveFig=True,
            nameFig=output_file_name,
            pad=0.5,
            metrics_name=[fp_divergence_fpr.metric_name, fp_divergence_fnr.metric_name],
            show_figure=show_figures,
        )

        caption_str = f"Figure 8: Contributions of individual items to the divergence of the adult frequent patterns having greatest FPR (Line 1 of Table 5) and FNR (Line 4 of Table 5) divergence."
        print(caption_str)

    if "figure_9" in compute_results:

        # Derive the divergence w.r.t. the FPR

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        outputdir = os.path.join(main_output_dir, "figures")
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(outputdir, "figure_9")

        global_shapley_fpr = fp_divergence_fpr.computeGlobalShapleyValue()

        K = 12

        # For visualization purposes, we reported in the paper the results for the TOP K=12 POSITIVE contributions

        # We firstly extract the POSITIVE contributions
        global_shapley_positive_fpr = {
            k: v for k, v in global_shapley_fpr.items() if v > 0
        }

        # We then derive the top K=12

        global_shapley_positive_fpr_top = {
            k: v
            for k, v in global_shapley_positive_fpr.items()
            if k
            in sorted(
                global_shapley_positive_fpr,
                key=lambda x: global_shapley_positive_fpr[x],
            )[::-1][:K]
        }

        # We get the individual divergence for the FNR divergence of individual items
        # We filter the ones matching the top K=12 for the positive Global Shapley divergence
        individual_shapley_top = {
            k: v
            for k, v in fp_divergence_fpr.getFItemsetsDivergence()[1].items()
            if k in global_shapley_positive_fpr_top.keys()
        }

        from divexplorer.FP_Divergence import abbreviateDict

        from divexplorer.shapley_value_FPx import (
            compareShapleyValues,
            normalizeMax,
        )

        print("-----------------------------------------------------------------------")

        # Plot (and visual comparison) of
        # - global Shapley value for the FNR divergence
        # - individual divergence for the FNR divergence of individual items

        # The results are normalized (since the individual and global divergence have different ranges).
        # We abbreviate the item name as in the paper, for vizualization purporses
        compareShapleyValues(
            normalizeMax(
                abbreviateDict(global_shapley_positive_fpr_top, abbreviations)
            ),
            normalizeMax(abbreviateDict(individual_shapley_top, abbreviations)),
            title=[r"$\tilde{\Delta}^g_{FPR}$", "$\Delta_{FPR}$"],
            labelsize=8,
            height=0.5,
            sizeFig=(5, 4),
            saveFig=True,
            subcaption=True,
            pad=0.5,
            nameFig=output_file_name,
            show_figure=show_figures,
        )

        caption_str = f"Figure 9: Relative magnitude of global Shapley value (a) andindividual item divergence (b), for FPR, adult dataset, s = {min_sup} 0.05. Top {K} global item positive contributions are reported."

        print(caption_str)

    if "table_6" in compute_results:

        # Derive the divergence w.r.t. the FPR

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        ##### Redundandancy threshold.

        # We derive pattern divergence for each frequent patter I if its marginal contribution with all its subsets of lenght len(I)-s is greater than th_redundancy

        # A pattern I is pruned if there exists an item a in I whose absolute marginal contribution is lower than a threshold th_redundancy, i.e. abs( delta_f(I) − delta_f(I \ a) \le th_redundancy.

        th_redundancy = 0.05

        # We get the summarized pattern divergence using th_redundancy as redundancy pruning threshold
        div_fpr_redundancy = fp_divergence_fpr.getDivergence(
            th_redundancy=th_redundancy
        )[
            [
                "support",
                "itemsets",
                fp_divergence_fpr.metric,
                fp_divergence_fpr.t_value_col,
            ]
        ]

        # Printable version for vizualization purposes, as in the paper
        div_fpr_redundancy = printable(
            div_fpr_redundancy.head(n_rows), abbreviations=abbreviations
        )

        print("-----------------------------------------------------------------------")

        print(div_fpr_redundancy)

        outputdir = os.path.join(main_output_dir, "tables")

        Path(outputdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outputdir, "table_6.csv")
        div_fpr_redundancy.to_csv(filename, index=False)

        caption__str = f"Table 6: Top-3 divergent itemsets for FPR with redundancy pruning. adult dataset, ε = {th_redundancy}, s = {min_sup}"
        print(caption__str)

    if "figure_11" in compute_results:

        # Derive the divergence w.r.t. the FPR

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")

        output_dir_lattice = os.path.join(main_output_dir, "figures")
        Path(output_dir_lattice).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(output_dir_lattice, "figure_11.pdf")

        # We select an itemset showing a corrective behavior
        # We firstly get the itemset showing a corrective behavior
        corrSign = fp_divergence_fnr.getCorrectiveItems()

        id_col = 23

        # We visualize and save the lattice of the selected itemset
        if len(corrSign) > id_col:
            S_i = corrSign[["S+i"]].head(id_col + 1).values[id_col][0]
            itemsetsOfInterest = [
                frozenset(
                    {"workclass=Private", "capital-gain=0", "education=Bachelors"}
                ),
                frozenset({"capital-loss=0", "capital-gain=0", "education=Bachelors"}),
                frozenset({"capital-loss=0", "capital-gain=0", "PROVA"}),
            ]
            fig1 = fp_divergence_fnr.plotLatticeItemset(
                S_i,
                Th_divergence=0.15,
                sizeDot="small",
                getLower=True,
                round_v=2,
                displayItemsetLabels=True,
                show=show_figures,
                font_size_div=12,
                font_size_ItemsetLabels=13,
                itemsetsOfInterest=itemsetsOfInterest,
            )

            fig1.write_image(output_file_name, width=600, height=330)


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
        default=["table_5", "figure_8", "figure_9", "table_6", "figure_11"],
        help='specify the figures and tables to compute, specify one or more among  ["table_5", "figure_8", "figure_9", "table_6", "figure_11"]',
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

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    adult_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
        use_processed=args.retrain,
    )
