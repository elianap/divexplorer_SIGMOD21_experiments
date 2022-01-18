import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def compas_experiments(
    name_output_dir="output",
    compute_results=[
        "table_1",
        "figure_1",
        "table_2",
        "figure_2",
        "table_3",
        "figure_3",
        "figure_5",
    ],
    dataset_dir=DATASET_DIRECTORY,
    show_figures=True,
):

    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
    from divexplorer.FP_Divergence import FP_Divergence, abbreviateDict

    from import_datasets import import_process_compas, discretize

    from utils_print import printable

    # Check input
    mex = []

    supported_values = [
        "table_1",
        "figure_1",
        "table_2",
        "figure_2",
        "table_3",
        "figure_3",
        "figure_5",
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

    # Abbreviation as in the paper, just for vizualization (and space constraint) purposes

    abbreviations = {
        "age_cat": "age",
        "priors_count": "#prior",
        "Greater than 45": ">45",
        "25 - 45": "25-45",
        "African-American": "Afr-Am",
        "c_charge_degree": "charge",
        "Less than 25": "<25",
        "=>": ">",
        "=<": "<",
        "length_of_stay": "stay",
        "Caucasian": "Cauc",
    }

    dataset_name = "compas"

    risk_class_type = True

    # COMPAS dataset
    # Probublica analysis https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    # We use the dataset "compas-scores-two-years.csv" available at https://github.com/propublica/compas-analysis  --> the dataset should be in the ./datasets folder.
    # We follow the pre-process of Propublica (https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm/)
    # and as the NIPS 2017 paper "Optimized Data Pre-Processing for Discrimination Prevention" as in  https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb

    dfI, class_map = import_process_compas(
        risk_class=risk_class_type, inputDir=dataset_dir
    )
    dfI.reset_index(drop=True, inplace=True)

    dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})

    attributes = dfI.columns.drop(["class", "predicted"])

    # Discretize the dataset
    X_discretized = discretize(dfI, attributes=attributes, dataset_name=dataset_name)

    X_discretized["class"] = dfI["class"]
    X_discretized["predicted"] = dfI["predicted"]

    # # Extract divergence

    # Input:
    # - discretized dataframe
    # - true class column name
    # - predicted class column name

    # Parameters: minimum support of the extracted patterns

    min_sup = 0.1

    fp_diver = FP_DivergenceExplorer(
        X_discretized,
        "class",
        "predicted",
        class_map=class_map,
        dataset_name=dataset_name,
    )

    # Frequent pattern divergence extraction
    FP_fm = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup, metrics=["d_fpr", "d_fnr", "d_error", "d_accuracy"]
    )

    if "table_1" in compute_results:

        # Print the teaser table (Table 1)

        fpr_dataset, fnr_dataset = FP_fm.loc[FP_fm.itemsets == frozenset()][
            ["fpr", "fnr"]
        ].values[0]
        overall_values = {"d_fpr": fpr_dataset, "d_fnr": fnr_dataset}

        caption_str = f"Table 1: Example of patterns in the COMPAS dataset, along with their FPR or FNR. The overall FPR and FNR are {fpr_dataset:.3f} and {fnr_dataset:.3f}"

        # We select the itemset and the metric of interest, the ones used as running example in the teaser table
        itemset_metric_of_interest = {
            frozenset(
                {
                    "sex=Male",
                    "priors_count=>3",
                    "race=African-American",
                    "age_cat=25 - 45",
                }
            ): "d_fpr",
            frozenset({"race=Caucasian", "age_cat=Greater than 45"}): "d_fnr",
            frozenset({"race=African-American", "sex=Male"}): "d_fpr",
            frozenset(
                {
                    "sex=Male",
                    "priors_count=>3",
                    "race=African-American",
                }
            ): "d_fpr",
            frozenset({"priors_count=0", "race=African-American", "sex=Male"}): "d_fpr",
        }

        teaser_table = []

        for itemset, metric in itemset_metric_of_interest.items():

            itemset, metric_value = (
                FP_fm.loc[FP_fm.itemsets == itemset][["itemsets", metric]]
                .round(3)
                .values[0]
            )
            metric_value = f"{metric}={metric_value+overall_values[metric]:.3f}"
            teaser_table.append([itemset, metric_value])
        teaser_table = pd.DataFrame(teaser_table, columns=["itemsets", "metric"])

        print("-----------------------------------------------------------------------")
        print(
            printable(teaser_table, abbreviations=abbreviations)[["itemsets", "metric"]]
        )
        print(caption_str)

        outputdir = os.path.join(main_output_dir, "tables")

        Path(outputdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outputdir, "table_1.csv")
        teaser_table.to_csv(filename, index=False)

    if "figure_1" in compute_results:

        def getAttributeLocalShapley(fp_divergence_ofI, attribute_name):
            div1 = fp_divergence_ofI.getFItemsetsDivergence()[1]
            div1_ofI = {k: v for k, v in div1.items() if attribute_name in list(k)[0]}
            return div1_ofI

        def getAbbreviateDiscretized(disc):
            abbreviations_discretized = {}
            for v in disc:
                if ("(") in v and "]" in v:
                    v_new = v.replace("(", "")
                    v_new = v_new.replace("]", "")
                    vs = v_new.split("-")
                    if int(vs[0]) + 1 == int(vs[1]):
                        abbreviations_discretized[v] = vs[1]
                    else:
                        abbreviations_discretized[v] = f"[{int(vs[0])+1}-{vs[1]}]"
            return abbreviations_discretized

        def plot_compare_ShapleyValues(
            sh_score_1,
            sh_score_2,
            title=None,
            sharedAxis=False,
            height=0.8,
            linewidth=0.8,
            sizeFig=(10, 10),
            saveFig=False,
            nameFig=None,
            labelsize=10,
            subcaption=True,
            deltaLim=None,
            wspace=0.75,
            show_figure=True,
        ):
            h1, h2 = (
                (height[0], height[1]) if type(height) == list else (height, height)
            )
            import matplotlib.pyplot as plt

            sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
            sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}
            sh_score_1 = {
                k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])
            }
            sh_score_2 = {
                k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])
            }
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100)

            b1 = ax1.barh(
                range(len(sh_score_1)),
                sh_score_1.values(),
                align="center",
                color="#7CBACB",
                height=h1,
                linewidth=linewidth,
                edgecolor="#0C4A5B",
            )
            ax1.set_yticks(range(len(sh_score_1)))
            ax1.set_yticklabels(list(sh_score_1.keys()))

            b2 = ax2.barh(
                range(len(sh_score_2)),
                sh_score_2.values(),
                align="center",
                color="#7CBACB",
                height=h2,
                linewidth=linewidth,
                edgecolor="#0C4A5B",
            )
            ax2.set_yticks(range(len(sh_score_2)))
            ax2.set_yticklabels(list(sh_score_2.keys()))
            if title and len(title) > 1:
                ax1.set_title(title[0])
                ax2.set_title(title[1])

            plt.subplots_adjust(wspace=wspace)

            ax1.tick_params(axis="y", labelsize=labelsize)
            ax2.tick_params(axis="y", labelsize=labelsize)
            ax1.tick_params(axis="x", labelsize=labelsize)
            ax2.tick_params(axis="x", labelsize=labelsize)
            if sharedAxis:
                sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
                if deltaLim:
                    min_x, max_x = min(sh_scores) - deltaLim, max(sh_scores) + deltaLim
                else:
                    min_x, max_x = (
                        min(sh_scores) + min(0.01, min(sh_scores)),
                        max(sh_scores) + min(0.01, max(sh_scores)),
                    )
                ax1.set_xlim(min_x, max_x)
                ax2.set_xlim(min_x, max_x)
            if subcaption:
                ax1.set_xlabel("(a)", size=labelsize)
                ax2.set_xlabel("(b)", size=labelsize)

            # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
            fig.tight_layout()
            if saveFig:
                nameFig = "./shap.pdf" if nameFig is None else f"{nameFig}.pdf"
                plt.savefig(nameFig, format="pdf", bbox_inches="tight", pad=0.01)
            if show_figure:
                plt.show()
                plt.close()
            return fig

        min_sup_discr = 0.05

        # Extraction of the pattern with the default discretization

        fp_diver_discr = FP_DivergenceExplorer(
            X_discretized,
            "class",
            "predicted",
            class_map=class_map,
            dataset_name=dataset_name,
        )
        FP_fm_discr = fp_diver_discr.getFrequentPatternDivergence(
            min_support=min_sup_discr
        )

        attribute_name = "priors_count"

        from divexplorer.FP_Divergence import FP_Divergence, abbreviateDict

        fp_divergence_fpr_discr = FP_Divergence(FP_fm_discr, "d_fpr")

        # We get the individual contribution of the items which are based #prior ( attribute_name)
        div_local_default = getAttributeLocalShapley(
            fp_divergence_fpr_discr, attribute_name
        )

        div_local_default = abbreviateDict(div_local_default, abbreviations)

        # We derive an additional discretization, determined by the ranges <0 (i.e. #prior=0), 1, 2, 3, (3-7] (i.e. #prior=[4-7]), >7

        from discretize import discretizeDf

        col_edges = {attribute_name: [0, 0, 1, 2, 3, 7]}

        X_discretized_6_ranges = discretizeDf(
            dfI, attributes=attributes, dataset_name=dataset_name, col_edges=col_edges
        )
        disc = list(X_discretized_6_ranges.priors_count.value_counts().keys())

        X_discretized_6_ranges["class"] = dfI["class"]
        X_discretized_6_ranges["predicted"] = dfI["predicted"]
        abbreviations_discretized = abbreviations.copy()

        # Since we cannot have negative priors, for visualization purposes we map <= 0 just as =0
        abbreviations_discretized.update({"#prior<=0": "#prior=0"})

        # We extract the itemset divergence for the defined discretization
        fp_diver_6_ranges = FP_DivergenceExplorer(
            X_discretized_6_ranges,
            "class",
            "predicted",
            class_map=class_map,
            dataset_name=dataset_name,
        )
        FP_fm_6_ranges = fp_diver_6_ranges.getFrequentPatternDivergence(
            min_support=min_sup_discr
        )

        fp_divergence_fpr_6_ranges = FP_Divergence(FP_fm_6_ranges, "d_fpr")

        abbreviations_discretized.update(getAbbreviateDiscretized(disc))

        # We get the individual contribution of the items which are based #prior ( attribute_name) for the discretization in 6 ranges
        div_local_6ranges = getAttributeLocalShapley(
            fp_divergence_fpr_6_ranges, attribute_name
        )

        # We abbreviate the itemset names, just for visualization purposes in the figures and tables of the paper
        div_local_6ranges = abbreviateDict(div_local_6ranges, abbreviations_discretized)

        ids = "default_6ranges"

        outputdir = os.path.join(main_output_dir, "figures")
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        nameFig = os.path.join(outputdir, "figure_1")

        r = len(div_local_default) / len(div_local_6ranges)
        h = 0.5

        print("-----------------------------------------------------------------------")
        fig = plot_compare_ShapleyValues(
            div_local_default,
            div_local_6ranges,
            labelsize=7.6,
            height=[h * r, h],
            sharedAxis=True,
            wspace=0.75,
            subcaption=True,
            sizeFig=(3.7, 1.4),
            nameFig=nameFig,
            saveFig=True,
            show_figure=show_figures,
        )

        caption_str = f"Figure 1: Individual item divergence for false-positive rate of prior attribute value of the COMPAS dataset where the attribute is discretized in 3 (a) and 6 (b) intervals (s={min_sup}.)"
        print(caption_str)

    if "table_2" in compute_results:
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

        # Derive the divergence w.r.t. the ERROR raye

        fp_divergence_error = FP_Divergence(FP_fm, "d_error")

        div_error = fp_divergence_error.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_error.metric,
                fp_divergence_error.t_value_col,
            ]
        ]
        div_pr_error = printable(div_error.head(n_rows), abbreviations=abbreviations)

        # Derive the divergence w.r.t. the ACCURACY

        fp_divergence_accuracy = FP_Divergence(FP_fm, "d_accuracy")
        div_accuracy = fp_divergence_accuracy.getDivergence(th_redundancy=0)[
            [
                "support",
                "itemsets",
                fp_divergence_accuracy.metric,
                fp_divergence_accuracy.t_value_col,
            ]
        ]

        div_pr_accuracy = printable(
            div_accuracy.head(n_rows), abbreviations=abbreviations
        )

        dfs = [div_pr_fpr, div_pr_fnr, div_pr_error, div_pr_accuracy]

        from utils_print import printableAll

        # We visualize the results in single table, as in the paper
        div_all = printableAll(dfs)

        print("-----------------------------------------------------------------------")

        print(div_all)

        caption_str = f"Table 2: Top-3 divergent patterns with respect to FPR, FNR, error rate (ER) and accuracy (ACC) for the COMPAS dataset. The support threshold is s = {min_sup}."
        print(caption_str)

        outputdir = os.path.join(main_output_dir, "tables")

        Path(outputdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outputdir, "table_2.csv")
        div_all.to_csv(filename, index=False)

    if "figure_2" in compute_results:
        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # Get the itemset with highest divergence for the FPR
        itemset_top_fpr = list(
            fp_divergence_fpr.getDivergenceTopK(K=1, th_redundancy=0).keys()
        )[0]

        # Compute the LOCAL Shapley value for the itemset with highest divergence for the FPR
        shap_itemset_top_fpr = abbreviateDict(
            fp_divergence_fpr.computeShapleyValue(itemset_top_fpr), abbreviations
        )

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

        import os

        output_dir_shap = os.path.join(main_output_dir, "figures")

        output_file_name = os.path.join(output_dir_shap, "figure_2")

        Path(output_dir_shap).mkdir(parents=True, exist_ok=True)

        from divexplorer.shapley_value_FPx import plotComparisonShapleyValues

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

        caption_str = "Figure 2: Contributions of individual items to the divergence of the COMPAS frequent patterns having greatest false-positive and false-negative divergence."

        print(caption_str)

    if "table_3" in compute_results:

        from utils_print import printableCorrective

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # Get the corrective items for the FPR. These are sorted for highest corrective behavior (corrective factor)
        corrective_fpr = fp_divergence_fpr.getCorrectiveItems()

        # We extract the top 3 corrective behaviors for the FPR
        # The printable corrective is just for visualization purposes (as in the paper)
        corrective_pr_fpr = printableCorrective(
            corrective_fpr.head(3).reset_index(drop=True),
            fp_divergence_fpr.metric_name,
            abbreviations=abbreviations,
        )

        print("-----------------------------------------------------------------------")
        print(corrective_pr_fpr)

        fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")
        # Get the corrective items for the FNR. These are sorted for highest corrective behavior (corrective factor)
        corrective_fnr = fp_divergence_fnr.getCorrectiveItems()

        # We extract the top 3 corrective behaviors for the FNR
        corrective_pr_fnr = printableCorrective(
            corrective_fnr.head(3).reset_index(drop=True),
            fp_divergence_fnr.metric_name,
            abbreviations=abbreviations,
        )
        print(corrective_pr_fnr)

        caption_str = "Table 3: Top corrective items for FPR and FNR of COMPAS dataset."

        print(caption_str)
        from utils_print import printableAll

        df_merged = printableAll(
            [corrective_pr_fpr, corrective_pr_fnr], rename_cols=False
        )

        outputdir = os.path.join(main_output_dir, "tables")

        Path(outputdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outputdir, "table_3.csv")
        df_merged.to_csv(filename, index=False)

    if "figure_3" in compute_results:

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        # Get the corrective items for the FPR. These are sorted for highest corrective behavior (corrective factor)
        corrective_fpr = fp_divergence_fpr.getCorrectiveItems()
        # We extract the itemsets with highest corrective behavior for the FPR
        itemset_corrective_behavior = corrective_fpr.head(1)["S+i"].values[0]

        import os

        output_dir_shap = os.path.join(main_output_dir, "figures")

        output_file_name = os.path.join(output_dir_shap, "figure_3.pdf")

        from pathlib import Path

        Path(output_dir_shap).mkdir(parents=True, exist_ok=True)

        print("-----------------------------------------------------------------------")
        fp_divergence_fpr.plotShapleyValue(
            itemset_corrective_behavior,
            abbreviations=abbreviations,
            saveFig=True,
            nameFig=output_file_name,
            show_figure=show_figures,
        )

        caption_str = (
            "Figure 3: An itemset where an item has a negative divergence contribution."
        )
        print(caption_str)
    if "figure_5" in compute_results:
        # Derive the divergence w.r.t. the FPR

        fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")

        outputdir = os.path.join(main_output_dir, "figures")
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(outputdir, "figure_5")

        # Compute the global Shapley value for the FPR divergence
        global_shapley_fpr = fp_divergence_fpr.computeGlobalShapleyValue()

        print("-----------------------------------------------------------------------")

        from divexplorer.shapley_value_FPx import (
            compareShapleyValues,
            normalizeMax,
        )

        # Plot (and visual comparison) of
        # - global Shapley value for the FPR divergence
        # - individual divergence for the FPR divergence of individual items
        compareShapleyValues(
            normalizeMax(abbreviateDict(global_shapley_fpr, abbreviations)),
            normalizeMax(
                abbreviateDict(
                    fp_divergence_fpr.getFItemsetsDivergence()[1], abbreviations
                )
            ),
            title=[r"$\tilde{\Delta}^g_{FPR}$", "$\Delta_{FPR}$"],
            labelsize=8,
            height=0.5,
            sizeFig=(6, 5),
            saveFig=True,
            subcaption=True,
            nameFig=output_file_name,
            show_figure=show_figures,
        )
        caption_str = f"Figure 5: Relative magnitudes of global Shapley value and individual item divergence, for false-positive rate in the COMPAS dataset with s = {min_sup}."
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
        default=[
            "table_1",
            "figure_1",
            "table_2",
            "figure_2",
            "table_3",
            "figure_3",
            "figure_5",
        ],
        help='specify the figures and tables to compute, specify one or more among ["table_1", "figure_1", "table_2", "figure_2", "table_3", "figure_3", "figure_5"]',
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

    compas_experiments(
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )
