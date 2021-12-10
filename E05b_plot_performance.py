import os


DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def extract_statistics_FP_divergence(inputDir, dataset_names, min_supports):
    import os
    import pandas as pd

    infos_dataset = {}
    for dataset_name in dataset_names:

        name_dir = os.path.join(inputDir, dataset_name)

        infos_dataset[dataset_name] = {}
        import os

        if os.path.isdir(name_dir):
            for sup in min_supports:
                pathNameFile = os.path.join(name_dir, f"{dataset_name}_{sup}.csv")

                if os.path.isfile(pathNameFile):
                    infos_dataset[dataset_name][sup] = pd.read_csv(
                        pathNameFile, index_col=0
                    )
                else:
                    print("NA:", pathNameFile)
        else:
            print(f"Dir not existing: {name_dir}")
    return infos_dataset


def plotDicts(
    info_dicts,
    title="",
    xlabel="",
    ylabel="",
    marker=None,
    limit=None,
    nameFig="",
    colorMap="tab10",
    sizeFig=(4, 3),
    labelSize=8,
    markersize=4,
    outside=False,
    linewidth=1.5,
    titleLegend="",
    tickF=False,
    yscale="linear",
    legendSize=5,
    saveFig=False,
    show_figure=True,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

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
    for label, info_dict in info_dicts.items():
        label_name = (
            label if label != "artificial_10" else "artificial"
        )  # For clarity reasons
        if marker:
            ax.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label_name,
                marker=markers[m_i],
                linewidth=linewidth,
                markersize=markersize,
            )
            m_i = m_i + 1
        else:
            ax.plot(list(info_dict.keys()), list(info_dict.values()), label=label_name)
    import cycler

    if colorMap:
        plt.rcParams["axes.prop_cycle"] = cycler.cycler(
            color=plt.get_cmap(colorMap).colors
        )
    else:
        color = plt.cm.winter(np.linspace(0, 1, 10))
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", color)

    if limit is not None:
        plt.ylim(top=limit)
    if tickF:
        xt = list(info_dict.keys())
        plt.xticks([xt[i] for i in range(0, len(xt)) if i == 0 or xt[i] * 100 % 5 == 0])

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    if outside:
        plt.legend(
            prop={"size": labelSize},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title=titleLegend,
            fontsize=5,
            title_fontsize=5,
        )
    else:
        plt.legend(
            prop={"size": labelSize},
            title=titleLegend,
            fontsize=legendSize,
            title_fontsize=9,
        )
    if saveFig:
        plt.savefig(nameFig, bbox_inches="tight")
    if show_figure:
        plt.show()
        plt.close()


def plot_performance(
    name_input_dir="performance_results",
    name_output_dir="output",
    dataset_experiment=["adult", "artificial_10", "bank", "compas", "german", "heart"],
    show_figures=True,
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
    compute_results=["figure_6", "figure_7"],
):

    import pandas as pd
    import numpy as np
    import os
    from pathlib import Path

    output_fig_dir = os.path.join(os.path.curdir, name_output_dir, "figures")

    print(f"Output results in directory {output_fig_dir}")

    Path(output_fig_dir).mkdir(parents=True, exist_ok=True)

    input_dir = os.path.join(os.path.curdir, name_input_dir)

    infos_dataset = None

    if "figure_6" in compute_results:
        figure_name = os.path.join(output_fig_dir, "figure_6.pdf")

        print("-----------------------------------------------------------------------")

        infos_dataset = extract_statistics_FP_divergence(
            input_dir, dataset_experiment, min_supports
        )

        info = "time"
        type_info = "mean"

        list_info_time = {
            k: {
                sup: dataset_info[sup][info][type_info]
                for sup in sorted(dataset_info.keys())
            }
            for k, dataset_info in infos_dataset.items()
        }

        plotDicts(
            list_info_time,
            xlabel="Minimum support s",
            ylabel="Execution time (s)",
            marker=".",
            linewidth=1,
            markersize=3,
            nameFig=figure_name,
            yscale="log",
            tickF=True,
            saveFig=True,
            show_figure=show_figures,
        )
        caption_str = "Figure 6: DivExplorer execution time when varying the minimum support threshold."
        print(caption_str)

    if "figure_7" in compute_results:
        figure_name = os.path.join(output_fig_dir, "figure_7.pdf")
        if infos_dataset is None:

            infos_dataset = extract_statistics_FP_divergence(
                input_dir, dataset_experiment, min_supports
            )

        info = "support"
        type_info = "count"

        figure_name = os.path.join(output_fig_dir, "figure_7.pdf")
        list_info_FP = {
            k: {
                sup: dataset_info[sup][info][type_info]
                for sup in sorted(dataset_info.keys())
            }
            for k, dataset_info in infos_dataset.items()
        }

        print("-----------------------------------------------------------------------")

        plotDicts(
            list_info_FP,
            xlabel="Minimum support s",
            ylabel="#FP",
            marker=".",
            nameFig=figure_name,
            yscale="log",
            linewidth=1,
            markersize=3,
            tickF=True,
            saveFig=True,
            show_figure=show_figures,
        )
        caption_str = "Figure 7: Number of frequent itemsets when varying the minimum support threshold."
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
        "--name_input_dir",
        default="performance_results",
        help="specify the name of the input folder",
    )

    parser.add_argument(
        "--compute_results",
        nargs="*",
        type=str,
        default=["figure_6", "figure_7"],
        help='specify the figures and tables to compute, specify one or more among ["figure_6","figure_7"]',
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
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

    args = parser.parse_args()

    plot_performance(
        name_input_dir=args.name_input_dir,
        name_output_dir=args.name_output_dir,
        compute_results=args.compute_results,
        show_figures=args.no_show_figs,
        dataset_experiment=args.dataset_experiment,
        min_supports=args.min_supports,
    )
