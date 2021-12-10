def showHitsPartialHits(
    hits,
    partial_hits,
    keys,
    saveFig=False,
    nameFig="hits_partial_hits",
    ylabel="",
    max_y_tick=10,
    step=1,
    show_figure=True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    labelsize = 9.4

    N = 4

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = ax.bar(ind, hits, width, color="steelblue")
    p2 = ax.bar(ind, partial_hits, width, bottom=hits, hatch="///", color="skyblue")

    plt.ylabel(ylabel)
    plt.title("Hits and partial hits", fontsize=labelsize)
    plt.xticks(ind, keys, fontsize=labelsize)
    plt.yticks(np.arange(0, (1 + max_y_tick) * step, step), fontsize=labelsize)

    plt.legend((p1[0], p2[0]), ("Hits", "Partial hits"), fontsize=labelsize)
    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = (4.4, 1.9), 100

    if saveFig:
        plt.savefig(nameFig, bbox_inches="tight")

    if show_figure:
        plt.show()
        plt.close()


def show_survey_results(name_output_dir="output", show_figures=True):
    import os
    from pathlib import Path

    c1 = "P"  # first campaign
    uni2 = "S"  # second campaign

    survey = {"None": {}, "DivExplorer": {}, "Slice Finder": {}, "LIME": {}}

    survey["None"][c1] = {"hit": 0, "partial": 0, "total": 5}
    survey["None"][uni2] = {"hit": 1, "partial": 1, "total": 5}

    survey["DivExplorer"][c1] = {"hit": 3, "partial": 0, "total": 3}
    survey["DivExplorer"][uni2] = {"hit": 3, "partial": 2, "total": 6}

    survey["Slice Finder"][c1] = {"hit": 1, "partial": 2, "total": 4}
    survey["Slice Finder"][uni2] = {"hit": 0, "partial": 2, "total": 4}

    survey["LIME"][c1] = {"hit": 1, "partial": 0, "total": 3}
    survey["LIME"][uni2] = {"hit": 1, "partial": 1, "total": 5}

    # Total participants for group

    for k in survey:
        survey[k]["total"] = {
            key2: survey[k][c1][key2] + survey[k][uni2][key2]
            for uni in survey[k]
            for key2 in survey[k][uni]
        }

    output_dir_fig = os.path.join(os.path.curdir, name_output_dir, "figures")

    print(f"Output figure in directory {output_dir_fig}")

    Path(output_dir_fig).mkdir(parents=True, exist_ok=True)
    name_figure = os.path.join(output_dir_fig, "figure_12.pdf")

    tot_k = "total"

    # Derived the information of the combined hit, defined as the sum of the complete and partial hit rates
    combined_hit = {
        k: 100
        * float(
            (survey[k][tot_k]["hit"] + survey[k][tot_k]["partial"])
            / survey[k][tot_k]["total"]
        )
        for k in survey
    }

    print("Combined hit rate")
    print(combined_hit)

    hits = [
        100 * float(survey[k][tot_k]["hit"] / survey[k][tot_k]["total"]) for k in survey
    ]
    partial_hits = [
        100 * float(survey[k][tot_k]["partial"] / survey[k][tot_k]["total"])
        for k in survey
    ]
    showHitsPartialHits(
        hits,
        partial_hits,
        list(survey.keys()),
        nameFig=name_figure,
        saveFig=True,
        ylabel="Percentage %",
        max_y_tick=10,
        step=10,
        show_figure=show_figures,
    )

    caption_str = "Figure 12: User study results. Percentage of hits for the injected bias according to the provided information."
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
        "--no_show_figs",
        action="store_false",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    args = parser.parse_args()

    show_survey_results(
        name_output_dir=args.name_output_dir, show_figures=args.no_show_figs
    )
