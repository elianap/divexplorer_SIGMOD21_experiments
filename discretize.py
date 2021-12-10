from import_datasets import KBinsDiscretizer_continuos, quantizeLOS, cap_gains_fn


def discretizeDf(
    dfI, bins=4, dataset_name=None, attributes=None, indexes_FP=None, col_edges=None
):
    indexes_validation = dfI.index if indexes_FP is None else indexes_FP
    attributes = dfI.columns if attributes is None else attributes
    if dataset_name == "compas":
        X_discretized = dfI[attributes].copy()
        # X_discretized["priors_count"]=X_discretized["priors_count"].apply(lambda x: quantizePrior(x))
        X_discretized["length_of_stay"] = X_discretized["length_of_stay"].apply(
            lambda x: quantizeLOS(x)
        )
        if col_edges:
            for col, edges in col_edges.items():
                X_discretized[col] = discretizeGivenEdges(
                    X_discretized[[col]], col, edges
                )
        else:
            X_discretized = KBinsDiscretizer_continuos(
                X_discretized, attributes, bins=bins
            )
    elif dataset_name == "adult":
        X_discretized = dfI[attributes].copy()
        X_discretized["capital-gain"] = cap_gains_fn(
            X_discretized["capital-gain"].values
        )
        X_discretized["capital-gain"] = X_discretized["capital-gain"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized["capital-loss"] = cap_gains_fn(
            X_discretized["capital-loss"].values
        )
        X_discretized["capital-loss"] = X_discretized["capital-loss"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized = KBinsDiscretizer_continuos(X_discretized, attributes, bins=bins)
    else:
        X_discretized = KBinsDiscretizer_continuos(dfI, attributes, bins=bins)
    return X_discretized.loc[indexes_validation].reset_index(drop=True)


def discretizeGivenEdges(dt, col, edges):
    X_discretize = dt.copy()

    if len(set(edges)) != len(edges):
        edges = [
            edges[i]
            for i in range(0, len(edges))
            if len(edges) - 1 == i or edges[i] != edges[i + 1]
        ]
    for i in range(0, len(edges)):
        if i == 0:
            data_idx = dt.loc[dt[col] <= edges[i]].index
            X_discretize.loc[data_idx, col] = f"<={edges[i]}"
        if i == len(edges) - 1:
            data_idx = dt.loc[dt[col] > edges[i]].index
            X_discretize.loc[data_idx, col] = f">{edges[i]}"

        data_idx = dt.loc[(dt[col] > edges[i - 1]) & (dt[col] <= edges[i])].index
        X_discretize.loc[data_idx, col] = f"({edges[i-1]}-{edges[i]}]"

    return X_discretize[col]


def discretize_visualize_support(df, attr, bins=3):
    import pandas as pd
    from discretize import KBinsDiscretizer_continuos_ranges
    from discretize import getSupportAttribute
    from IPython.display import display

    X_discretized, ranges_id_dict = KBinsDiscretizer_continuos_ranges(
        df, [attr], bins=bins
    )
    ranges_support_dict = getSupportAttribute(X_discretized[attr], ranges_id_dict)
    display(pd.DataFrame.from_dict(ranges_support_dict, orient="index").T)
    print("Minimum", min(ranges_support_dict.items(), key=lambda x: x[1]))
    plotSupportAttribute(X_discretized[attr], attr, ranges_id_dict)
    return ranges_support_dict


def plotDistributionContinuous(df, attr, bins=10, density=True):
    import matplotlib.pyplot as plt

    plt.hist(df[attr], bins=10, density=True)
    plt.title(attr)
    plt.show()


def plotBarDict(D, col, ylabel=""):
    import matplotlib.pyplot as plt

    plt.bar(
        range(len(D)),
        list(D.values()),
        align="center",
        color="lightsteelblue",
        edgecolor="black",
    )
    plt.xticks(range(len(D)), list(D.keys()))
    plt.title(col)
    plt.ylabel(ylabel)
    plt.show()


def plotSupportAttribute(series, col, range_mapping=None):
    ylabel = "support"
    D = getSupportAttribute(series, range_mapping)
    plotBarDict(D, col, ylabel=ylabel)


def sortByRangeMapping(D, range_mapping=None):
    if range_mapping:
        D = {k: D[k] for k in range_mapping}
    return D


def getSupportAttribute(series, range_mapping=None):
    D = {k: v / len(series) for k, v in dict(series.value_counts()).items()}
    D = sortByRangeMapping(D, range_mapping)
    return D


def printSupportAttribute(series, col):
    D = getSupportAttribute(series)
    print(col, {k: round(v, 4) for k, v in D.items()})


# TODO -- 1 attr at the time
def KBinsDiscretizer_continuos_ranges(dt, attributes=None, bins=3, verbose=False):
    import numpy as np

    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != np.object]
    X_discretize = dt[attributes].copy()
    ranges_dict = {}

    for col in continuous_attributes:
        if len(dt[col].value_counts()) > 10:
            from sklearn.preprocessing import KBinsDiscretizer

            est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
            est.fit(dt[[col]])
            edges = [i.round() for i in est.bin_edges_][0]
            edges = [int(i) for i in edges][1:-1]
            if len(set(edges)) != len(edges):
                edges = [
                    edges[i]
                    for i in range(0, len(edges))
                    if len(edges) - 1 == i or edges[i] != edges[i + 1]
                ]
            if verbose:
                print(edges)
            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]}"
                    ranges_dict[i] = f"<={edges[i]}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]}"
                    ranges_dict[i + 1] = f">{edges[i]}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[data_idx, col] = f"({edges[i-1]}-{edges[i]}]"
                if i != 0:
                    ranges_dict[i] = f"({edges[i-1]}-{edges[i]}]"
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    ranges_dict = dict(sorted(ranges_dict.items(), key=lambda item: item[0]))
    ranges_dict = {v: k for k, v in ranges_dict.items()}
    return X_discretize, ranges_dict


class distribution_visualizer:
    # def __init__(self):

    def visualizeDistributionContinuous(self, df, continuous_attributes):

        import ipywidgets as widgets
        from IPython.display import clear_output

        # TODO
        from IPython.display import display

        # layout = widgets.Layout(width='auto')

        def clearOutput(btn_cl):
            clear_output()
            display(v_box)
            display(v_box_btn)

        def getSelectedItems(b):
            import matplotlib.pyplot as plt

            attr = w.value
            bins = w_bins.value
            plt.hist(df[attr], bins=bins, density=True)
            plt.title(attr)
            plt.show()

        style = {"description_width": "initial"}
        w = widgets.Dropdown(options=continuous_attributes, style=style)

        from ipywidgets import HBox

        w_bins = widgets.IntSlider(
            min=2, max=20, step=1, value=10, description="#bins histogram", style=style
        )

        v_box = HBox([w, w_bins])
        display(v_box)
        btn = widgets.Button(description="Visualize distribution")
        btn.on_click(getSelectedItems)

        btn_cl = widgets.Button(description="Clear output")
        btn_cl.on_click(clearOutput)
        v_box_btn = HBox([btn, btn_cl])

        display(v_box_btn)

    def discretizeVisualizeDistribution(self, df, continuous_attributes):

        import ipywidgets as widgets
        from IPython.display import clear_output

        from IPython.display import display

        def clearOutput(btn_cl):
            clear_output()
            display(v_box)
            display(v_box_btn)

        def getValues_discretize_visualize_support(b):
            from discretize import discretize_visualize_support

            attr = w.value
            bins = w_bins.value
            discretize_visualize_support(df, attr, bins=bins)

        style = {"description_width": "initial"}
        w = widgets.Dropdown(options=continuous_attributes, style=style)

        from ipywidgets import HBox

        w_bins = widgets.IntSlider(
            min=2, max=20, step=1, value=3, description="#bins discretize", style=style
        )

        v_box = HBox([w, w_bins])
        display(v_box)
        btn = widgets.Button(description="Visualize distribution")
        btn.on_click(getValues_discretize_visualize_support)

        btn_cl = widgets.Button(description="Clear output")
        btn_cl.on_click(clearOutput)
        v_box_btn = HBox([btn, btn_cl])

        display(v_box_btn)

    def visualizeDistributionDiscrete(self, df, attributes=None):

        import ipywidgets as widgets
        from IPython.display import clear_output
        import numpy as np

        from IPython.display import display

        if attributes is None:
            attributes = df.columns
        attributes = [a for a in attributes if df.dtypes[a] == np.object]

        def clearOutput(btn_cl):
            clear_output()
            display(v_box)
            display(v_box_btn)

        def getValues_visualize_support(b):
            from discretize import plotSupportAttribute

            attr = w.value
            plotSupportAttribute(df[attr], attr)

        style = {"description_width": "initial"}
        w = widgets.Dropdown(options=attributes, style=style)

        from ipywidgets import HBox

        v_box = HBox([w])
        display(v_box)
        btn = widgets.Button(description="Visualize distribution")
        btn.on_click(getValues_visualize_support)

        btn_cl = widgets.Button(description="Clear output")
        btn_cl.on_click(clearOutput)
        v_box_btn = HBox([btn, btn_cl])

        display(v_box_btn)


# Item name in the paper
i_name = "α"  # or i
# Pattern or itemset name in the paper
p_name = "I"
# Name for diverge in the paper
div_name = "Δ"


def plotShapleyValue2(
    shapley_values,
    sortedF=True,
    metric="",
    nameFig=None,
    saveFig=False,
    height=0.5,
    linewidth=0.8,
    sizeFig=(4, 3),
    labelsize=10,
    titlesize=10,
    title=None,
    abbreviations={},
):
    import matplotlib.pyplot as plt

    # plt.gcf().set_size_inches(20, 10)
    from shapley_value_FPx import abbreviateDict

    if abbreviations:
        shapley_values = abbreviateDict(shapley_values, abbreviations)
    sh_plt = {str(",".join(list(k))): v for k, v in shapley_values.items()}
    metric = f"{div_name}_{{{metric}}}" if metric != "" else ""
    if sortedF:
        sh_plt = {k: v for k, v in sorted(sh_plt.items(), key=lambda item: item[1])}
    plt.barh(
        range(len(sh_plt)),
        sh_plt.values(),
        height=height,
        align="center",
        color="#7CBACB",
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    plt.yticks(range(len(sh_plt)), list(sh_plt.keys()), fontsize=labelsize)
    plt.xticks(fontsize=labelsize)
    # plt.xlabel(
    #     f"${div_name}({i_name}|{p_name})$", size=labelsize
    # )  # - Divergence contribution
    # title="Divergence" if title is None else title
    title = "" if title is None else title
    title = f"{title} ${metric}$" if metric != "" else title  # Divergence
    plt.title(title, fontsize=titlesize)
    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    if saveFig:
        nameFig = "./shap.pdf" if nameFig is None else nameFig
        plt.savefig(f"{nameFig}.pdf", bbox_inches="tight", pad=0.05)
    plt.show()
