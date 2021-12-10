# Item name in the paper
i_name = "α"  # or i
# Pattern or itemset name in the paper
p_name = "I"
# Name for diverge in the paper
div_name = "Δ"


def printable(
    df_print,
    cols=["itemsets"],
    abbreviations={},
    n_rows=3,
    decimals=(2, 3),
    resort_cols=True,
):

    if type(decimals) is tuple:
        r1, r2 = decimals[0], decimals[1]
    else:
        r1, r2 = decimals, decimals
    df_print = df_print.copy()
    if "support" in df_print.columns:
        df_print["support"] = df_print["support"].round(r1)
    t_v = [c for c in df_print.columns if "t_value_" in c]
    if t_v:
        df_print[t_v] = df_print[t_v].round(1)
    df_print = df_print.round(r2)
    df_print.rename(columns={"support": "sup"}, inplace=True)
    df_print.columns = df_print.columns.str.replace("d_*", f"{div_name}_")
    df_print.columns = df_print.columns.str.replace("t_value", "t")
    for c in cols:
        df_print[c] = df_print[c].apply(lambda x: sortItemset(x, abbreviations))

    cols = list(df_print.columns)

    if resort_cols:
        cols = [cols[1], cols[0]] + cols[2:]

    return df_print[cols]


def sortItemset(x, abbreviations={}):
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def printableCorrective(corr_df, metric_name, n_rows=5, abbreviations={}, colsOfI=None):
    colsOfI = (
        ["S", "item i", "v_S", "v_S+i", "corr_factor", "t_value_corr"]
        if colsOfI is None
        else colsOfI
    )
    output_cols = [c for c in colsOfI if c in corr_df.columns]
    corr_df = corr_df[output_cols]
    corr_df_pr = printable(
        corr_df.head(n_rows),
        cols=["item i", "S"],
        abbreviations=abbreviations,
        resort_cols=False,
    )
    corr_df_pr.rename(
        columns={
            "item i": f"corr. item",
            "S": f"{p_name}",
            "v_i": f"${div_name}_{{metric_name}}({i_name})$",
            "v_S": f"${div_name}_{{{metric_name}}}({p_name})$",
            "v_S+i": f"${div_name}_{{{metric_name}}}({p_name} \cup {i_name})$",
            "corr_factor": "c_f",
            "t_S+i": f"t",
        },
        inplace=True,
    )

    return corr_df_pr


def printableAll(dfs, rename_cols=True):
    import pandas as pd

    div_all = pd.DataFrame()
    for df_i in dfs:
        if rename_cols:
            df_i = df_i.rename(
                columns={"d_fpr": "FPR", "d_fnr": "FNR", "d_accuracy": "ACC"}
            )
        df_i = df_i.T.reset_index().T
        div_all = div_all.append(df_i, ignore_index=True)
    return div_all.rename(columns=div_all.iloc[0]).drop(div_all.index[0])
