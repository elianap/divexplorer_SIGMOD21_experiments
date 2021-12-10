import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def count_per_type(d):
    attr_types = dict(d.dtypes)
    cat = []
    for a, t in attr_types.items():
        if t == object:
            cat.append(a)

    cont = list(set(attr_types.keys()) - set(cat))
    return (len(d), len(d.columns), len(cont), len(cat))


def dataset_descriptions(
    name_output_dir="output",
    dataset_dir=DATASET_DIRECTORY,
    dataset_experiment=["adult", "artificial_10", "bank", "compas", "german", "heart"],
):

    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    out_table = {}

    if "adult" in dataset_experiment:
        dataset_name = "adult"

        from import_datasets import import_process_adult

        dfI, _ = import_process_adult(inputDir=dataset_dir)

        attributes = dfI.columns.drop("class")
        out_table[dataset_name] = count_per_type(dfI[attributes])

    if "bank" in dataset_experiment:

        dataset_name = "bank"

        from import_datasets import import_process_bank

        dfI, _ = import_process_bank(inputDir=dataset_dir)

        attributes = dfI.columns.drop("class")
        out_table[dataset_name] = count_per_type(dfI[attributes])

    if "compas" in dataset_experiment:
        dataset_name = "compas"

        risk_class_type = True

        # Probublica analysis https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm

        from import_datasets import import_process_compas

        dfI, _ = import_process_compas(risk_class=risk_class_type)

        attributes = dfI.columns.drop(["class", "predicted"])

        out_table[dataset_name] = count_per_type(dfI[attributes])

    if "german" in dataset_experiment:

        dataset_name = "german"

        from import_datasets import import_process_german

        dfI, _ = import_process_german(inputDir=dataset_dir)

        attributes = dfI.columns.drop("class")
        out_table[dataset_name] = count_per_type(dfI[attributes])

    if "heart" in dataset_experiment:

        dataset_name = "heart"

        from import_datasets import import_process_heart

        dfI, _ = import_process_heart(inputDir=dataset_dir)

        attributes = dfI.columns.drop("class")
        out_table[dataset_name] = count_per_type(dfI[attributes])

    if "artificial_10" in dataset_experiment:
        dataset_name = "artificial_10"

        data = np.concatenate([np.full((25000, 10), 0), np.full((25000, 10), 1)])

        out_table[dataset_name] = len(data), 10, 0, 10

    out_table = pd.DataFrame(out_table).T.reset_index()

    out_table.columns = [
        "dataset",
        "|D|",
        "|A|",
        "|A|_cont",
        "|A|_cat",
    ]
    print(out_table)

    outputdir = os.path.join(os.path.curdir, name_output_dir, "tables")

    print(f"Output in directory {outputdir}")

    Path(outputdir).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(outputdir, "table_4.csv")
    out_table.to_csv(filename, index=False)
    caption_str = "Table 4: Dataset characteristics. A_cont is the set of continuous attributes, A_cat of categorical ones."
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
        "--dataset_experiment",
        nargs="*",
        type=str,
        default=["adult", "artificial_10", "bank", "compas", "german", "heart"],
        help='specify a list of dataset among ["adult", "artificial_10", "bank", "compas", "german", "heart"]',
    )

    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )

    args = parser.parse_args()

    dataset_descriptions(
        name_output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        dataset_experiment=args.dataset_experiment,
    )
