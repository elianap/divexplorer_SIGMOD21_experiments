import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


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
        action="store_true",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )

    parser.add_argument(
        "--name_output_performance_dir",
        default="performance_results",
        help="specify the name of the output folder of the performance results",
    )

    parser.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="specify the number of times to repeat the estimation, 5 as in the paper experimental results",
    )

    args = parser.parse_args()

    time_results = {}

    import time

    print("Experiments with COMPAS dataset")
    from E01_compas import compas_experiments

    start_time = time.time()
    compas_experiments(
        name_output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )
    time_results["compas"] = time.time() - start_time

    print("______________________________________________\n")

    print("Experiments with adult dataset")

    from E02_adult import adult_experiments

    start_time = time.time()
    adult_experiments(
        name_output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )
    time_results["adult"] = time.time() - start_time
    print("______________________________________________\n")

    print("Experiments with artificial dataset")

    from E03_artificial import artificial_experiments

    start_time = time.time()

    artificial_experiments(
        name_output_dir=args.name_output_dir,
        show_figures=args.no_show_figs,
    )

    time_results["artificial"] = time.time() - start_time

    print("______________________________________________\n")

    print("Redundancy pruning with COMPAS and adult datasets")

    from E04_redundancy import redundancy_experiments

    start_time = time.time()

    redundancy_experiments(
        name_output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        show_figures=args.no_show_figs,
    )

    time_results["redundancy"] = time.time() - start_time

    print("______________________________________________\n")

    print("Dataset description")

    from E06_stats_dataset import dataset_descriptions

    start_time = time.time()

    dataset_descriptions(
        name_output_dir=args.name_output_dir, dataset_dir=args.dataset_dir
    )

    time_results["dataset_description"] = time.time() - start_time

    print("______________________________________________\n")

    print("Estimate time, #FP varying the minimum support")

    from E05a_compute_performance import compute_performance

    start_time = time.time()

    compute_performance(
        name_output_dir=args.name_output_performance_dir,
        n_times=args.n_times,  # 5 as in the paper
        dataset_dir=args.dataset_dir,
    )

    time_results["compute_performance"] = time.time() - start_time

    print("______________________________________________\n")

    print("Visualize time and FP results varying the minimum support")

    from E05b_plot_performance import plot_performance

    start_time = time.time()

    plot_performance(
        name_input_dir=args.name_output_performance_dir,
        name_output_dir=args.name_output_dir,
        show_figures=args.no_show_figs,
    )

    time_results["visualize_performance"] = time.time() - start_time

    print("______________________________________________\n")

    print("Visualize time and FP results varying the minimum support")

    from E07_survey import show_survey_results

    start_time = time.time()

    show_survey_results(
        name_output_dir=args.name_output_dir, show_figures=args.no_show_figs
    )

    time_results["visualize_survey"] = time.time() - start_time

    print("______________________________________________\n")

    import pandas as pd

    time_results = pd.DataFrame.from_dict(
        time_results, orient="index", columns=["time"]
    )
    time_results.index.name = "exp"

    print("Script execution time:")

    print(time_results)
    time_results.to_csv(os.path.join(args.name_output_dir, "execution_time_script.csv"))
