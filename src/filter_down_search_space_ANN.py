import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import numpy as np
import itertools

pdflatex_path = "/usr/bin/pdflatex"
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def find_matching_rows(filename):
    path = "tuning_results_ANN/"
    df = pd.read_csv(path + filename, delimiter="\t")

    matching_rows = []
    for i in range(len(df)):
        matches = []
        for j in range(i + 1, len(df)):
            common_values = sum(
                df.loc[i, ["neurons", "num_hidden_layers", "loss_function", "batch_size"]].values
                == df.loc[j, ["neurons", "num_hidden_layers", "loss_function", "batch_size"]].values
            )
            if common_values >= 3:
                matches.append(j + 1)
        if len(matches) >= 1:
            matching_rows.append((i + 1, matches))

    return matching_rows


def plot_neurons_vs_rmse():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")

    fig, ax = plt.subplots(figsize=(3, 3))

    unique_combinations = df.drop_duplicates(subset=["num_hidden_layers", "loss_function", "batch_size"])

    markers = ["o", "s", "v", "^", "D", "X", "*", "h", "p", "P"]

    for i, combination in enumerate(unique_combinations.values):
        group_df = df[
            (df["num_hidden_layers"] == combination[2])
            & (df["loss_function"] == combination[3])
            & (df["batch_size"] == combination[4])
        ]

        unique_neurons = group_df["neurons"].unique()

        if len(unique_neurons) <= 1:
            continue

        marker = markers[i % len(markers)]

        sorted_df = group_df.sort_values("neurons")

        ax.plot(
            sorted_df["neurons"],
            sorted_df["val_rmse"],
            marker=marker,
            markersize=4,
            linestyle="-",
            linewidth=1,
            color="gray",
            label=f"{combination[2]} HL, {combination[3].upper()}, {combination[4]} BS",
        )

        ax.scatter(
            sorted_df["neurons"],
            sorted_df["val_rmse"],
            marker=marker,
            color="white",
            edgecolors="gray",
            s=20,
        )

    x_ticks = np.arange(100, df["neurons"].max() + 100, 100)
    ax.set_xticks(x_ticks)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Number of Neurons")
    ax.set_ylabel("Validation loss, RMSE")

    ax.legend()
    for ftype in ["pdf", "pgf"]:
        plt.tight_layout()
        plt.savefig(f"summarized_data_figures_datafiles/{ftype}_plots/tuning_ANN_comparing_neurons.{ftype}")


def plot_HL_vs_rmse():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")

    fig, ax = plt.subplots(figsize=(3, 3))

    unique_combinations = df.drop_duplicates(subset=["neurons", "loss_function", "batch_size"])

    markers = itertools.cycle(["o", "s", "v", "*", "D", "X", "*", "h", "p", "P"])

    for i, combination in enumerate(unique_combinations.values):
        group_df = df[
            (df["neurons"] == combination[1])
            & (df["loss_function"] == combination[3])
            & (df["batch_size"] == combination[4])
        ]

        unique_HL = group_df["num_hidden_layers"].unique()

        if len(unique_HL) <= 1:
            continue

        marker = next(markers)

        sorted_df = group_df.sort_values("num_hidden_layers")

        ax.plot(
            sorted_df["num_hidden_layers"].astype(int),
            sorted_df["val_rmse"],
            marker=marker,
            markersize=4,
            linestyle="-",
            linewidth=1,
            color="gray",
            label=f"{combination[1]} N, {combination[3].upper()}, {combination[4]} BS",
        )

        ax.scatter(
            sorted_df["num_hidden_layers"],
            sorted_df["val_rmse"],
            marker=marker,
            color="white",
            edgecolors="gray",
            s=20,
        )

    ax.set_xlabel("Number of Hidden Layers")
    ax.set_ylabel("Validation loss, RMSE")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    for ftype in ["pdf", "pgf"]:
        plt.tight_layout()
        plt.savefig(f"summarized_data_figures_datafiles/{ftype}_plots/tuning_ANN_comparing_HL.{ftype}")


def plot_lossfunc_vs_rmse():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")

    fig, ax = plt.subplots(figsize=(3, 3))

    unique_combinations = df.drop_duplicates(subset=["neurons", "num_hidden_layers", "batch_size"])

    markers = itertools.cycle(["o", "s", "v", "*", "D", "X", "*", "h", "p", "P"])

    for i, combination in enumerate(unique_combinations.values):
        group_df = df[
            (df["neurons"] == combination[1])
            & (df["num_hidden_layers"] == combination[2])
            & (df["batch_size"] == combination[4])
        ]

        unique_loss = group_df["loss_function"].unique()

        if len(unique_loss) <= 1:
            continue

        marker = next(markers)

        sorted_df = group_df.sort_values("loss_function")

        ax.plot(
            sorted_df["loss_function"],
            sorted_df["val_rmse"],
            marker=marker,
            markersize=4,
            linestyle="-",
            linewidth=1,
            color="gray",
            label=f"{combination[1]} N, {combination[2]} HL, {combination[4]} BS",
        )

        ax.scatter(
            sorted_df["loss_function"],
            sorted_df["val_rmse"],
            marker=marker,
            color="white",
            edgecolors="gray",
            s=20,
        )

    ax.set_xlabel("Loss func")
    ax.set_ylabel("Validation loss, RMSE")

    ax.legend()
    for ftype in ["pdf", "pgf"]:
        plt.tight_layout()
        plt.savefig(f"summarized_data_figures_datafiles/{ftype}_plots/tuning_ANN_comparing_lossfuncs.{ftype}")


def plot_batch_size_vs_rmse():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")

    fig, ax = plt.subplots(figsize=(3, 3))

    unique_combinations = df.drop_duplicates(subset=["neurons", "num_hidden_layers", "loss_function"])

    markers = ["o", "s", "v", "^", "D", "X", "*", "h", "p", "P"]

    for i, combination in enumerate(unique_combinations.values):
        group_df = df[
            (df["num_hidden_layers"] == combination[2])
            & (df["loss_function"] == combination[3])
            & (df["neurons"] == combination[1])
        ]

        unique_neurons = group_df["batch_size"].unique()

        if len(unique_neurons) <= 1:
            continue

        marker = markers[i % len(markers)]

        sorted_df = group_df.sort_values("batch_size")

        ax.plot(
            sorted_df["batch_size"],
            sorted_df["val_rmse"],
            marker=marker,
            markersize=4,
            linestyle="-",
            linewidth=1,
            color="gray",
            label=f"{combination[1]} N,{combination[2]} HL, {combination[3].upper()}",
        )

        ax.scatter(
            sorted_df["batch_size"],
            sorted_df["val_rmse"],
            marker=marker,
            color="white",
            edgecolors="gray",
            s=20,
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Validation loss, RMSE")

    ax.legend()
    for ftype in ["pdf", "pgf"]:
        plt.tight_layout()
        plt.savefig(f"summarized_data_figures_datafiles/{ftype}_plots/tuning_ANN_comparing_batch_size.{ftype}")


def check_duplicates_loss():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")
    df["sum"] = df["neurons"] + df["num_hidden_layers"] + df["batch_size"]
    print(df["sum"])
    print(df["sum"].duplicated().any())
    print(df["sum"][df["sum"].duplicated()])


def check_duplcates_batch_size():
    path = "tuning_results_ANN/df_results_8_HL.csv"
    df = pd.read_csv(path, delimiter="\t")
    df["sum"] = df["neurons"].astype(str) + " " + df["num_hidden_layers"].astype(str) + " " + df["loss_function"]
    print(df["sum"])
    print(df["sum"].duplicated().any())
    print(df["sum"][df["sum"].duplicated()])


if __name__ == "__main__":
    plot_neurons_vs_rmse()
    plot_HL_vs_rmse()
    plot_lossfunc_vs_rmse()
    # check_duplicates_loss()
    plot_batch_size_vs_rmse()
    # check_duplcates_batch_size()
