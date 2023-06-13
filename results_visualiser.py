import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def visualise_results():
    results_df = _transform_json_to_dataframes(_parse_results_file())
    _visualise_model_param(results_df["GRU"], "GRU", "Loss", (8,6))
    _visualise_model_param(results_df["GRU"], "GRU", "Perplexity", (12,6))
    _visualise_model_param(results_df["LSTM"], "LSTM", "Loss", (8,6))
    _visualise_model_param(results_df["LSTM"], "LSTM", "Perplexity", (12,6))



def _parse_results_file():
    results_json = {"GRU": {"training": [], "testing": []}, 
                    "LSTM": {"training": [], "testing": []}}
    model_name, dataset_name = None, None

    with open("results.txt", "r") as f:
        for line in f:
            if "Epoch" in line:
                if line.startswith("Test"):
                    dataset_name = "testing"
                else:
                    dataset_name = "training"
                loss, perp = _get_loss_perplexity(line)
                results_json[model_name][dataset_name].append({"loss": loss, "perplexity": perp})
            elif "GRU" in line:
                model_name = "GRU"
            elif "LSTM" in line:
                model_name = "LSTM"

    return results_json


def _get_loss_perplexity(line: str):
    loss, perplexity = None, None
    for part in line.split(", "):
        if "Loss" in part:
            loss = float(part.split(" ")[1])
        elif "Perplexity" in part:
            perplexity = float(part.split(" ")[1])
    return loss, perplexity

def _transform_json_to_dataframes(results_json: dict):
    results_df = dict()
    epochs = [i for i in range(1, 14)] + [i for i in range(1, 14)]
    for model in ["GRU", "LSTM"]:
        dataset_name, perplexity, loss = [], [], []
        for dataset in ["training", "testing"]:
            for epoch_records in results_json[model][dataset]:
                dataset_name.append(dataset)
                loss.append(epoch_records["loss"])
                perplexity.append(epoch_records["perplexity"])
        results_df[model] = pd.DataFrame(list(zip(epochs, dataset_name, loss, perplexity)),
                                         columns=["Epoch", "Dataset", "Loss", "Perplexity"])
    return results_df

def _visualise_model_param(df, model_name: str, param_to_vis: str, fig_size: tuple):
    sns.set(rc={"figure.figsize":fig_size})
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(data=df, x=df["Epoch"], y=df[param_to_vis], hue=df["Dataset"], palette=["blue", "green"], marker="o")
    plt.title("GCRN {} - {}".format(model_name, param_to_vis))
    for item, color in zip(df.groupby("Dataset"),["green", "blue"]):
        for x,y,m in item[1][["Epoch",param_to_vis,param_to_vis]].values:
            plt.text(x,round(y, 2),round(m, 2),color=color)
    plt.savefig("{}_{}.png".format(model_name, param_to_vis), format="png")
    plt.close()
