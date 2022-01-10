import common
import common_org
import datasets
import os
import numpy as np
import random
import torch
from random import SystemRandom
from parse import parse_args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASELINE_MODELS = ["ncde", "odernn", "dt", "decay", "gruode", "odernn_forecasting"]

args = parse_args()


def main(
    dataset_name=args.dataset_name,
    manual_seed=args.seed,
    missing_rate=args.missing_rate,  # dataset parameters
    device="cuda",
    max_epochs=1000,
    *,  # training parameters
    model_name=args.model,
    hidden_channels=args.h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    lr=args.lr,
    slope_check=args.slope_check,
    soft=args.soft,
    timewise=args.timewise,
    attention_channel=args.attention_channel,
    attention_attention_channel=args.attention_attention_channel,
    step_mode=args.step_mode,
    dry_run=False,
    c1=args.c1,
    c2=args.c2,
    rtol=args.rtol,
    atol=args.atol,
    **kwargs
):

    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)

    batch_size = 32
    lr = lr * (batch_size / 32)

    PATH = os.path.dirname(os.path.abspath(__file__))

    # Need the intensity data to know how long to evolve for in between observations, but the model doesn't otherwise
    # use it because of use_intensity=False below.
    intensity_data = True if model_name in ("odernn", "dt", "decay") else False

    (
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_classes,
        input_channels,
    ) = datasets.uea.get_data(
        dataset_name,
        missing_rate,
        device,
        intensity=intensity_data,
        batch_size=batch_size,
    )

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes
    experiment_id = int(SystemRandom().random() * 100000)
    file = PATH + "/" + dataset_name + "_h_prime/" + f"{experiment_id}.npy"
    if model_name in BASELINE_MODELS:
        make_model = common_org.make_model(
            model_name,
            input_channels,
            output_channels,
            hidden_channels,
            hidden_hidden_channels,
            num_hidden_layers,
            use_intensity=False,
            initial=True,
        )
    else:  # attention models
        make_model = common.make_model(
            model_name,
            input_channels,
            output_channels,
            hidden_channels,
            hidden_hidden_channels,
            attention_channel,
            attention_attention_channel,
            num_hidden_layers,
            rtol=args.rtol,
            atol=args.atol,
            file=file,
            use_intensity=False,
            slope_check=slope_check,
            soft=soft,
            timewise=timewise,
            initial=True,
        )

    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))

    experiments = name + str(manual_seed)
    if model_name in BASELINE_MODELS:
        return common_org.main(
            name,
            times,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            make_model,
            num_classes,
            max_epochs,
            lr,
            kwargs,
            step_mode=False,
        )
    # attentive models

    return common.main(
        experiments,
        model_name,
        name,
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        make_model,
        num_classes,
        max_epochs,
        lr,
        slope_check,
        kwargs,
        step_mode=step_mode,
        c1=args.c1,
        c2=args.c2,
    )


def run_all(
    group, device, dataset_name, model_names=("ncde", "odernn", "dt", "decay", "gruode")
):
    if group == 1:
        missing_rate = 0.3
    elif group == 2:
        missing_rate = 0.5
    elif group == 3:
        missing_rate = 0.7
    else:
        raise ValueError
    model_kwargs = dict(
        ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
        dt=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        decay=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        gruode=dict(
            hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None
        ),
    )
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(
                dataset_name,
                missing_rate,
                device,
                model_name=model_name,
                **model_kwargs[model_name],
            )


if __name__ == "__main__":

    main()
