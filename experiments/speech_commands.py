import common_org
import common
import datasets
import os
import numpy as np
import random
import torch

from parse import parse_args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASELINE_MODELS = ["ncde", "odernn", "dt", "decay", "gruode", "odernn_forecasting"]

args = parse_args()


def main(
    manual_seed=args.seed,
    device="cuda",
    max_epochs=200,
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
    attention_attention_channel=args.attention_attention_channel,  # model parameters
    step_mode=args.step_mode,
    dry_run=False,
    c1=args.c1,
    c2=args.c2,
    rtol=args.rtol,
    atol=args.atol,
    **kwargs
):  # kwargs passed on to cdeint
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    batch_size = 1024
    lr = 0.00005

    intensity_data = True if model_name in ("odernn", "dt", "decay") else False
    (
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = datasets.speech_commands.get_data(intensity_data, batch_size)
    input_channels = 1 + (1 + intensity_data) * 20
    output_channels = 10
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
            use_intensity=False,
            slope_check=slope_check,
            soft=soft,
            timewise=timewise,
            initial=True,
        )
    # make_model = common.make_model(model_name, input_channels, 10, hidden_channels, hidden_hidden_channels,
    #                                num_hidden_layers, use_intensity=False, initial=True)
    # import pdb ; pdb.set_trace()
    experiments = "speech_commands" + str(manual_seed)
    if model_name in BASELINE_MODELS:

        def new_make_model():
            model, regularise = make_model()
            model.linear.weight.register_hook(lambda grad: 100 * grad)
            model.linear.bias.register_hook(lambda grad: 100 * grad)
            return model, regularise

    else:

        def new_make_model():
            model, regularise1, regularise2 = make_model()
            model.linear.weight.register_hook(lambda grad: 100 * grad)
            model.linear.bias.register_hook(lambda grad: 100 * grad)
            return model, regularise1, regularise2

    name = None if dry_run else "speech_commands"
    num_classes = 10
    if model_name in BASELINE_MODELS:

        return common.main(
            name,
            times,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            new_make_model,
            num_classes,
            max_epochs,
            lr,
            kwargs,
            step_mode=True,
        )
    else:
        # import pdb ; pdb.set_trace()
        return common.main(
            experiments,
            model_name,
            name,
            times,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            new_make_model,
            num_classes,
            max_epochs,
            lr,
            slope_check,
            kwargs,
            step_mode=step_mode,
            c1=args.c1,
            c2=args.c2,
        )


def run_all(device, model_names=("ncde", "odernn", "dt", "decay", "gruode")):
    model_kwargs = dict(
        ncde=dict(hidden_channels=90, hidden_hidden_channels=40, num_hidden_layers=4),
        odernn=dict(
            hidden_channels=128, hidden_hidden_channels=64, num_hidden_layers=4
        ),
        dt=dict(
            hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        decay=dict(
            hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None
        ),
        gruode=dict(
            hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None
        ),
    )
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(device, model_name=model_name, **model_kwargs[model_name])


if __name__ == "__main__":

    main()
