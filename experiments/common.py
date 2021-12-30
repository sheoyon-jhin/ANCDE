from http.client import MOVED_PERMANENTLY

# from NeuralCDE.experiments.uea_attentive import BASELINE_MODELS
import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm

BASELINE_MODELS = [
    "ncde",
    "odernn",
    "dt",
    "decay",
    "gruode",
    "odernn_forecasting",
    "ncde_forecasting",
    "decay_forecasting",
    "dt_forecasting",
    "gruode_forecasting",
    "double_ncde_new6",
]
BASELINE_MODELS_F = [
    "ncde",
    "odernn",
    "dt",
    "decay",
    "gruode",
    "odernn_forecasting",
    "decay_forecasting",
    "dt_forecasting",
]
NCDE_BASELINES = ["gruode_forecasting", "ncde_forecasting"]
LEARNABLEPATH = ["Learnable_Path"]
import time_dataset
import models
from models import metamodel

PATH = "/home/bigdyl/socar/NeuralCDE/experiments/trained_model/"
here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.03):
    def new_loss_fn(pred_y, true_y):

        total_loss = loss_fn(pred_y, true_y)

        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()

        return total_loss

    return new_loss_fn


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):

        return self.model(*args, **kwargs).squeeze(-1)  # TODO : only for Learnable Path


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics(
    model_name, dataloader, model, times, loss_fn, num_classes, slope, device, kwargs
):

    with torch.no_grad():
        total_accuracy = 0
        total_confusion = torch.zeros(
            num_classes, num_classes
        ).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        mse_loss = 0
        h_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:

            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)
            if model_name in LEARNABLEPATH:
                pred_y, loss_1, loss_2 = model(times, coeffs, lengths, slope, **kwargs)
            else:
                pred_y = model(times, coeffs, lengths, slope, **kwargs)
            if len(pred_y.shape) == 2:
                pred_y = pred_y.squeeze(-1)
            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()
            if num_classes == 2:
                # Assume that our datasets aren't so large that this breaks
                true_y_cpus.append(true_y_cpu)
                pred_y_cpus.append(pred_y_cpu)
            thresholded_y_cpu = thresholded_y.detach().cpu()

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_confusion += sklearn.metrics.confusion_matrix(
                true_y_cpu, thresholded_y_cpu, labels=range(num_classes)
            )
            total_dataset_size += batch_size
            loss_task = loss_fn(pred_y, true_y)
            total_loss += loss_task * batch_size
            if model_name in LEARNABLEPATH:

                mse_loss += loss_1 * batch_size
                h_loss += loss_2 * batch_size

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        if model_name in LEARNABLEPATH:
            mse_loss /= total_dataset_size
            h_loss /= total_dataset_size
            metrics = _AttrDict(
                accuracy=total_accuracy.item(),
                confusion=total_confusion,
                dataset_size=total_dataset_size,
                loss=total_loss.item(),
                mse_loss=mse_loss.item(),
                h_loss=h_loss.item(),
            )

        else:
            metrics = _AttrDict(
                accuracy=total_accuracy.item(),
                confusion=total_confusion,
                dataset_size=total_dataset_size,
                loss=total_loss.item(),
            )

        if num_classes == 2:

            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)

        return metrics


def _evaluate_metrics_forecasting(
    model_name, dataloader, model, times, loss_fn, num_classes, slope, device, kwargs
):

    with torch.no_grad():
        total_accuracy = 0
        total_confusion = torch.zeros(
            num_classes, num_classes
        ).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:

            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            if model_name in NCDE_BASELINES:
                pred_y = model(times, coeffs, lengths, stream=True, **kwargs)
            elif model_name in BASELINE_MODELS_F:
                pred_y = model(times, coeffs, lengths, **kwargs)
            else:
                pred_y = model(times, coeffs, lengths, slope, stream=True, **kwargs)

            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()

            total_dataset_size += batch_size

            total_loss += loss_fn(pred_y, true_y.float()) * batch_size

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        metrics = _AttrDict(dataset_size=total_dataset_size, loss=total_loss.item())

        return metrics


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write("Caught AssertionError: " + str(exc_val))
            return True


def _train_loop_forecasting(
    model_name,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model,
    times,
    optimizer,
    loss_fn,
    max_epochs,
    num_classes,
    slope_check,
    device,
    kwargs,
    step_mode,
):

    model.train()
    best_model = model
    best_train_loss = math.inf
    best_val_loss = math.inf
    best_test_loss = math.inf
    best_train_loss_epoch = 0
    best_val_loss_epoch = 0

    history = []
    breaking = False
    if step_mode == "trainloss":
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    elif step_mode == "valloss":
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    elif step_mode == "none":
        epoch_per_metric = 1
        plateau_terminate = 50

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write("Starting training for model:\n\n" + str(model) + "\n\n")
    for epoch in tqdm_range:

        if slope_check:
            slope = (epoch * 0.12) + 1.0
        else:
            slope = 0.0
        if breaking:
            break
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):

                *train_coeffs, train_y, lengths = batch

                if model_name in NCDE_BASELINES:
                    pred_y = model(times, train_coeffs, lengths, stream=True, **kwargs)

                elif model_name in BASELINE_MODELS_F:

                    pred_y = model(times, train_coeffs, lengths, **kwargs)

                else:
                    pred_y = model(
                        times, train_coeffs, lengths, slope, stream=True, **kwargs
                    )

                # import pdb ; pdb.set_trace()
                loss = loss_fn(pred_y, train_y.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            train_metrics = _evaluate_metrics_forecasting(
                model_name,
                train_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )
            val_metrics = _evaluate_metrics_forecasting(
                model_name,
                val_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )
            test_metrics = _evaluate_metrics_forecasting(
                model_name,
                test_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )

            model.train()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                del best_model  # so that we don't have three copies of a model simultaneously
                best_model = copy.deepcopy(model)

            if val_metrics.loss * 1.0001 < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_loss_epoch = epoch
                ### TODO : [CHANGE ME ]
                del best_model
                print(
                    f"\n [Epoch : {epoch}] best validation : {val_metrics.loss} Test Loss : {test_metrics.loss} "
                )
                best_model = copy.deepcopy(model)
                # ckpt_file = PATH+"at"+str(epoch)+"_model_loss"+str(test_metrics.loss)+".pth"
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': best_model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss
                #     }, ckpt_file)
                # so that we don't have three copies of a model simultaneously

            tqdm_range.write(
                "Epoch: {}  Train MSE loss: {:.3}  "
                "Val MSE loss: {:.3} Test MSE loss {:.3} slope: {:.5}  "
                "".format(
                    epoch,
                    train_metrics.loss,
                    val_metrics.loss,
                    test_metrics.loss,
                    slope,
                )
            )

            if step_mode == "trainloss":
                scheduler.step(train_metrics.loss)
            elif step_mode == "valloss":
                scheduler.step(val_metrics.loss)
            elif step_mode == "valaccuracy":
                scheduler.step(val_metrics.accuracy)
            elif step_mode == "valauc":
                scheduler.step(val_metrics.auroc)

            history.append(
                _AttrDict(
                    epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics
                )
            )

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write(
                    "Breaking because of no improvement in training loss for {} epochs."
                    "".format(plateau_terminate)
                )
                breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history, epoch


def _train_loop(
    model_name,
    experiments,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model,
    times,
    optimizer,
    loss_fn,
    max_epochs,
    num_classes,
    slope_check,
    device,
    kwargs,
    step_mode,
    c1,
    c2,
):

    model.train()
    best_model = model
    best_train_loss = math.inf
    best_train_accuracy = 0
    best_val_accuracy = 0
    best_train_auc = 0
    best_val_auc = 0
    best_test_auc = 0
    best_train_accuracy_epoch = 0
    best_train_loss_epoch = 0
    best_train_auc_epoch = 0

    history = []
    breaking = False
    if step_mode == "trainloss":
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    elif step_mode == "valloss":
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    elif step_mode == "valaccuracy":
        print("valaccuracy")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, mode="max"
        )

    elif step_mode == "valauc":
        print("valauc")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, mode="max"
        )

    elif step_mode == "none":
        epoch_per_metric = 1
        plateau_terminate = 50

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write("Starting training for model:\n\n" + str(model) + "\n\n")
    for epoch in tqdm_range:

        if slope_check:
            slope = (epoch * 0.12) + 1.0
        else:
            slope = 0.0
        if breaking:
            break
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                *train_coeffs, train_y, lengths = batch

                if model_name in LEARNABLEPATH:
                    pred_y, loss_1, loss_2 = model(
                        times, train_coeffs, lengths, slope, **kwargs
                    )

                    if len(pred_y.shape) == 2:
                        pred_y = pred_y.squeeze(-1)

                    loss_h = c1 * loss_1 + c2 * loss_2
                    loss = loss_fn(pred_y, train_y)
                    final_loss = loss + loss_h
                    final_loss.backward()
                else:
                    pred_y = model(times, train_coeffs, lengths, slope, **kwargs)
                    # import pdb ; pdb.set_trace()
                    if len(pred_y.shape) == 2:
                        pred_y = pred_y.squeeze(-1)
                    loss = loss_fn(pred_y, train_y)
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            train_metrics = _evaluate_metrics(
                model_name,
                train_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )
            val_metrics = _evaluate_metrics(
                model_name,
                val_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )
            test_metrics = _evaluate_metrics(
                model_name,
                test_dataloader,
                model,
                times,
                loss_fn,
                num_classes,
                slope,
                device,
                kwargs,
            )

            model.train()

            # import pdb ; pdb.set_trace()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch

            if train_metrics.accuracy > best_train_accuracy * 1.001:
                best_train_accuracy = train_metrics.accuracy
                best_train_accuracy_epoch = epoch

            if num_classes == 2:
                if train_metrics.auroc > best_train_auc * 1.001:
                    best_train_auc = train_metrics.auroc
                    best_train_auc_epoch = epoch
                tqdm_range.write(
                    "Epoch: {}  Train loss: {:.3}  Train AUC : {:.3}  "
                    "Val loss: {:.3} Val AUC : {:.3} Test loss : {:.3} Test AUC :{:.3} slope: {:.5}  "
                    "".format(
                        epoch,
                        train_metrics.loss,
                        train_metrics.auroc,
                        val_metrics.loss,
                        val_metrics.auroc,
                        test_metrics.loss,
                        test_metrics.auroc,
                        slope,
                    )
                )

                if val_metrics.auroc > best_val_auc:
                    best_val_auc = val_metrics.auroc
                    del best_model
                    best_model = copy.deepcopy(model)

                    tqdm_range.write(
                        "[Epoch: {}]  Test loss: {:.3}  Test AUC : {:.3}  slope: {:.5}  "
                        "".format(epoch, test_metrics.loss, test_metrics.auroc, slope)
                    )

            else:
                tqdm_range.write(
                    "Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  "
                    "Val loss: {:.3}  Val accuracy: {:.3} Test loss : {:.3} Test accuracy : {:.3}  "
                    "".format(
                        epoch,
                        train_metrics.loss,
                        train_metrics.accuracy,
                        val_metrics.loss,
                        val_metrics.accuracy,
                        test_metrics.loss,
                        test_metrics.accuracy,
                    )
                )

                if val_metrics.accuracy > best_val_accuracy:
                    best_val_accuracy = val_metrics.accuracy
                    del best_model  # so that we don't have three copies of a model simultaneously
                    best_model = copy.deepcopy(model)
                    print(
                        f"\n[ Epoch {epoch} ] Test Loss : {test_metrics.loss},  Test accuracy: {test_metrics.accuracy}"
                    )

            if step_mode == "trainloss":
                scheduler.step(train_metrics.loss)
            elif step_mode == "valloss":
                scheduler.step(val_metrics.loss)
            elif step_mode == "valaccuracy":
                scheduler.step(val_metrics.accuracy)
            elif step_mode == "valauc":
                scheduler.step(val_metrics.auroc)

            history.append(
                _AttrDict(
                    epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics
                )
            )
            if num_classes > 2:
                if epoch > best_train_loss_epoch + plateau_terminate:
                    tqdm_range.write(
                        "Breaking because of no improvement in training loss for {} epochs."
                        "".format(plateau_terminate)
                    )
                    breaking = True
                if epoch > best_train_accuracy_epoch + plateau_terminate:
                    tqdm_range.write(
                        "Breaking because of no improvement in training accuracy for {} epochs."
                        "".format(plateau_terminate)
                    )
                    breaking = True
            else:
                if epoch > best_train_loss_epoch + plateau_terminate:
                    tqdm_range.write(
                        "Breaking because of no improvement in training loss for {} epochs."
                        "".format(plateau_terminate)
                    )
                    breaking = True
                if epoch > best_train_auc_epoch + plateau_terminate:
                    tqdm_range.write(
                        "Breaking because of no improvement in training auc for {} epochs."
                        "".format(plateau_terminate)
                    )
                    breaking = True
            # if epoch > best_train_loss_epoch + plateau_terminate:
            #     tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
            #                      ''.format(plateau_terminate))
            #     breaking = True
            # if epoch > best_train_accuracy_epoch + plateau_terminate:
            #     tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
            #                      ''.format(plateau_terminate))
            #     breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history, epoch


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / "results" / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save["train_dataloader"]
    del result_to_save["val_dataloader"]
    del result_to_save["test_dataloader"]
    result_to_save["model"] = str(result_to_save["model"])

    num += 1
    with open(loc / str(num), "w") as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)


def main(
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
    step_mode,
    c1,
    c2,
    pos_weight=torch.tensor(1),
):

    times = times.to(device)
    if device != "cpu":
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None
    # model, regularise_parameters = make_model()
    model, regularise_parameters, regularise_parameters2 = make_model()
    if num_classes == 2:
        # model = _SqueezeEnd(model)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.functional.cross_entropy
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    # loss_fn = _add_weight_regularisation2(loss_fn, regularise_parameters,regularise_parameters2)
    model.to(device)
    print(f"\nparameter of Neural CDE {_count_parameters(model)}")
    print(
        f"\nparameter of ORIGINAL ODE FUNC {_count_parameters(regularise_parameters)}"
    )
    print(
        f"\nparameter of ATTENTION ODE FUNC {_count_parameters(regularise_parameters2)}"
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history, epoch = _train_loop(
        model_name,
        experiments,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        times,
        optimizer,
        loss_fn,
        max_epochs,
        num_classes,
        slope_check,
        device,
        kwargs,
        step_mode,
        c1,
        c2,
    )

    model.eval()
    if slope_check:
        slope = (epoch * 0.12) + 1.0
    else:
        slope = 0.0
    train_metrics = _evaluate_metrics(
        train_dataloader, model, times, loss_fn, num_classes, slope, device, kwargs
    )
    val_metrics = _evaluate_metrics(
        val_dataloader, model, times, loss_fn, num_classes, slope, device, kwargs
    )
    test_metrics = _evaluate_metrics(
        test_dataloader, model, times, loss_fn, num_classes, slope, device, kwargs
    )
    if num_classes == 2:
        print(
            "Test AUC : {} Test Loss : {}".format(test_metrics.auroc, test_metrics.loss)
        )
        ckpt_final = (
            PATH + str(experiments) + "_" + str(test_metrics.auroc) + "_fianl_model.pth"
        )
    else:
        print(
            "Final test Metrics : {} Test Loss : {}".format(
                test_metrics.accuracy, test_metrics.loss
            )
        )
        ckpt_final = (
            PATH
            + str(experiments)
            + "_"
            + str(test_metrics.accuracy)
            + "_fianl_model.pth"
        )

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),}, ckpt_final)
    if device != "cpu":
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None

    result = _AttrDict(
        times=times,
        memory_usage=memory_usage,
        baseline_memory=baseline_memory,
        num_classes=num_classes,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model.to("cpu"),
        parameters=_count_parameters(model),
        history=history,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    if name is not None:
        _save_results(name, result)
    return result


def main_forecasting(
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
    step_mode,
    pos_weight=torch.tensor(1),
):

    times = times.to(device)
    if device != "cpu":
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None
    if model_name in BASELINE_MODELS:

        model, regularise_parameters = make_model()

    else:
        model, regularise_parameters, regularise_parameters2 = make_model()

    loss_fn = torch.nn.functional.mse_loss
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    # loss_fn = _add_weight_regularisation2(loss_fn, regularise_parameters,regularise_parameters2)
    model.to(device)

    if model_name in BASELINE_MODELS:
        print(f"\nparameter of Neural CDE {_count_parameters(model)}")
        print(
            f"\nparameter of ORIGINAL ODE FUNC {_count_parameters(regularise_parameters)}"
        )

    else:
        print(f"\nparameter of Neural CDE {_count_parameters(model)}")
        print(
            f"\nparameter of ORIGINAL ODE FUNC {_count_parameters(regularise_parameters)}"
        )
        print(
            f"\nparameter of ATTENTION ODE FUNC {_count_parameters(regularise_parameters2)}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, epoch = _train_loop_forecasting(
        model_name,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        times,
        optimizer,
        loss_fn,
        max_epochs,
        num_classes,
        slope_check,
        device,
        kwargs,
        step_mode,
    )

    model.eval()

    if slope_check:
        slope = (epoch * 0.12) + 1.0
    else:
        slope = 0.0

    train_metrics = _evaluate_metrics_forecasting(
        model_name,
        train_dataloader,
        model,
        times,
        loss_fn,
        num_classes,
        slope,
        device,
        kwargs,
    )
    val_metrics = _evaluate_metrics_forecasting(
        model_name,
        val_dataloader,
        model,
        times,
        loss_fn,
        num_classes,
        slope,
        device,
        kwargs,
    )
    test_metrics = _evaluate_metrics_forecasting(
        model_name,
        test_dataloader,
        model,
        times,
        loss_fn,
        num_classes,
        slope,
        device,
        kwargs,
    )
    ckpt_final = (
        PATH + str(experiments) + "_" + str(test_metrics.auroc) + "_fianl_model.pth"
    )
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),}, ckpt_final)
    if device != "cpu":
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None

    result = _AttrDict(
        times=times,
        memory_usage=memory_usage,
        baseline_memory=baseline_memory,
        num_classes=num_classes,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model.to("cpu"),
        parameters=_count_parameters(model),
        history=history,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    if name is not None:
        _save_results(name, result)
    return result


def make_model(
    name,
    input_channels,
    output_channels,
    hidden_channels,
    hidden_hidden_channels,
    attention_channel,
    attention_attention_channel,
    num_hidden_layers,
    use_intensity,
    slope_check,
    soft,
    timewise,
    rtol,
    atol,
    file,
    initial,
):

    if name == "ncde":

        def make_model():
            vector_field = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.NeuralCDE(
                func=vector_field,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
            )
            return model, vector_field

    elif name == "double_ncde":

        def make_model():
            vector_field_f = models.FinalTanh_ff(
                input_channels=input_channels,
                hidden_atten_channels=attention_channel,
                hidden_hidden_atten_channels=attention_attention_channel,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.DoubleNeuralCDE(
                func=vector_field_f,
                func_g=vector_field_g,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                attention_channel=attention_channel,
                slope_check=slope_check,
                soft=soft,
                timewise=timewise,
                initial=initial,
            )
            return model, vector_field_g

    elif name == "double_ncde_new":

        def make_model():
            vector_field_f = models.FinalTanh_ff(
                input_channels=input_channels,
                hidden_atten_channels=attention_channel,
                hidden_hidden_atten_channels=attention_attention_channel,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.DoubleNeuralCDE_debug(
                func=vector_field_f,
                func_g=vector_field_g,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                attention_channel=attention_channel,
                slope_check=slope_check,
                soft=soft,
                timewise=timewise,
                initial=initial,
            )
            return model, vector_field_g, vector_field_f

    elif name == "double_ncde_socar":

        def make_model():
            vector_field_f = models.FinalTanh_socar(
                input_channels=input_channels,
                hidden_atten_channels=attention_channel,
                hidden_hidden_atten_channels=attention_attention_channel,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.DoubleNeuralCDE_socar(
                func=vector_field_f,
                func_g=vector_field_g,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                attention_channel=attention_channel,
                slope_check=slope_check,
                soft=soft,
                timewise=timewise,
                initial=initial,
            )
            return model, vector_field_g, vector_field_f

    elif name == "ancde":

        def make_model():
            vector_field_f = models.FinalTanh_ff6(
                input_channels=input_channels,
                hidden_atten_channels=attention_channel,
                hidden_hidden_atten_channels=attention_attention_channel,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.ANCDE(
                func=vector_field_f,
                func_g=vector_field_g,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                attention_channel=attention_channel,
                slope_check=slope_check,
                soft=soft,
                timewise=timewise,
                file=file,
                initial=initial,
            )
            return model, vector_field_g, vector_field_f

    elif name == "Learnable_Path":

        def make_model():

            vector_field_f = models.FinalTanh_f(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh_g(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_hide = models.FinalTanh_hide(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.Learnable_Path(
                func=vector_field_f,
                func_g=vector_field_g,
                func_h=vector_field_hide,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
                rtol=rtol,
                atol=atol,
            )
            return model, vector_field_g, vector_field_f

    elif name == "ancde_forecasting":

        def make_model():
            output_time = 1
            vector_field_f = models.FinalTanh_ff6(
                input_channels=input_channels,
                hidden_atten_channels=attention_channel,
                hidden_hidden_atten_channels=attention_attention_channel,
                num_hidden_layers=num_hidden_layers,
            )
            vector_field_g = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.ANCDE_forecasting(
                func=vector_field_f,
                func_g=vector_field_g,
                input_channels=input_channels,
                output_time=output_time,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                attention_channel=attention_channel,
                slope_check=slope_check,
                soft=soft,
                timewise=timewise,
                file=file,
                initial=initial,
            )

            return model, vector_field_g, vector_field_f

    elif name == "ncde_forecasting":

        def make_model():
            vector_field = models.FinalTanh(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            model = models.NeuralCDE_forecasting(
                func=vector_field,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
            )
            return model, vector_field

    elif name == "gruode":

        def make_model():
            vector_field = models.GRU_ODE(
                input_channels=input_channels, hidden_channels=hidden_channels
            )
            model = models.NeuralCDE(
                func=vector_field,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
            )
            return model, vector_field

    elif name == "gruode_forecasting":

        def make_model():
            vector_field = models.GRU_ODE(
                input_channels=input_channels, hidden_channels=hidden_channels
            )
            model = models.NeuralCDE_forecasting(
                func=vector_field,
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
            )
            return model, vector_field

    elif name == "dt":

        def make_model():
            model = models.GRU_dt(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    elif name == "decay":

        def make_model():
            model = models.GRU_D(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    elif name == "odernn":

        def make_model():
            model = models.ODERNN(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    elif name == "dt_forecasting":

        def make_model():
            model = models.GRU_dt_forecasting(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    elif name == "decay_forecasting":

        def make_model():
            model = models.GRU_D_forecasting(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    elif name == "odernn_forecasting":

        def make_model():

            model = models.ODERNN_forecasting(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                output_channels=output_channels,
                use_intensity=use_intensity,
            )
            return model, model

    else:
        raise ValueError(
            "Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
            "".format(name)
        )
    return make_model
