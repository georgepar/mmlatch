import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    f_score = f1_score((test_preds >= 0), (test_truth >= 0), average="weighted")
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    f_score_neg = f1_score((test_preds > 0), (test_truth > 0), average="weighted")
    binary_truth_neg = test_truth > 0
    binary_preds_neg = test_preds > 0

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    f_score_non_zero = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_non_zero = test_truth[non_zeros] > 0
    binary_preds_non_zero = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1_pos": f_score,  # zeros are positive
        "bin_acc_pos": accuracy_score(binary_truth, binary_preds),  # zeros are positive
        "f1_neg": f_score_neg,  # zeros are negative
        "bin_acc_neg": accuracy_score(
            binary_truth_neg, binary_preds_neg
        ),  # zeros are negative
        "f1": f_score_non_zero,  # zeros are excluded
        "bin_acc": accuracy_score(
            binary_truth_non_zero, binary_preds_non_zero
        ),  # zeros are excluded
    }


def eval_mosei_senti_old(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] >= 0), (test_truth[non_zeros] >= 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] >= 0
    binary_preds = test_preds[non_zeros] >= 0
    f_score_neg = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_neg = test_truth[non_zeros] > 0
    binary_preds_neg = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1": f_score,
        "f1_neg": f_score_neg,
        "bin_acc_neg": accuracy_score(binary_truth_neg, binary_preds_neg),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print("{}:\t{}".format(k, v))


def save_metrics(metrics, results_file):
    with open(results_file, "w") as fd:
        for k, v in metrics.items():
            print("{}:\t{}".format(k, v), file=fd)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["neutral", "happy", "sad", "angry"]
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()

    results = {}
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
        acc = accuracy_score(test_truth_i, test_preds_i)
        results["{}_acc".format(emos[emo_ind])] = acc
        results["{}_f1".format(emos[emo_ind])] = f1

    return results
