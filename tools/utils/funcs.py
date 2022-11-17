import os
import numpy as np
from inspect import isclass
import torch
import torch.nn.functional as F
import rpy2
from rpy2.robjects import FloatVector, pandas2ri
from rpy2.robjects.packages import importr


def list2np(outputs, targets):
    all_outputs = outputs[0]
    all_targets = targets[0]
    outputs, targets = outputs[1:], targets[1:]
    for output, target in zip(outputs, targets):
        all_outputs = np.vstack((all_outputs, output))
        all_targets = np.vstack((all_targets, target))
    return all_outputs, all_targets


def compute_icc(outputs, targets):

    psych = importr("psych")
    num_class = outputs.shape[1]

    outputs, targets = list2np(outputs, targets)
    if num_class == 1:
        matrix = np.column_stack((outputs, targets)).reshape(-1)
        values = rpy2.robjects.r.matrix(FloatVector(matrix),
                                        ncol=2,
                                        byrow=True)
        icc = psych.ICC(values)
        icc_df = pandas2ri.rpy2py(icc[0])
        total_icc = icc_df.ICC[2]
    else:
        total_icc = []
        for au_idx in range(num_class):
            output = outputs[:, au_idx]
            target = targets[:, au_idx]
            matrix = np.column_stack((output, target)).reshape(-1)
            values = rpy2.robjects.r.matrix(FloatVector(matrix),
                                            ncol=2,
                                            byrow=True)
            icc = psych.ICC(values)
            icc_df = pandas2ri.rpy2py(icc[0])
            au_icc = icc_df.ICC[2]
            total_icc.append(au_icc)
    return total_icc


def compute_mae(outputs, labels):

    num_class = outputs.shape[1]
    outputs, labels = torch.tensor(outputs), torch.tensor(labels)

    if num_class == 1:
        au_mae = F.l1_loss(outputs, labels)
        total_mae = au_mae.item() * 5
    else:
        total_mae = []
        for au_idx in range(num_class):
            output = outputs[:, au_idx]
            label = labels[:, au_idx]
            au_mae = F.l1_loss(output, label)
            total_mae.append(au_mae * 5)

    return total_mae


def record_intensity_metrics(outputs,
                             labels,
                             loss,
                             num_class,
                             savepath,
                             mode='val'):

    labels = np.clip(labels, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    if mode == 'seg':
        icc = compute_icc(outputs, labels)
        mae = compute_mae(outputs, labels)

        return icc, mae
    else:
        labels = labels.reshape(-1, num_class)
        outputs = outputs.reshape(-1, num_class)

        icc = np.array(compute_icc(outputs, labels))
        mae = np.array(compute_mae(outputs, labels))

        res_txt_path = os.path.join(savepath, 'log.txt')
        fp = open(res_txt_path, 'a')
        fp.write("===> loss: {}\n".format(loss))
        fp.write("===> icc: {}\n".format(icc))
        fp.write("===> mae: {}\n".format(mae))
        fp.write("===> average icc: {}\n".format(np.mean(icc)))
        fp.write("===> average mae: {}\n".format(np.mean(mae)))
        fp.close()

        print("icc {} \n mae {}".format(icc, mae))
        print("average icc {}\n".format(np.mean(icc)))
        print("average mae {}\n".format(np.mean(mae)))

        return icc, mae, np.mean(icc), np.mean(mae)


def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x)) and x not in exclude
                 and getattr(module, x) not in exclude])

if __name__ == '__main__':

    outputs = np.array([[5], [2], [4], [6]])
    targets = np.array([[5.3], [2.2], [4.7], [4.6]])
    compute_icc(outputs, targets)
