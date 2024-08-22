from notears import nonlinear, linear
import numpy as np
import pandas as pd
import notears.utils as ut
import torch
import torch.nn as nn
import random
import os
from pprint import pprint
import csv


torch.set_default_dtype(torch.double)
np.set_printoptions(precision=2)


def get_correct_knowledge(d, B_true, W_notears, X, current_list_knowledge_index):
    ## returns indices of correctly classified known inactive and known active edges (not in existing knowledge set)
    mask_index_0, mask_val_0, mask_index_1, mask_val_1 = [], [], [], []
    for i in range(d):
        for j in range(d):
            if (B_true[i,j] == 0 and W_notears[i,j] != 0) \
                    or (B_true[i,j] == 1 and W_notears[i,j] == 0) \
                    or ((i,j) in current_list_knowledge_index):
                pass
            else:
                if B_true[i,j] == 0:
                    mask_index_0.append((i,j))
                    mask_val_0.append(0)
                else:
                    mask_index_1.append((i,j))
                    mask_val_1.append(1)
    return mask_index_0, mask_val_0, mask_index_1, mask_val_1


def get_total_0_1_mistake_0_1(d, B_true, W_notears):
    ## t0- total known inactive, t1- total known active, c0- num mistake inactive, c1- num mistake active
    t0, t1, c0, c1 = 0, 0, 0, 0
    for i in range(d):
        for j in range(d):
            if B_true[i,j] == 0:
                t0 += 1
                if W_notears[i,j] != 0:
                    c0 += 1
            else:
                t1 += 1
                if W_notears[i,j] == 0:
                    c1 += 1
    return t0, t1, c0, c1


def get_mistake_knowledge(d, B_true, W_notears, X, current_list_knowledge_index):
    ## returns indices of misclassified known inactive and known active edges (not in existing knowledge set)
    mask_index_0=[]
    mask_val_0=[]
    mask_index_1=[]
    mask_val_1=[]
    for i in range(d):
        for j in range(d):
            if (B_true[i,j] == 0 and W_notears[i,j] == 0) \
                    or (B_true[i,j] == 1 and W_notears[i,j] != 0) \
                    or ((i,j) in current_list_knowledge_index):
                pass
            else:
                if B_true[i,j] == 0:
                    mask_index_0.append((i,j))
                    mask_val_0.append(0)
                else:
                    mask_index_1.append((i,j))
                    mask_val_1.append(1)
    return mask_index_0, mask_val_0, mask_index_1, mask_val_1


def get_expected_W_notears(W_notears, list_knowledge_index, list_knowledge_val):
    expected_W_notears = W_notears
    for i in range(len(list_knowledge_index)):
        expected_W_notears[list_knowledge_index[i]] = list_knowledge_val[i]
    return expected_W_notears


def count_accuracy_unchecked(B_true, B_est):
    """Compute various accuracy metrics for B_est (where B_est is not necessary a DAG).

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


def run_initial(
        data_type = None, n=None, d=None, s0=None, gt=None, sem=None, mask_num = None, correct_mask_num_0 = None,
        correct_mask_num_1 = None, mask=None, trial_no=None, seed=None, l1l2=None, w_threshold=None, hidden_units=None,
        learned_model=None, dict_data=None, list_dict_data=None, folder_outputs=None, B_true=None, X=None
):
    ## runs the baseline model without any induced knowledge
    PARAMS = {
        'data_type': data_type, 'n': n, 'd': d, 's0': s0, 'gt': gt, 'sem': sem, 'mask_num': mask_num,
        'correct_mask_num_0': correct_mask_num_0, 'correct_mask_num_1': correct_mask_num_1, 'trial_no': trial_no,
        'seed': seed, 'l1': l1l2, 'l2': l1l2, 'w_threshold': w_threshold, 'hidden_units': hidden_units,
    }
    exp_name = str(data_type) + '_' + str(n) + '_' + str(d) + '_' + str(s0) + '_' \
               + str(gt) + '_' + str(sem) + '_' + str(mask_num) + '_' + str(trial_no)
    print(exp_name)
    model = nonlinear.NotearsMLP(
        dims=[PARAMS['d'], PARAMS['hidden_units'], 1], bias=True,
        mask=mask, w_threshold=PARAMS['w_threshold'], learned_model=None
    )
    W_notears, res = nonlinear.notears_nonlinear(model, X, lambda1=PARAMS['l1'], lambda2=PARAMS['l2'])
    learned_model = res['learned_model']
    t0, t1, c0, c1 = get_total_0_1_mistake_0_1(d, B_true, W_notears)

    filename_wnotears = exp_name + '_W_notears.csv'
    np.savetxt(folder_outputs + filename_wnotears, W_notears, delimiter=',')
    local_data = {}
    local_data['data_type'] = PARAMS['data_type']
    local_data['n'] = PARAMS['n']
    local_data['d'] = PARAMS['d']
    local_data['s0'] = PARAMS['s0']
    local_data['gt'] = PARAMS['gt']
    local_data['sem'] = PARAMS['sem']
    local_data['mask_num'] = PARAMS['mask_num']
    local_data['trial_no'] = PARAMS['trial_no']
    local_data['seed'] = PARAMS['seed']
    local_data['l1'] = PARAMS['l1']
    local_data['l2'] = PARAMS['l2']
    local_data['w_threshold'] = PARAMS['w_threshold']
    local_data['hidden_units'] = PARAMS['hidden_units']
    local_data['t0'] = t0
    local_data['t1'] = t1
    local_data['c0'] = c0
    local_data['c1'] = c1
    try:
        assert ut.is_dag(W_notears)
        acc = ut.count_accuracy(B_true, W_notears != 0)
        local_data['status'] = 'success'
        local_data['message'] = '-'
        local_data['fdr'] = acc['fdr']
        local_data['tpr'] = acc['tpr']
        local_data['fpr'] = acc['fpr']
        local_data['shd'] = acc['shd']
        expected_fdr, expected_tpr, expected_fpr, expected_shd = acc['fdr'], acc['tpr'], acc['fpr'], acc['shd']
    except Exception as e:
        local_data['status'] = 'fail'
        local_data['message'] = str(e)
        local_data['fdr'] = '-'
        local_data['tpr'] = '-'
        local_data['fpr'] = '-'
        local_data['shd'] = '-'
        expected_fdr, expected_tpr, expected_fpr, expected_shd = '-', '-', '-', '-'
    local_data['correct_mask_num_0'] = correct_mask_num_0
    local_data['correct_mask_num_1'] = correct_mask_num_1
    local_data['expected_fdr'] = expected_fdr
    local_data['expected_tpr'] = expected_tpr
    local_data['expected_fpr'] = expected_fpr
    local_data['expected_shd'] = expected_shd

    dict_data[exp_name] = local_data
    list_dict_data.append(local_data)
    return B_true, X, W_notears, learned_model, dict_data, list_dict_data


def run_iterative(
    data_type = None, n = None, d = None, s0 = None, gt = None, sem = None, mask_num = None, correct_mask_num_0=None,
    correct_mask_num_1=None, mask = None, trial_no = None, seed = None, l1l2 = None, w_threshold = None,
    hidden_units = None, learned_model = None, dict_data = None, list_dict_data = None, folder_outputs = None,
    B_true = None, X = None, expected_W_notears=None
):
    PARAMS = {
        'data_type': data_type, 'n': n, 'd': d, 's0': s0, 'gt': gt, 'sem': sem, 'mask_num': mask_num,
        'correct_mask_num_0': correct_mask_num_0, 'correct_mask_num_1': correct_mask_num_1, 'trial_no': trial_no,
        'seed': seed, 'l1': l1l2, 'l2': l1l2, 'w_threshold': w_threshold, 'hidden_units': hidden_units,
    }
    exp_name = str(data_type) + '_' + str(n) + '_' + str(d) + '_' + str(s0) + '_' \
               + str(gt) + '_' + str(sem) + '_' + str(mask_num) + '_' + str(trial_no)
    print(exp_name)
    model = nonlinear.NotearsMLP(
        dims=[PARAMS['d'], PARAMS['hidden_units'], 1], bias=True,
        mask=mask, w_threshold=PARAMS['w_threshold'], learned_model=learned_model
    )
    W_notears, res = nonlinear.notears_nonlinear(model, X, lambda1=PARAMS['l1'], lambda2=PARAMS['l2'])
    learned_model = res['learned_model']
    t0, t1, c0, c1 = get_total_0_1_mistake_0_1(d, B_true, W_notears)

    filename_wnotears = exp_name + '_W_notears.csv'
    np.savetxt(folder_outputs + filename_wnotears, W_notears, delimiter=',')
    local_data = {}
    local_data['data_type'] = PARAMS['data_type']
    local_data['n'] = PARAMS['n']
    local_data['d'] = PARAMS['d']
    local_data['s0'] = PARAMS['s0']
    local_data['gt'] = PARAMS['gt']
    local_data['sem'] = PARAMS['sem']
    local_data['mask_num'] = PARAMS['mask_num']
    local_data['trial_no'] = PARAMS['trial_no']
    local_data['seed'] = PARAMS['seed']
    local_data['l1'] = PARAMS['l1']
    local_data['l2'] = PARAMS['l2']
    local_data['w_threshold'] = PARAMS['w_threshold']
    local_data['hidden_units'] = PARAMS['hidden_units']
    local_data['t0'] = t0
    local_data['t1'] = t1
    local_data['c0'] = c0
    local_data['c1'] = c1
    try:
        assert ut.is_dag(W_notears)
        acc = ut.count_accuracy(B_true, W_notears != 0)
        local_data['status'] = 'success'
        local_data['message'] = '-'
        local_data['fdr'] = acc['fdr']
        local_data['tpr'] = acc['tpr']
        local_data['fpr'] = acc['fpr']
        local_data['shd'] = acc['shd']
    except Exception as e:
        local_data['status'] = 'fail'
        local_data['message'] = str(e)
        local_data['fdr'] = '-'
        local_data['tpr'] = '-'
        local_data['fpr'] = '-'
        local_data['shd'] = '-'
    local_data['correct_mask_num_0'] = correct_mask_num_0
    local_data['correct_mask_num_1'] = correct_mask_num_1
    if expected_W_notears is not None:
        expected_acc = count_accuracy_unchecked(B_true, expected_W_notears != 0)
        local_data['expected_fdr'] = expected_acc['fdr']
        local_data['expected_tpr'] = expected_acc['tpr']
        local_data['expected_fpr'] = expected_acc['fpr']
        local_data['expected_shd'] = expected_acc['shd']
    dict_data[exp_name] = local_data
    list_dict_data.append(local_data)
    return B_true, X, W_notears, learned_model, dict_data, list_dict_data


def generate_datasets(
        list_n, list_d, list_s0_factor, list_gt, list_sem, list_data_type, root_folder_path, ntrial
):
    ## generate datasets based on given combinations and num of trials
    list_options = []
    for n in list_n:
        for d in list_d:
            for s0_factor in list_s0_factor:
                for gt in list_gt:
                    for sem in list_sem:
                        for dt in list_data_type:
                            s0 = d * s0_factor
                            o = (n,d,s0,gt,sem,dt)
                            list_options.append(o)
    for o in list_options:
        for i in range(ntrial):
            data_type = o[5]
            B_true = ut.simulate_dag(o[1], o[2], o[3])
            X = ut.simulate_nonlinear_sem(B_true, o[0], o[4])
            subfolder_name = str(data_type) + '_n_d_s0_gt_sem_' + str(o[0]) + '_' + str(o[1]) + '_' + str(o[2]) + '_' \
                             + str(o[3]) + '_' + str(o[4])
            subfolder_path = root_folder_path + subfolder_name
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print('folder created!')
            else:
                print('folder exists!')
            filename_wtrue = str(i) + '_W_true.csv'
            np.savetxt(subfolder_path + '/' + filename_wtrue, B_true, delimiter=',')
            filename_x = str(i) + '_X.csv'
            np.savetxt(subfolder_path + '/' + filename_x, X, delimiter=',')


if __name__=='__main__':


    ## random dataset generation
    list_n, list_d, list_s0_factor, list_gt, list_sem = [200, 1000], [10, 20], [1, 4], ['ER', 'SF'], ['mim']
    list_data_type = ['nonlinear']
    ntrial, seed = 10, 123
    root_folder_path = 'datasets/synthetic/'
    ut.set_random_seed(seed)
    generate_datasets(list_n, list_d, list_s0_factor, list_gt, list_sem, list_data_type, root_folder_path, ntrial)
    print('random dataset generation done!')


    ## iterative knowledge induction for all different combinations with random trials
    list_graph = [
        ('nonlinear', 200, 10, 10, 'ER', 'mim'),
        ('nonlinear', 200, 10, 10, 'SF', 'mim'),
        ('nonlinear', 200, 10, 40, 'ER', 'mim'),
        ('nonlinear', 200, 10, 40, 'SF', 'mim'),

        ('nonlinear', 200, 20, 20, 'ER', 'mim'),
        ('nonlinear', 200, 20, 20, 'SF', 'mim'),
        ('nonlinear', 200, 20, 80, 'ER', 'mim'),
        ('nonlinear', 200, 20, 80, 'SF', 'mim'),

        ('nonlinear', 1000, 10, 10, 'ER', 'mim'),
        ('nonlinear', 1000, 10, 10, 'SF', 'mim'),
        ('nonlinear', 1000, 10, 40, 'ER', 'mim'),
        ('nonlinear', 1000, 10, 40, 'SF', 'mim'),

        ('nonlinear', 1000, 20, 20, 'ER', 'mim'),
        ('nonlinear', 1000, 20, 20, 'SF', 'mim'),
        ('nonlinear', 1000, 20, 80, 'ER', 'mim'),
        ('nonlinear', 1000, 20, 80, 'SF', 'mim')
    ]
    for g in list_graph:
        ## data_type, n, d, s0, gt, sem = 'nonlinear', 200, 10, 10, 'ER', 'mim'
        data_type, n, d, s0, gt, sem = g[0], g[1], g[2], g[3], g[4], g[5]

        l1l2, seed, w_threshold, hidden_units, ntrial = 0.01, 123, 0.3, 10, 10
        dict_data, list_dict_data = {}, []

        root_folder_path = 'datasets/synthetic/'
        ut.set_random_seed(seed)
        subfolder_name = str(data_type) + '_n_d_s0_gt_sem_' + str(n) + '_' + str(d) + '_' + str(s0) + '_' \
                         + str(gt) + '_' + str(sem)
        subfolder_path = root_folder_path + subfolder_name

        knowledge_source = 0  # 0: mistake, 1: (both) correct & mistake, 2: correctly classified set
        knowledge_type = 0  # 0: known inactive, 1: known active, 2: (both) known inactive & active
        folder_outputs = str('outputs/source_') + str(knowledge_source) + '_type_' + str(knowledge_type) + '/'
        folder_results = str('results/source_') + str(knowledge_source) + '_type_' + str(knowledge_type) + '/'
        if not os.path.exists(folder_outputs):
            os.makedirs(folder_outputs)
            print('output folder created!')
        else:
            print('output folder exists!')
        if not os.path.exists(folder_results):
            os.makedirs(folder_results)
            print('result folder created!')
        else:
            print('result folder exists!')

        for trial_no in range(ntrial):
            B_true = pd.read_csv(subfolder_path + '/' + str(trial_no) + '_W_true.csv', sep=',', header=None).values
            X = pd.read_csv(subfolder_path + '/' + str(trial_no) + '_X.csv', sep=',', header=None).values

            mask = np.ones((d, d)) * np.nan
            mask_num = 0
            correct_mask_num_0, correct_mask_num_1 = 0, 0
            B_true, X, W_notears, learned_model, dict_data, list_dict_data = run_initial(
                data_type=data_type, n=n, d=d, s0=s0, gt=gt, sem=sem, mask_num=mask_num,
                correct_mask_num_0=correct_mask_num_0, correct_mask_num_1=correct_mask_num_1, mask=mask,
                trial_no=trial_no, seed=seed, l1l2=l1l2, w_threshold=w_threshold, hidden_units=hidden_units,
                learned_model=None, dict_data=dict_data, list_dict_data= list_dict_data, folder_outputs=folder_outputs,
                B_true=B_true, X=X
            )
            if knowledge_source == 0:
                mi0, mv0, mi1, mv1 = get_mistake_knowledge(d, B_true, W_notears, X, [])

                if knowledge_type == 0:
                    len_mv0 = len(mv0)
                    list_knowledge_index = []
                    list_knowledge_val = []
                    limit_x_axis = 10
                    count_x_axis = 0
                    while (len_mv0 > 0 and count_x_axis < len_mv0 and count_x_axis < limit_x_axis):
                        r = random.randint(0, len_mv0 - 1)
                        knowledge_index = mi0[r]
                        knowledge_val = mv0[r]
                        while knowledge_index in list_knowledge_index:
                            r = random.randint(0, len_mv0 - 1)
                            knowledge_index = mi0[r]
                            knowledge_val = mv0[r]
                        list_knowledge_index.append(knowledge_index)
                        list_knowledge_val.append(knowledge_val)

                        expected_W_notears = get_expected_W_notears(W_notears, list_knowledge_index, list_knowledge_val)

                        mask_num = 0
                        mask = np.ones((d, d)) * np.nan
                        for i in range(len(list_knowledge_index)):
                            mask[list_knowledge_index[i]] = list_knowledge_val[i]
                            mask_num += 1
                        B_true, X, W_notears, learned_model, dict_data, list_dict_data = run_iterative(
                            data_type=data_type, n=n, d=d, s0=s0, gt=gt, sem=sem, mask_num=mask_num,
                            correct_mask_num_0=correct_mask_num_0, correct_mask_num_1=correct_mask_num_1, mask=mask,
                            trial_no=trial_no, seed=seed, l1l2=l1l2, w_threshold=w_threshold, hidden_units=hidden_units,
                            learned_model=None, dict_data=dict_data, list_dict_data=list_dict_data,
                            folder_outputs=folder_outputs, B_true=B_true, X=X, expected_W_notears=expected_W_notears,
                        )
                        mi0, mv0, mi1, mv1 = get_mistake_knowledge(d, B_true, W_notears, X, list_knowledge_index)
                        len_mv0 = len(mv0)
                        count_x_axis +=1
                elif knowledge_type == 1:
                    len_mv1 = len(mv1)
                    list_knowledge_index = []
                    list_knowledge_val = []
                    limit_x_axis = 10
                    count_x_axis = 0
                    while (len_mv1 > 0 and count_x_axis < len_mv1 and count_x_axis < limit_x_axis):
                        r = random.randint(0, len_mv1 - 1)
                        knowledge_index = mi1[r]
                        knowledge_val = mv1[r]
                        while knowledge_index in list_knowledge_index:
                            r = random.randint(0, len_mv1 - 1)
                            knowledge_index = mi1[r]
                            knowledge_val = mv1[r]
                        list_knowledge_index.append(knowledge_index)
                        list_knowledge_val.append(knowledge_val)

                        expected_W_notears = get_expected_W_notears(W_notears, list_knowledge_index, list_knowledge_val)

                        mask_num = 0
                        mask = np.ones((d, d)) * np.nan
                        for i in range(len(list_knowledge_index)):
                            mask[list_knowledge_index[i]] = list_knowledge_val[i]
                            mask_num += 1
                        B_true, X, W_notears, learned_model, dict_data, list_dict_data = run_iterative(
                            data_type=data_type, n=n, d=d, s0=s0, gt=gt, sem=sem, mask_num=mask_num,
                            correct_mask_num_0=correct_mask_num_0, correct_mask_num_1=correct_mask_num_1, mask=mask,
                            trial_no=trial_no, seed=seed, l1l2=l1l2, w_threshold=w_threshold, hidden_units=hidden_units,
                            learned_model=None, dict_data=dict_data, list_dict_data=list_dict_data,
                            folder_outputs=folder_outputs, B_true=B_true, X=X, expected_W_notears=expected_W_notears,
                        )
                        mi0, mv0, mi1, mv1 = get_mistake_knowledge(d, B_true, W_notears, X, list_knowledge_index)
                        len_mv1 = len(mv1)
                        count_x_axis +=1

            elif knowledge_source == 2:
                mi0, mv0, mi1, mv1 = get_correct_knowledge(d, B_true, W_notears, X, [])
                mi01 = mi0 + mi1
                mv01 = mv0 + mv1
                if knowledge_type == 2:
                    len_mv01 = len(mv01)
                    list_knowledge_index = []
                    list_knowledge_val = []
                    limit_x_axis = 10
                    count_x_axis = 0
                    while (len_mv01 > 0 and count_x_axis < len_mv01 and count_x_axis < limit_x_axis):
                        r = random.randint(0, len_mv01 - 1)
                        knowledge_index = mi01[r]
                        knowledge_val = mv01[r]
                        while knowledge_index in list_knowledge_index:
                            r = random.randint(0, len_mv01 - 1)
                            knowledge_index = mi01[r]
                            knowledge_val = mv01[r]
                        list_knowledge_index.append(knowledge_index)
                        list_knowledge_val.append(knowledge_val)
                        mask_num = 0
                        mask = np.ones((d, d)) * np.nan
                        for i in range(len(list_knowledge_index)):
                            mask[list_knowledge_index[i]] = list_knowledge_val[i]
                            mask_num += 1
                        B_true, X, W_notears, learned_model, dict_data, list_dict_data = run_iterative(
                            data_type=data_type, n=n, d=d, s0=s0, gt=gt, sem=sem, mask_num=mask_num, mask=mask,
                            trial_no=trial_no, seed=seed, l1l2=l1l2, w_threshold=w_threshold, hidden_units=hidden_units,
                            learned_model=None, dict_data=dict_data, list_dict_data=list_dict_data,
                            folder_outputs=folder_outputs, B_true=B_true, X=X
                        )
                        mi0, mv0, mi1, mv1 = get_correct_knowledge(d, B_true, W_notears, X, list_knowledge_index)
                        mi01 = mi0 + mi1
                        mv01 = mv0 + mv1
                        len_mv01 = len(mv01)
                        count_x_axis += 1


        filename_result = str(subfolder_name) + '.csv'
        field_names = [
            'data_type', 'n', 'd', 's0', 'gt', 'sem', 'mask_num', 'trial_no',
            'fdr', 'tpr', 'fpr', 'shd', 't0', 't1', 'c0', 'c1',
            'seed', 'l1', 'l2', 'w_threshold', 'hidden_units', 'status', 'message',
            'correct_mask_num_0', 'correct_mask_num_1',
            'expected_fdr', 'expected_tpr', 'expected_fpr', 'expected_shd',
        ]
        with open(folder_results + str(filename_result), 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(list_dict_data)
