#------------------------------------------------------#
#------------------ DATASET ---------------------------#
#------------------------------------------------------#
# dataset characteristics := d=11, n=7466
# features := praf, pmek, plcg, PIP2, PIP3, (p44/42==erk), pacts473, PKA, PKC, p38, pjnk
# adjacency matrix := [
#     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#     [0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
#     [1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
# ]


from notears import nonlinear
import numpy as np
import pandas as pd
import notears.utils as ut
import torch
import random
import os
import csv


torch.set_default_dtype(torch.double)
np.set_printoptions(precision=2)


def get_mask_0_1(d,B_true,W_notears,X, current_list_knowledge_index):
    ## get mistakes on inactive and active edge types
    mask_index_0=[]
    mask_val_0=[]
    mask_index_1=[]
    mask_val_1=[]
    for i in range(d):
        for j in range(d):
            if B_true[i,j] == 0 and W_notears[i,j] != 0 and (i,j) not in current_list_knowledge_index:
                mask_index_0.append((i,j))
                mask_val_0.append(0)
            if B_true[i,j] == 1 and W_notears[i,j] == 0 and (i,j) not in current_list_knowledge_index:
                mask_index_1.append((i,j))
                mask_val_1.append(1)
    return mask_index_0, mask_val_0, mask_index_1, mask_val_1


def get_mask_index_mask_val(d, B_true, W_notears, X):
    ## get performance metrics
    total_predicted_1 = 0
    total_agree_1 = 0  # agree with consensus
    total_reversed_1 = 0  # edge detected but reversed
    mask_index = []
    mask_val = []
    for i in range(d):
        for j in range(d):
            if B_true[i, j] == 0 and W_notears[i, j] == 0:
                pass
            if B_true[i, j] == 0 and W_notears[i, j] != 0:
                mask_index.append((i, j))
                mask_val.append(0)
                total_predicted_1 += 1
            if B_true[i, j] == 1 and W_notears[i, j] == 0:
                mask_index.append((i, j))
                mask_val.append(1)
            if B_true[i, j] == 1 and W_notears[i, j] != 0:
                total_agree_1 += 1
                total_predicted_1 += 1
            if B_true[i, j] == 1 and W_notears[j, i] != 0:
                total_reversed_1 += 1
    len_mask = len(mask_val)
    c0 = 0
    c1 = 0
    for mv in mask_val:
        if mv == 0:
            c0 += 1
        else:
            c1 += 1
    return mask_index, mask_val, len_mask, c0, c1, total_predicted_1, total_agree_1, total_reversed_1


def run_initial(
        n=None, l1l2=None, s0=None, mask_num=None, mask=None, seed=None, d=None, graph_type=None, sem_type=None,
        w_threshold=None, hidden_units=None, dict_data=None, exp_no=None, B_true=None, X=None, learned_model=None,
        trial_no=None, list_dict_data=None
):
    ## initial run with baseline model without any imposed knowledge
    PARAMS = {
        'seed': seed,
        'lambda1': l1l2,
        'lambda2': l1l2,
        'w_threshold': w_threshold,
        'hidden_units': hidden_units,
        'n': n,
        'd': d,
        's0': s0,
        'mask_num': mask_num,
        'graph_type': graph_type,
        'sem_type': sem_type,
    }
    exp_name = str('exp_' + 'real_data_' + str(exp_no))
    ut.set_random_seed(PARAMS['seed'])
    model = nonlinear.NotearsMLP(
        dims=[PARAMS['d'], PARAMS['hidden_units'], 1], bias=True,
        mask=mask, w_threshold=PARAMS['w_threshold'], learned_model=None)
    W_notears, res = nonlinear.notears_nonlinear(model, X, lambda1=PARAMS['lambda1'], lambda2=PARAMS['lambda2'])
    learned_model = res['learned_model']
    mask_index, mask_val, len_mask, c0, c1, pred_1, agree_1, reversed_1 = get_mask_index_mask_val(d, B_true, W_notears, X)
    try:
        assert ut.is_dag(W_notears)
        filename_mask = exp_name + '_mask.csv'
        np.savetxt('outputs/real/sachs_protein/' + filename_mask, mask, delimiter=',')
        filename_wnotears = exp_name + '_W_notears.csv'
        np.savetxt('outputs/real/sachs_protein/' + filename_wnotears, W_notears, delimiter=',')
        acc = ut.count_accuracy(B_true, W_notears != 0)
        local_data = {}
        local_data['mask_num'] = mask_num
        local_data['trial_no'] = trial_no
        local_data['c0'] = c0
        local_data['c1'] = c1
        local_data['pred_1'] = pred_1
        local_data['agree_1'] = agree_1
        local_data['reversed_1'] = reversed_1
        local_data['fdr'] = acc['fdr']
        local_data['tpr'] = acc['tpr']
        local_data['fpr'] = acc['fpr']
        local_data['shd'] = acc['shd']
        local_data['h'] = res['h']
        local_data['h_zero'] = res['h_zero']
        local_data['h_ineq'] = res['h_ineq']
        local_data['nnz'] = acc['nnz']
        dict_data[exp_no] = local_data
        list_dict_data.append(local_data)
    except Exception as e:
        print('Error: ' + str(e))
    return dict_data, B_true, W_notears, X, learned_model, list_dict_data


def run_iterative(
        n=None, l1l2=None, s0=None, mask_num=None, mask=None, seed=None, d=None, graph_type=None,
        sem_type=None, w_threshold=None, hidden_units=None, dict_data=None, exp_no=None, B_true=None,
        X=None, learned_model=None, trial_no=None, list_dict_data=None
):
    ## iterative run with imposed knowledge as mask
    PARAMS = {
        'seed': seed,
        'lambda1': l1l2,
        'lambda2': l1l2,
        'w_threshold': w_threshold,
        'hidden_units': hidden_units,
        'n': n,
        'd': d,
        's0': s0,
        'mask_num': mask_num,
        'graph_type': graph_type,
        'sem_type': sem_type,
    }
    exp_name = str('exp_' + 'real_data_' + str(exp_no))
    ut.set_random_seed(PARAMS['seed'])
    model = nonlinear.NotearsMLP(
        dims=[PARAMS['d'], PARAMS['hidden_units'], 1], bias=True,
        mask=mask, w_threshold=PARAMS['w_threshold'], learned_model=None)
    W_notears, res = nonlinear.notears_nonlinear(model, X, lambda1=PARAMS['lambda1'], lambda2=PARAMS['lambda2'])
    learned_model = res['learned_model']
    mask_index, mask_val, len_mask, c0, c1, pred_1, agree_1, reversed_1 = get_mask_index_mask_val(d, B_true, W_notears, X)
    try:
        assert ut.is_dag(W_notears)
        filename_mask = exp_name + '_mask.csv'
        np.savetxt('outputs/real/sachs_protein/' + filename_mask, mask, delimiter=',')
        filename_wnotears = exp_name + '_W_notears.csv'
        np.savetxt('outputs/real/sachs_protein/' + filename_wnotears, W_notears, delimiter=',')
        acc = ut.count_accuracy(B_true, W_notears != 0)
        local_data = {}
        local_data['mask_num'] = mask_num
        local_data['trial_no'] = trial_no
        local_data['c0'] = c0
        local_data['c1'] = c1
        local_data['pred_1'] = pred_1
        local_data['agree_1'] = agree_1
        local_data['reversed_1'] = reversed_1
        local_data['fdr'] = acc['fdr']
        local_data['tpr'] = acc['tpr']
        local_data['fpr'] = acc['fpr']
        local_data['shd'] = acc['shd']
        local_data['h'] = res['h']
        local_data['h_zero'] = res['h_zero']
        local_data['h_ineq'] = res['h_ineq']
        local_data['nnz'] = acc['nnz']
        dict_data[exp_no] = local_data
        list_dict_data.append(local_data)
    except Exception as e:
        print('Error: ' + str(e))
    return dict_data, B_true, W_notears, X, learned_model, list_dict_data


if __name__=='__main__':
    dfile = 'datasets/real/sachs_protein/sachs_protein_total.xls'
    exp_no, l1l2, s0, seed, w_threshold, hidden_units, ntrial = 1, 0.03, 20, 123, 0.3, 10, 10
    knowledge_source = 0   # imposing knowledge from the set of misclassified edges
    knowledge_type = 1  # imposing active edges from the consensus network as knowledge

    ## consensus network
    B_true = np.asarray([
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    df = pd.read_excel(dfile)
    old_X = df.values

    ## standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit(old_X).transform(old_X)

    folder_results = 'results/real/sachs_protein/'
    folder_outputs = 'outputs/real/sachs_protein/'
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
        print('result folder created!')
    else:
        print('result folder exists!')
    if not os.path.exists(folder_outputs):
        os.makedirs(folder_outputs)
        print('output folder created!')
    else:
        print('output folder exists!')


    n, d = X.shape
    dict_data = {}
    list_dict_data = []
    for trial_no in range(ntrial):
        mask = np.ones((d, d)) * np.nan
        mask_num = 0
        dict_data, B_true, W_notears, X, learned_model, list_dict_data = run_initial(
            n=n, l1l2=l1l2, s0=s0, mask_num=mask_num, mask=mask, seed=seed, d=d, w_threshold=w_threshold,
            hidden_units=hidden_units, dict_data=dict_data, exp_no=exp_no, B_true=B_true, X=X, learned_model=None,
            trial_no=trial_no, list_dict_data=list_dict_data
        )
        exp_no += 1
        mi0, mv0, mi1, mv1 = get_mask_0_1(d, B_true, W_notears, X, [])
        if knowledge_source == 0 and knowledge_type == 1:
            len_mv1 = len(mv1)
            list_knowledge_index = []
            list_knowledge_val = []
            while (len_mv1 > 0):
                r = random.randint(0, len_mv1 - 1)
                knowledge_index = mi1[r]
                knowledge_val = mv1[r]
                while knowledge_index in list_knowledge_index:
                    r = random.randint(0, len_mv1 - 1)
                    knowledge_index = mi1[r]
                    knowledge_val = mv1[r]
                list_knowledge_index.append(knowledge_index)
                list_knowledge_val.append(knowledge_val)
                mask_num = 0
                mask = np.ones((d, d)) * np.nan
                for i in range(len(list_knowledge_index)):
                    mask[list_knowledge_index[i]] = list_knowledge_val[i]
                    mask_num += 1
                dict_data, B_true, W_notears, X, learned_model, list_dict_data = run_iterative(
                    n=n, l1l2=l1l2, s0=s0, mask_num=mask_num, mask=mask, seed=seed, d=d, w_threshold=w_threshold,
                    hidden_units=hidden_units, dict_data=dict_data, exp_no=exp_no, B_true=B_true, X=X,
                    learned_model=None, trial_no=trial_no, list_dict_data=list_dict_data
                )
                exp_no += 1
                mi0, mv0, mi1, mv1 = get_mask_0_1(d, B_true, W_notears, X, list_knowledge_index)
                len_mv1 = len(mv1)


        filename_result = 'real_sachs_protein' + '.csv'
        field_names = [
            'mask_num', 'trial_no', 'c0', 'c1', 'pred_1', 'agree_1', 'reversed_1', 'fdr', 'tpr', 'fpr', 'shd',
            'h', 'h_zero', 'h_ineq', 'nnz'
        ]
        with open(folder_results + str(filename_result), 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(list_dict_data)