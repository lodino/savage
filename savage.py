import time
import optuna
import logging
optuna.logging.set_verbosity(logging.WARNING)
from optuna.samplers import TPESampler
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

from err_injection import *


# create pattern function given subpopulation
def create_pattern(col_list, lb_list, ub_list):
    # Check if inputs are valid
    try:
        assert len(col_list) == len(lb_list) == len(ub_list)
    except:
        print(col_list, lb_list, ub_list)
        raise SyntaxError

    def pattern(data_X, data_y):
        # Initialize a mask of all True values
        mask = np.ones(len(data_X), dtype=bool)

        # Iterate over each condition in col_list, lb_list, and ub_list
        for col, lb, ub in zip(col_list, lb_list, ub_list):
            if col == 'Y':
                mask &= (data_y >= lb) & (data_y <= ub)
            else:
                mask &= (data_X[col] >= lb) & (data_X[col] <= ub)

        # Convert Boolean mask to binary indicators (1 for True, 0 for False)
        binary_indicators = mask.astype(int)
        
        return binary_indicators

    return pattern

        
def objective(trial, X_train, X_test, y_train, y_test, budget, 
              pipeline, metric, col_id=0, dependent_cols=['Y'], 
              error_type='MNAR', seed=42):
    lb_list = []
    ub_list = []
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)

    def map_percent_to_value(data, percent):
        sorted_data = data.sort_values().reset_index(drop=True)
        percent = min(max(percent, 0), 1)
        index = int(percent * (len(sorted_data) - 1))
        return sorted_data.iloc[index]

    for col in dependent_cols:
        if col == 'Y':
            if pd.api.types.is_integer_dtype(y_train):
                mv_lb = trial.suggest_int(col + '_mv_lb', int(y_train.min()), int(y_train.max()))
                mv_interval = trial.suggest_int(col + '_mv_int', 0, int(y_train.max()) - mv_lb)
            else:
                mv_lb = trial.suggest_float(col + '_mv_lb', y_train.min(), y_train.max())
                mv_interval = trial.suggest_float(col + '_mv_int', 0, y_train.max() - mv_lb)
            lb_list.append(mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)
        else:
            if pd.api.types.is_integer_dtype(X_train[col]):
                mv_lb = trial.suggest_int(col + '_mv_lb', int(X_train[col].min()), int(X_train[col].max()))
                mv_interval = trial.suggest_int(col + '_mv_int', 0, int(X_train[col].max()) - mv_lb)
                mv_ub = mv_interval + mv_lb
            else:
                mv_lb_percent = trial.suggest_float(col + '_mv_lb', 0, 1)
                mv_interval_percent = trial.suggest_float(col + '_mv_int', 0, 1 - mv_lb_percent)
                mv_ub_percent = mv_interval_percent + mv_lb_percent
                # map precent to real value
                print(
                    f"Before mapping to actual values: mv_lb_percent: {mv_lb_percent}, mv_ub_percent: {mv_ub_percent}")
                mv_lb = map_percent_to_value(X_train[col], mv_lb_percent)
                mv_ub = map_percent_to_value(X_train[col], mv_ub_percent)
                print(f"After mapping to actual values: mv_lb: {mv_lb}, mv_ub: {mv_ub}")
            lb_list.append(mv_lb)
            ub_list.append(mv_ub)

    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned()

    mv_pattern = create_pattern(dependent_cols, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train, y_train))

    if mv_pattern_len == 0:
        raise optuna.exceptions.TrialPruned()
    mv_num = min(mv_pattern_len, budget)

    if error_type == 'Label':
        label_flip_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = LabelError(pattern=mv_pattern, ratio=label_flip_ratio)
    elif error_type == 'Sampling':
        sampling_error_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = SamplingError(pattern=mv_pattern, ratio=sampling_error_ratio)
    else:
        mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)

    injecter = Injector(error_seq=[mv_err])
    dirty_X_train, dirty_y_train, _, _ = injecter.inject(X_train.copy(), y_train.copy(), 
                                                         X_train, y_train, seed=seed)

    if len(np.unique(dirty_y_train)) < 2:
        raise optuna.exceptions.TrialPruned()
    trial.set_user_attr('num_errs', min(mv_pattern_len, budget))
    
    y_test_pred = pipeline(dirty_X_train, dirty_y_train, X_test)
    # assume higher the better
    perf_metric = metric(X_test, y_test, y_test_pred)

    trial.set_user_attr('num_errs', mv_num)
    trial.set_user_attr('perf_metric', perf_metric)
    trial.set_user_attr('error_injector', injecter)
    trial.set_user_attr('col_list', dependent_cols)
    trial.set_user_attr('lb_list', lb_list)
    trial.set_user_attr('ub_list', ub_list)

    return perf_metric


def run_beam_search(
    X_train, X_test, y_train, y_test,
    pipeline, metric, budget, 
    error_type='MNAR', num_rounds=2, 
    random_state=42, top_k=5, verbose=True
):
    """
    Perform beam search to find the most impactful error injection patterns.

    Parameters:
        X_train, X_test: pd.DataFrame
        y_train, y_test: array-like
        pipeline: sklearn pipeline
        metric: function (y_true, y_pred) -> float
        budget: float, total injection budget (e.g., proportion of corruptions)
        error_type: str, type of error to inject (used in `objective`)
        num_rounds: int, number of beam search rounds
        random_state: int, random seed
        top_k: int, how many top patterns to return
        verbose: bool, whether to print logs

    Returns:
        top_k_patterns: list of ((tuple of column names), score)
    """
    print("Start Beam Search...")
    beam_search_start_time = time.time()

    all_columns = X_train.columns.tolist()
    top_candidates = [(list(), 1)]
    all_columns_res = dict()
    last_top_candidate_value = None

    for round_num in tqdm(range(1, num_rounds + 1), desc="Beam search rounds"):
        candidates_this_round = []
        for candidate in tqdm(top_candidates, desc=f"Round {round_num} candidates", leave=False):
            candidate_col_list = candidate[0]
            for col in tqdm(all_columns, desc=f"Expanding candidate {candidate_col_list + ['Y']}", leave=False):
                if col in candidate_col_list:
                    continue
                new_col_list = candidate_col_list + [col]
                if len(new_col_list) > 1:
                    sorted_new_col_list = (new_col_list[0],) + tuple(sorted(set(new_col_list[1:])))
                else:
                    sorted_new_col_list = tuple(new_col_list)
                if sorted_new_col_list in all_columns_res:
                    continue

                target_col = new_col_list[0]
                target_col_id = X_train.columns.tolist().index(target_col)

                if verbose:
                    print(f"target_col: {target_col}, id: {target_col_id}, cols: {sorted_new_col_list + ('Y',)}")

                study = optuna.create_study(sampler=TPESampler(seed=random_state))
                study.optimize(
                    lambda trial: objective(
                        trial, X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        budget=budget, pipeline=pipeline,
                        metric=metric, col_id=target_col_id,
                        dependent_cols=list(sorted_new_col_list) + ['Y'],
                        error_type=error_type
                    ),
                    n_trials=20,
                    show_progress_bar=False
                )

                try:
                    num_errs_total = study.best_trial.user_attrs['num_errs']
                except KeyError:
                    continue

                if verbose:
                    print(f"Injected {num_errs_total} errors ({num_errs_total/X_train.shape[0]:.2%}). Best value: {study.best_trial.value:.5f}")

                all_columns_res[sorted_new_col_list + ('Y',)] = study.best_trial.value
                candidates_this_round.append((new_col_list, study.best_trial.value))

        top_candidates = sorted(candidates_this_round, key=lambda x: x[1])[:3]
        if verbose:
            print('----------- ROUND BEST -----------')
            print([(x[0]+['Y'], x[1]) for x in top_candidates])
            print('----------------------------------')
        
        # early stop
        if not(last_top_candidate_value is None) and (last_top_candidate_value <= top_candidates[0][1]):
            if verbose:
                print('NO IMPROVEMENT, EARLY STOP')
            break
        
        last_top_candidate_value = top_candidates[0][1]

    beam_search_end_time = time.time()
    beam_search_duration = beam_search_end_time - beam_search_start_time

    if verbose:
        print(f"Beam Search execution time: {beam_search_duration:.2f} seconds")
        print("TOP-k PATTERN:")
        print(sorted(all_columns_res.items(), key=lambda x: x[1])[:top_k])

    return sorted(all_columns_res.items(), key=lambda x: x[1])[:top_k]
