from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from boostclean import boost_clean
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd

from diffprep.prep_space import space
from diffprep.pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
from diffprep.trainer.diffprep_trainer import DiffPrepSGD
from diffprep.model import LogisticRegression as DiffprepLogisticRegression
from diffprep.experiment.experiment_utils import min_max_normalize, set_random_seed

def make_pipeline_func(
    cleaning: str,
    classifier,
    *,
    autosklearn_time: int = 30,
    random_n_trials: int = 10,
):
    """
    Returns a pipeline(X_train, y_train, X_test) function.

    The returned pipeline must accept (X_train, y_train, X_test)
    and return a N x 2 array of predict_proba outputs.
    """
    # --- simple imputers ---
    if cleaning == 'MeanImputer':
        imputer = SimpleImputer(strategy='mean')
    elif cleaning == 'MedianImputer':
        imputer = SimpleImputer(strategy='median')
    elif cleaning == 'KNNImputer':
        imputer = KNNImputer(n_neighbors=10)
    elif cleaning == 'IterativeImputer':
        imputer = IterativeImputer(max_iter=10, random_state=42)
    else:
        imputer = None

    def pipeline_impute(X_train, y_train, X_test):
        clf = clone(classifier)

        X_tr = imputer.fit_transform(X_train)
        X_te = imputer.transform(X_test)

        clf.fit(X_tr_s, y_train)
        return clf.predict_proba(X_te_s)

    def pipeline_h2o(X_train, y_train, X_test):
        clf = clone(classifier)
        # 1) mean‐impute
        imputer = SimpleImputer(strategy='mean')
        X_tr_imp = imputer.fit_transform(X_train)
        X_te_imp = imputer.transform(X_test)
        # 2) scale
        scaler = StandardScaler().fit(X_tr_imp)
        X_tr_s = scaler.transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)
        # 3) fit & predict
        clf.fit(X_tr_s, y_train)
        return clf.predict_proba(X_te_s)

    def pipeline_autosklearn(X_train, y_train, X_test):
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn_add_custom_clfs import add_clf

        model_name = classifier  # expecting classifier to be a string here
        if model_name == 'DT':
            clf_name = 'CustomDecisionTree'
        elif model_name == 'RF':
            clf_name = 'CustomRandomForest'
        elif model_name == 'SVM':
            clf_name = 'CustomSVM'
        elif model_name == 'NN':
            clf_name = 'CustomMLPClassifier'
        elif model_name == 'LR':
            clf_name = 'CustomLogisticRegression'
        else:
            raise ValueError(f"Unsupported autosklearn model {model_name!r}")

        add_clf(clf_name)
        per_run = max(1, autosklearn_time // 5)
        asm = AutoSklearnClassifier(
            time_left_for_this_task=autosklearn_time,
            per_run_time_limit=per_run,
            include={'classifier': [clf_name]},
            memory_limit=8 * 1024
        )
        asm.fit(X_train, y_train)
        return asm.predict_proba(X_test)

    def pipeline_random(X_train, y_train, X_test):
        from random_search import RandomSearch

        model_name = classifier  # expecting classifier to be a string here
        if model_name == 'DT':
            clf_name = 'DecisionTree'
        elif model_name == 'RF':
            clf_name = 'RandomForest'
        elif model_name == 'SVM':
            clf_name = 'SVM'
        elif model_name == 'NN':
            clf_name = 'MLPClassifier'
        else:
            clf_name = 'LogisticRegression'

        fitted, _ = RandomSearch(X_train, y_train, None, None,
                                 clf_name=clf_name,
                                 n_trials=random_n_trials)
        return fitted.predict_proba(X_test)

    def pipeline_boostclean(X_train, y_train, X_test):
        # The returned `test_proba` is a 1-D array of probabilities for the positive class

        # Create a dummy y_test with at least two classes to avoid TrialPruned error in boost_clean
        num_test = len(X_test)
        y_test_dummy = np.tile([0, 1], num_test // 2 + 1)[:num_test] if num_test > 0 else np.array([])

        _, test_proba, _, _, _ = boost_clean(
            model=clone(classifier),
            X_train=X_train, y_train=y_train.to_numpy(),
            X_test=X_test, y_test=y_test_dummy, # y_test is not used for this pipeline's output
            X_test_sensitive=np.zeros(len(X_test)), # sensitive attribute is not used for this pipeline's output
        )
        # The pipeline function is expected to return a (N, 2) array
        return np.c_[1 - test_proba, test_proba]

    def pipeline_diffprep(X_train, y_train, X_test):

        params = {
            "num_epochs": 100, "batch_size": 512, "device": "cpu",
            "model_lr": 0.01, "weight_decay": 0, "model": "log",
            "train_seed": 1, "split_seed": 1, "method": "diffprep_fix",
            "save_model": True, "logging": False, "no_crash": False,
            "patience": 10, "momentum": 0.9, "prep_lr": None,
            "temperature": 0.1, "grad_clip": None, "pipeline_update_sample_size": 512,
            "init_method": "default", "diff_method": "num_diff", "sample": False
        }
        seed = 42

        # Data splitting and preparation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.5, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test_dummy = pd.Series(np.zeros(len(X_test)), index=X_test.index)

        # Create dummy sensitive attributes as they are not available in the pipeline
        sensitive_attr_val = np.zeros(len(X_val))
        sensitive_attr_test = np.zeros(len(X_test))

        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        set_random_seed(params)

        prep_pipeline = DiffPrepFixPipeline(space, temperature=params["temperature"],
                                            use_sample=params["sample"],
                                            diff_method=params["diff_method"],
                                            init_method=params["init_method"])
        prep_pipeline.init_parameters(X_train, X_val, X_test)

        # Model setup
        input_dim = prep_pipeline.out_features
        output_dim = len(set(y_train.values.ravel()))
        set_random_seed(params)
        model = DiffprepLogisticRegression(input_dim, output_dim).to(params["device"])
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer setup
        model_optimizer = torch.optim.SGD(
            model.parameters(), lr=params["model_lr"],
            weight_decay=params["weight_decay"], momentum=params["momentum"]
        )
        prep_lr = params["prep_lr"] if params["prep_lr"] is not None else params["model_lr"]
        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(), lr=prep_lr,
            betas=(0.5, 0.999), weight_decay=params["weight_decay"]
        )

        # Trainer setup
        diff_prep = DiffPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                                None, None, params, writer=None)

        result, _ = diff_prep.fit(X_train, y_train, X_val, y_val,
                                  sensitive_attr_val, sensitive_attr_test,
                                  X_test, y_test_dummy)
        
        test_proba = result["best_test_prob"]
        return np.c_[1 - test_proba, test_proba]

    def not_impl(*args, **kwargs):
        raise NotImplementedError(f"'{cleaning}' pipeline does not return probabilities directly")

    # --- dispatch ---
    if cleaning in ('MeanImputer', 'MedianImputer', 'KNNImputer', 'IterativeImputer'):
        return pipeline_impute
    elif cleaning == 'h2o':
        return pipeline_h2o
    elif cleaning == 'autosklearn':
        return pipeline_autosklearn
    elif cleaning == 'random':
        return pipeline_random
    elif cleaning == 'boostclean':
        return pipeline_boostclean
    elif cleaning == 'diffprep':
        return pipeline_diffprep
    else:
        raise ValueError(f"Unsupported cleaning: {cleaning!r}")
