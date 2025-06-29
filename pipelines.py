from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

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
    elif cleaning in ('boostclean', 'diffprep'):
        return not_impl
    else:
        raise ValueError(f"Unsupported cleaning: {cleaning!r}")

