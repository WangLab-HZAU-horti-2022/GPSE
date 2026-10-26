"""CatBoost adapters covering Pool construction and post-fit native operations.

CatBoost 1.2.10 uses thread_count=-1 for predict and feature importance even
when its training parameter specifies a positive count. Its implicit training
Pool also defaults to -1. These subclasses keep all those operations bounded.
They are only selected by the explicit lease execution path.
"""

from catboost import CatBoostClassifier, CatBoostRegressor, Pool


class _LeaseCatBoost:
    def _threads(self):
        from gpse.train._thread_plan import current_plan

        plan = current_plan()
        count = plan.estimator_threads if plan else self.get_param("thread_count")
        if type(count) is not int or count < 1:
            raise ValueError("Lease CatBoost requires a positive thread_count")
        return count

    def fit(self, X, y=None, **kwargs):
        from gpse.train._task_store import native_log_directory

        train_dir = native_log_directory()
        if train_dir is not None:
            self.set_params(train_dir=train_dir)
        count = self._threads()
        if not isinstance(X, Pool):
            pool_kwargs = {}
            for name in ("cat_features", "text_features", "embedding_features"):
                value = kwargs.pop(name, self.get_param(name))
                if value is not None:
                    pool_kwargs[name] = value
            if kwargs.get("sample_weight") is not None:
                pool_kwargs["weight"] = kwargs.pop("sample_weight")
            X, y = Pool(X, y, thread_count=count, **pool_kwargs), None
        if kwargs.get("eval_set") is not None:

            def bounded_pool(value):
                if isinstance(value, Pool):
                    return value
                if isinstance(value, tuple) and len(value) == 2:
                    return Pool(value[0], value[1], thread_count=count)
                raise ValueError("Lease CatBoost eval_set must contain Pools or (X, y) tuples")

            evaluation = kwargs["eval_set"]
            kwargs["eval_set"] = (
                [bounded_pool(value) for value in evaluation]
                if isinstance(evaluation, list)
                else bounded_pool(evaluation)
            )
        return super().fit(X, y, **kwargs)

    def predict(self, data, **kwargs):
        kwargs["thread_count"] = self._threads()
        return super().predict(data, **kwargs)

    def get_feature_importance(self, data=None, **kwargs):
        kwargs["thread_count"] = self._threads()
        return super().get_feature_importance(data=data, **kwargs)


class LeaseCatBoostRegressor(_LeaseCatBoost, CatBoostRegressor):
    pass


class LeaseCatBoostClassifier(_LeaseCatBoost, CatBoostClassifier):
    def predict_proba(self, X, **kwargs):
        kwargs["thread_count"] = self._threads()
        return super().predict_proba(X, **kwargs)


def adapter_class(import_path):
    return {
        "catboost.CatBoostRegressor": LeaseCatBoostRegressor,
        "catboost.CatBoostClassifier": LeaseCatBoostClassifier,
    }[import_path]
