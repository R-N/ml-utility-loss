import math
import os
from .util import mkdir, split_df, split_df_2, split_df_ratio, split_df_kfold
import json
from .params import PARAM_MAP, BOOLEAN

def map_parameter(param, source):
    try:
        return source[param]
    except Exception as ex:
        return param

def sample_parameter(trial, name, type, args, kwargs):
    return getattr(trial, f"suggest_{type}")(name, *args, **kwargs)

def sample_int_exp_2(trial, k, low, high):
    low = max(low, 1)
    low = math.log(low, 2)
    high = math.log(high, 2)
    assert low % 1 == 0
    assert high % 1 == 0
    param = int(math.pow(2, trial.suggest_int(f"{k}_exp_2", low, high)))
    return param

def sample_parameter_2(trial, k, type_0, args, kwargs=None, param_map={}):
    kwargs = kwargs or {}
    param, param_raw = None, None
    type_1 = type_0
    if type_0 == "conditional":
        sample = trial.suggest_categorical(f"{k}_boolc", [True, False])
        if sample:
            return sample_parameters(trial, args[0], param_map=param_map)
        return None, None
    if type_0.startswith("bool_"):
        sample = trial.suggest_categorical(f"{k}_bool", [True, False])
        if not sample:
            return 0, 0
        type_1 = type_0[5:]
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map)
    if type_0.startswith("log_"):
        type_1 = type_0[4:]
        kwargs["log"] = True
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map)
    if type_0 == "qloguniform":
        low, high, q = args
        type_1 = "float"
        param = round(math.exp(
            trial.suggest_float(f"{k}_qloguniform", low, high)
        ) / q) * q
        return param, param
    if type_0 == "list_int_exp_2":
        #type_1 = type_0[5:]
        min, max, low, high = args
        length = trial.suggest_int(f"{k}_len", min, max)
        param = [
            sample_int_exp_2(trial, f"{k}_{i}", low, high)
            for i in range(length)
        ]
        param_raw = repr(param)
        return param, param_raw
    if type_0 == "int_exp_2":
        low, high = args
        param = sample_int_exp_2(trial, k, low, high)
        type_1 = "int"
        return param, param
    if type_0 in {"bool", "boolean"}:
        type_1, *args = BOOLEAN
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map)

    if type_0 in param_map:
        type_1 = "categorical"

    if type_1:
        param = sample_parameter(trial, k, type_1, args, kwargs)

    param_raw = param
    if type_0 in param_map:
        param = map_parameter(param, param_map[type_0])

    return param, param_raw

def sample_parameters(trial, param_space, param_map={}):
    param_map = {**PARAM_MAP, **param_map}
    params = {}
    params_raw = {}
    for k, v in param_space.items():
        type_0, *args = v
        param, param_raw = sample_parameter_2(trial, k, type_0, args, param_map=param_map)
        params[k] = param
        params_raw[k] = param_raw
        
    #params["id"] = trial.number
    return params, params_raw


def map_parameters(params_raw, param_map={}):
    param_map = {**PARAM_MAP, **param_map}
    ret = {}
    for k, v in params_raw.items():
        if k.endswith("_exp_2"):
            v = int(math.pow(2, v))
            k = k[:-6]
        else:
            for k0, v0 in param_map.items():
                if k0 in k:
                    v = v0[v]
        ret[k] = v
    return ret


def create_objective(
    objective, sampler=sample_parameters, 
    objective_kwargs={}, sampler_kwargs={}, 
    checkpoint_dir=None, 
    log_dir="logs",
    study_dir="studies",
):
    objective_kwargs = dict(objective_kwargs)
    def f(trial):
        id = trial.number
        print(f"Begin trial {trial.number}")
        trial_dir = os.path.join(study_dir, str(id))
        mkdir(trial_dir)

        params, params_raw = sampler(trial, **sampler_kwargs)
        param_path = os.path.join(trial_dir, "params.json")
        with open(param_path, 'w') as f:
            try:
                json.dump(params_raw, f, indent=4)
            except TypeError as ex:
                print(params_raw)
                raise
        print(json.dumps(params_raw, indent=4))
        kwargs = {}
        if checkpoint_dir:
            kwargs["checkpoint_dir"] = os.path.join(trial_dir, checkpoint_dir)
        if log_dir:
            kwargs["log_dir"] = os.path.join(trial_dir, log_dir)
        return objective(
            **objective_kwargs,
            **params, 
            **kwargs,
            trial=trial,
        )
    return f

def make_objective_random(
    objective,
    loader=None,
    ratio=0.2,
    val=False,
):
    def f(df, *args, **kwargs):
        datasets = split_df_ratio(
            df, 
            ratio=ratio,
            val=val,
        )
        if loader:
            datasets = [loader(d) for d in datasets]
        return objective(
            datasets,
            *args,
            **kwargs
        )
    return f

def make_objective_kfold(
    objective,
    loader=None,
    ratio=0.2,
    val=False,
    seed=None
):
    def f(df, *args, **kwargs):
        values = []
        splits = split_df_kfold(
            df, 
            ratio=ratio,
            val=val,
            seed=seed,
        )
        for datasets in splits:
            if loader:
                datasets = [loader(d) for d in datasets]
            value = objective(
                datasets,
                *args,
                **kwargs
            )
            values.append(value)
        avg_value = sum(values) / len(values)
        return avg_value
    return f
