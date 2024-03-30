import math
import os
from .util import mkdir, split_df, split_df_2, split_df_ratio, split_df_kfold, roundmul
import json
from .params import PARAM_MAP, BOOLEAN
import optuna

def map_parameter(param, source):
    try:
        return source[param]
    except Exception as ex:
        return param

def sample_parameter(trial, name, type, args, kwargs):
    try:
        return getattr(trial, f"suggest_{type}")(name, *args, **kwargs)
    except ValueError as ex:
        msg = str(ex)
        if "CategoricalDistribution does not support dynamic value space" in msg:
            print("Offending parameter:", name)
        raise

def sample_int_exp_2(trial, k, low, high, *args, **kwargs):
    low = max(low, 1)
    low = math.log(low, 2)
    high = math.log(high, 2)
    assert low % 1 == 0
    assert high % 1 == 0
    param = int(math.pow(2, trial.suggest_int(f"{k}_exp_2", low, high, *args, **kwargs)))
    return param

def sample_int(trial, name, low, high, step=1, log=False, roundup=False, **kwargs):
    high = roundmul(high, multiple=step, offset=low, up=roundup)
    if log:
        step = None
    return trial.suggest_int(name, low, high, step=step, log=log, **kwargs)

def sample_float(trial, name, low, high, step=None, log=False, **kwargs):
    if log:
        step = None
    return trial.suggest_float(name, low, high, step=step, log=log, **kwargs)

def sample_parameter_2(trial, k, type_0, args, kwargs=None, param_map={}, map=True):
    kwargs = kwargs or {}
    param, param_raw = None, None
    type_1 = type_0
    if type_0 == "dict":
        return sample_parameters(trial, args[0], param_map=param_map, map=map)
    if type_0 == "conditional":
        sample = trial.suggest_categorical(f"{k}_boolc", [True, False])
        if sample:
            return sample_parameters(trial, args[0], param_map=param_map, map=map)
        return None, None
    if type_0.startswith("bool_"):
        sample = trial.suggest_categorical(f"{k}_bool", [True, False])
        if not sample:
            return None, None
        type_1 = type_0[5:]
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map, map=map)
    if type_0.startswith("log_"):
        type_1 = type_0[4:]
        kwargs["log"] = True
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map, map=map)
    if type_0 == "qloguniform":
        low, high, q = args
        type_1 = "float"
        param = round(math.exp(
            trial.suggest_float(f"{k}_qloguniform", low, high)
        ) / q) * q
        return param, param
    if type_0 == "list_int_exp_2":
        #raise ValueError(f"{k} list_int_exp_2 Deprecated")
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
        return sample_parameter_2(trial, k, type_1, args, kwargs, param_map=param_map, map=map)

    if type_0 in param_map:
        type_1 = "categorical"

    if type_1 == "int":
        param = sample_int(trial, k, *args, **kwargs)
    elif type_1 == "float":
        param = sample_float(trial, k, *args, **kwargs)
    elif type_1:
        param = sample_parameter(trial, k, type_1, args, kwargs)

    param_raw = param
    if not map:
        return param_raw
    if type_0 in param_map:
        param = map_parameter(param, param_map[type_0])

    return param, param_raw

def sample_parameters(trial, param_space, param_map={}, force_fix=None, map=True):
    param_map = {**PARAM_MAP, **param_map}
    params = {}
    params_raw = {}
    for k, v in param_space.items():
        if not isinstance(v, (list, tuple)):
            params[k] = params_raw[k] = v
            continue
        type_0, *args = v
        try:
            param, param_raw = sample_parameter_2(trial, k, type_0, args, param_map=param_map, map=map)
        except ValueError as ex:
            msg = str(ex)
            if "CategoricalDistribution does not support dynamic value space" in msg:
                print("Offending parameter:", k, type_0, args)
            raise
        if k in params:
            continue
        params[k] = param
        params_raw[k] = param_raw
        
    if force_fix:
        params_raw = force_fix(params_raw)
        try:
            params = {
                **params,
                **map_parameters(params_raw, param_space=param_space, param_map=param_map, strict=True),
            }
        except AssertionError as ex:
            pass
        trial_params = force_fix(trial.params)
        try:
            params = {
                **params,
                **map_parameters(trial_params, param_space=param_space, param_map=param_map, strict=True),
            }
        except AssertionError as ex:
            pass
        
    #params["id"] = trial.number
    return params, params_raw


def pop_update(kwargs, arg_name, out_kwargs=None):
    arg = kwargs.pop(arg_name, None)
    out_kwargs = kwargs if out_kwargs is None else out_kwargs
    if arg is not None:
        if isinstance(arg, dict):
            out_kwargs.update(arg)
        else:
            out_kwargs[arg_name] = arg
    return out_kwargs

def pop_repack(kwargs, arg_name, out_kwargs=None):
    arg = kwargs.pop(arg_name, None)
    out_kwargs = kwargs if out_kwargs is None else out_kwargs
    if arg is not None:
        l = len(arg_name) + 1
        if isinstance(arg, dict):
            if arg_name in arg:
                out_kwargs[arg_name] = arg.pop(arg_name)
            arg_kwargs = out_kwargs.pop(f"{arg_name}_kwargs", {})
            arg_kwargs.update(kwargs.pop(f"{arg_name}_kwargs", {}))
            arg_kwargs.update({k[l:]: v for k, v in arg.items()})
            out_kwargs[f"{arg_name}_kwargs"] = arg_kwargs
        else:
            out_kwargs[arg_name] = arg
            arg_kwargs = out_kwargs.pop(f"{arg_name}_kwargs", {})
            arg_kwargs.update(kwargs.pop(f"{arg_name}_kwargs", {}))
            for k in list(kwargs.keys()):
                if k.startswith(arg_name):
                    arg_kwargs[k[l:]] = kwargs.pop(k)
            out_kwargs[f"{arg_name}_kwargs"] = arg_kwargs
    return out_kwargs

def unpack_params(kwargs):
    kwargs = pop_update(kwargs, "tf_pma")
    kwargs = pop_update(kwargs, "tf_lora")
    kwargs = pop_update(kwargs, "ada_lora")
    kwargs = pop_update(kwargs, "head_lora")
    kwargs = pop_update(kwargs, "tf_num_inds")
    kwargs = pop_update(kwargs, "ada_n_seeds")

    gradient_penalty_kwargs = kwargs.pop("gradient_penalty_kwargs", {})
    gradient_penalty_kwargs = pop_repack(kwargs, "mse_mag", gradient_penalty_kwargs)
    gradient_penalty_kwargs = pop_repack(kwargs, "mag_corr", gradient_penalty_kwargs)
    gradient_penalty_kwargs = pop_repack(kwargs, "cos_loss", gradient_penalty_kwargs)
        
    kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_bool")}
    kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_boolc")}
    kwargs["gradient_penalty_kwargs"] = gradient_penalty_kwargs
    return kwargs

def map_parameters(params_raw, param_space={}, param_map={}, unpack=True, strict=False):
    param_map = {**PARAM_MAP, **param_map}
    ret = {}
    for k, v in params_raw.items():
        if k.endswith("_bool") or k.endswith("_boolc"):
            continue
        done = False
        if not done and k.endswith("_exp_2"):
            v = int(math.pow(2, v))
            k = k[:-6]
            done = True
        if not done and k in param_space:
            v0 = param_space[k]
            if not isinstance(v0, list) and not isinstance(v0, tuple):
                #v = v0
                #done = True
                pass
            else:
                type_0, *args = v0
                if not done and type_0 in param_map:
                    try:
                        v = param_map[type_0][v]
                        done = True # NO BREAK, NO LOOP HERE
                    except (KeyError, TypeError):
                        pass
        if not done:
            for k0, v0 in param_space.items():
                if not isinstance(v0, list) and not isinstance(v0, tuple):
                    #v = v0
                    #done = True
                    continue
                if not done and k0 in k:
                    try:
                        type_0, *args = v0
                        if not done and type_0 == "conditional":
                            v = map_parameters({k: v}, args[0], param_map=param_map)[k]
                            done = True
                        break
                    except (KeyError, TypeError):
                        pass
        if not done:
            for k0, v0 in param_map.items():
                if k0 in k:
                    try:
                        v = v0[v]
                        done = True
                        break
                    except (KeyError, TypeError):
                        pass
        if strict and k not in param_space:
            continue
        ret[k] = v
    for k, v in param_space.items():
        if k not in ret and not isinstance(v, (tuple, list)):
            ret[k] = v
    if unpack:
        ret = unpack_params(ret)
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
        #params = unpack_params(params)
        param_path = os.path.join(trial_dir, "params.json")
        with open(param_path, 'w') as f:
            try:
                json.dump(params_raw, f, indent=4)
            except TypeError as ex:
                print(params_raw)
                raise
        print(json.dumps(unpack_params(params_raw), indent=4))
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
    seed=42
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

def load_params(study_name, storage, trials):
    study = optuna.load_study(study_name=study_name, storage=storage)
    good_params = [t.params for t in study.trials if t.number in trials]
    del study
    return good_params

def enqueue_params(study, params):
    for p in params:
        study.enqueue_trial(p, skip_if_exists=True)
