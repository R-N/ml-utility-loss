from catboost import Pool
import pandas as pd
def create_pool(df, target, cat_features):
    X = df.drop(target, axis=1)
    y = df[target]
    cat_features = [x for x in cat_features if x != target]

    pool = Pool(
        X,
        label=y,
        cat_features=cat_features
    )
    y_mode = y.mode(dropna=True)[0]
    y_support = len(y[y==y_mode])
    
    pool.y_mode = y_mode
    pool.y_support = y_support
    return pool

def create_pool_2(df, info):
    return create_pool(df, info["target"], info["cat_features"])