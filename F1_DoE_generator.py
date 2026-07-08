#%%
# function to create DoE for Abaqus
import numpy as np
import pandas as pd
import itertools

def make_mixed_doe(
    n_runs,
    ranges,
    categories,
    seed=42,
    constraints=None,
    round_digits=4,
):
    rng = np.random.default_rng(seed)

    def lhs_1d(low, high, n):
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        return low + u * (high - low)

    num_df = pd.DataFrame({
        name: lhs_1d(low, high, n_runs)
        for name, (low, high) in ranges.items()
    })

    cat_names = list(categories.keys())
    if cat_names:
        cat_levels = list(itertools.product(*[categories[name] for name in cat_names]))
        idx = rng.choice(len(cat_levels), size=n_runs, replace=True)
        cat_df = pd.DataFrame([cat_levels[i] for i in idx], columns=cat_names)
    else:
        cat_df = pd.DataFrame(index=range(n_runs))

    df = pd.concat([num_df, cat_df], axis=1)

    if constraints is not None:
        mask = df.apply(constraints, axis=1)
        df = df[mask].reset_index(drop=True)

    num_cols = list(ranges.keys())
    df[num_cols] = df[num_cols].round(round_digits)

    # add counter as last column
    df["counter"] = np.arange(1, len(df) + 1)

    return df

#%%
ranges = {
           'length'  : (8,12),
           'breadth' : (16,24),
           'Cirlce_radius': (2,2.5),
           'Circle_pos_X' : (-1,1),
           'Circle_pos_Y' : (-1,1),
           'Load' : (10,20)
         }
categories = {
              'Material': ['Steel', 'Alu'],
              'bc_loc' :['BC1','BC2']
}

#%%
## creating the data

df = make_mixed_doe(
    n_runs=10000,
    ranges=ranges,
    categories=categories,
    seed=42,
    constraints=None,
)

#%%
# viewing data
import matplotlib.pyplot as plt
plt.scatter(df['length'], df['breadth'])
plt.show();
plt.hist(df['length'])
plt.show();

# %%
df.to_csv('train_validation_data.csv', index=False)

# %%
df_test = make_mixed_doe(
    n_runs=200,
    ranges=ranges,
    categories=categories,
    seed=48,
    constraints=None,
)
df_test.to_csv('test_set.csv', index = False)
# %%


