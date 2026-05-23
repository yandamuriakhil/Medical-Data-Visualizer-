import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
# 1. Import the data
df = pd.read_csv('medical_examination.csv')
 
# 2. Add overweight column
# BMI = weight(kg) / height(m)^2
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)
 
# 3. Normalize cholesterol and gluc:
#    value == 1 → 0 (good), value > 1 → 1 (bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
 
 
# 4. Draw the Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
 
    # 6. Group and reformat: count occurrences, rename column
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat.rename(columns={'size': 'total'}, inplace=True)
 
    # 7. Draw the catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).figure
 
    # 9. Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig
 
 
# 10. Draw the Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
 
    # 12. Calculate the correlation matrix
    corr = df_heat.corr()
 
    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
 
    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
 
    # 15. Plot the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmin=-0.1,
        vmax=0.3,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.5}
    )
 
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    draw_cat_plot()
    draw_heat_map()
    plt.show()