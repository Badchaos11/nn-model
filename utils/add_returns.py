import pandas as pd
from nn.functions import convert_yahoo


def extend_df(extender: pd.DataFrame, extandable: pd.DataFrame):

    # extender = extender.drop(['Unnamed: 0'], axis=1)
    extender = add_min_mean_max(extender)
    extandable = extandable.drop(['Unnamed: 0'], axis=1)
    fd = pd.DataFrame()
    for year in extandable.Year.unique().tolist():
        l_min = []
        l_mean = []
        l_max = []
        x = extender.where(extender.Date == year).dropna()
        y = extandable.where(extandable.Year == year).dropna()
        for company in y.Company.tolist():
            l_min.append(x.where(x.Company == company).dropna()["Min Predict Return"].values[0])
            l_mean.append(x.where(x.Company == company).dropna()["Mean Predict Return"].values[0])
            l_max.append(x.where(x.Company == company).dropna()["Max Predict Return"].values[0])

        y['Min Predict Return'] = l_min
        y['Mean Predict Return'] = l_mean
        y['Max Predict Return'] = l_max
        fd = pd.concat([fd, y])

    fd = fd.set_index(["Year", "Company"])
    print(fd)

    fd.to_csv('data/Extended_for_BL.csv')


def add_min_mean_max(df: pd.DataFrame) -> pd.DataFrame:

    df['Company'] = convert_yahoo(df.Company.tolist())

    lst_mean = []
    lst_min = []
    lst_max = []

    for i in df.index.tolist():
        lst_mean.append(df.iloc[i, 2:].mean())
        lst_min.append(df.iloc[i, 2:].min())
        lst_max.append(df.iloc[i, 2:].max())

    df['Min Predict Return'] = lst_min
    df['Mean Predict Return'] = lst_mean
    df['Max Predict Return'] = lst_max
    df['Date'] = df['Date'].astype(int)
    df = df.sort_values(by=['Date'], ascending=False)

    return df

