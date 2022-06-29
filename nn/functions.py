import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Memory
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from tensorflow import keras

plt.style.use('seaborn-darkgrid')

warnings.filterwarnings('ignore')

memory = Memory('./cachedir', verbose=0)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))


def selectFeatures(data_inp, train_labels, numb_of_feat):
    best_features = SelectKBest(score_func=f_regression, k=numb_of_feat)
    fit = best_features.fit(data_inp, train_labels)
    # Select best columns
    cols_KBest_numb = best_features.get_support(indices=True)
    cols_KBest = data_inp.iloc[:, cols_KBest_numb].columns
    return cols_KBest


def convert_yahoo(listus):
    yahoo_list = []
    for num in range(len(listus)):
        if "TSX:" in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.TO')
        elif 'TSXV:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.V')
        elif 'XSWX:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.SW')
        elif 'IST:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.IS')
        elif 'XKLS:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.KL')
        elif 'LSE:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.L')
        elif 'SGX:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.SI')
        elif 'HKSE:0' in listus[num]:
            yahoo_list.append(listus[num].replace('HKSE:0', '') + '.HK')
        elif 'ASX:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.AX')
        elif 'JSE:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.JO')
        elif 'XTAE:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.TA')
        elif 'TSE:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.T')
        elif 'MIC:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.ME')
        elif 'MEX:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.MX')
        elif 'XMAD:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.MC')
        elif 'FRA:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.F')
        elif 'XAMS:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.AS')
        elif 'XBRU:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.BR')
        elif 'XPAR:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.PA')
        elif 'MIL:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1] + '.MI')
        elif 'NAS:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1])
        elif 'NYSE:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1])
        elif 'AMEX:' in listus[num]:
            yahoo_list.append(listus[num].split(':')[1])

    return yahoo_list


def train_model(root_df: pd.DataFrame):

    def pred_data():
        company_list = []
        real_ret = []
        pred_ret = []
        date_list = []

        for company in root_df.index.unique().tolist():
            print('*' * 50)
            print(company)
            for year in range(5):
                # print('*'*50)
                df = root_df.loc[company]
                df['P_BV'] = df['Market_cap'] / df['Book_Value_per_Share'] * df[
                    'Shares_Outstanding_EOP']

                df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()

                end_date = str(2021 - year) + '-01-01'  # предсказываем на год вперед

                if int(df.Date[0][:4]) + 1 < 2014:
                    df_full = df.set_index('Date').copy()
                    df = df.set_index('Date').loc[:end_date]
                    df_native = df.copy()
                    df[['Benchmark_returns_per_year', 'Returns_per_year']] = df[
                        ['Benchmark_returns_per_year', 'Returns_per_year']].shift(-252)
                    df = df[:-252]  # обрезаем по смещённым данным

                    X = df_native[-252:]  # данные фичей которые остались без смещённой доходности

                    # Delete columns with only one unique value
                    for col in df.columns:
                        if len(df[col].unique()) == 1:
                            df.drop(col, inplace=True, axis=1)

                    dataset = df

                    # Extand remaining numbers
                    dataset = dataset.bfill(axis=0)
                    dataset = dataset.ffill(axis=0)

                    #  Replace inf with nan
                    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

                    # рельная доходность в промежутке времени за который мы пытаемся предсказать доходность,
                    # делаем для дальнейшего сравнения
                    real_signal = df_full[len(df) + 252:len(df) + 504]['Returns_per_year']

                    dataset['target'] = dataset['Returns_per_year']

                    X_train = dataset.drop(['Returns_per_year', 'target'], axis=1)  # 'Benchmark_returns_per_year',

                    y_train = dataset['target']

                    col = X_train.columns.tolist()
                    # normalize feach per tebele
                    scaler_feach = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    normal_values = scaler_feach.fit_transform(X_train)
                    X_train = pd.DataFrame(normal_values).set_axis(col, axis=1, inplace=False)

                    X_train = X_train.fillna(0)

                    X = X[X_train.columns.tolist()]

                    # нормализация
                    col = X.columns.tolist()
                    # normalize feach per tebele
                    scaler_feach_test = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    normal_values_test = scaler_feach_test.fit_transform(X)
                    X = pd.DataFrame(normal_values_test).set_axis(col, axis=1, inplace=False)

                    # NEIRON   =======================================================================================

                    def get_compiled_model():
                        model = keras.Sequential()
                        model.add(keras.layers.Dense(64, input_shape=(X_train.shape[1],)))
                        model.add(keras.layers.Dense(8, activation='relu'))
                        model.add(keras.layers.Dense(1, activation='linear'))
                        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
                        model.compile(loss='MSE', optimizer=optimizer, metrics=['MeanSquaredError'])
                        return model

                    callbacks = [
                        EarlyStopping(monitor='loss', min_delta=0.005, patience=20),
                        ModelCheckpoint(filepath='model_linear_loop.h5', save_best_only=True,
                                        monitor='loss', verbose=0)
                    ]

                    model = get_compiled_model()

                    model.fit(np.array(X_train), np.array(y_train), batch_size=256, epochs=100, callbacks=[callbacks],
                              verbose=0, shuffle=False)

                    model_load = load_model('model_linear_loop.h5')
                    pred = model_load.predict(X).flatten()

                    company_list.append(company)
                    print(f"Company {company}")
                    real_ret.append(real_signal[-1])
                    print(f"Real Returns {real_signal[-1]}")
                    pred_ret.append(pred[-1])
                    print(f"Predicted {pred[-1]}")
                    date_list.append(int(end_date.split('-')[0]))
                    print(f"Date print {int(end_date.split('-')[0])}")

        output_df = pd.DataFrame({'Company': company_list,
                                  'Date': date_list,
                                  'Real Return': real_ret,
                                  'Predict Return': pred_ret})

        print(output_df)

        return output_df

    fd = pd.DataFrame()
    my_dict = {}
    print("Starting model training")
    for loop in range(10):
        print(f'loop {loop}')
        output_df = pred_data()
        temp = pd.DataFrame()
        for year in output_df['Date'].unique().tolist():
            filtered_df = output_df.where(output_df.Date == year).dropna()
            temp = pd.concat([temp, filtered_df])

            top_list = filtered_df.sort_values('Predict Return', ascending=False).dropna()[
                       :int(len(filtered_df) * 0.3)].Company.tolist()

            try:
                my_dict[year].extend(top_list)
            except:
                new_dict = {year: top_list}
                my_dict.update(new_dict)

            print('my_dict')
            print(my_dict)

        temp.dropna(inplace=True)
        if loop == 0:
            fd = temp
            fd = fd.drop(['Real Return'], axis=1)
        else:
            fd = pd.concat([fd, temp['Predict Return']], axis=1)

    print(my_dict)
    print(fd)
    print(fd.columns.tolist())
    fd.to_csv("data/Returns_DF.csv", index=False)

    def count_to_dict(lst):
        return {k: lst.count(k) for k in lst}

    last_dict = {}

    for year_in_dict in my_dict.keys():
        print('year_in_dict')
        print(year_in_dict)

        print(count_to_dict(my_dict[year_in_dict]))

        final_df = pd.DataFrame(count_to_dict(my_dict[year_in_dict]), index=[0])

        final_df = final_df.T
        final_df['count'] = final_df.values
        print(final_df)

        sorted_final_df = final_df.sort_values('count', ascending=False)
        sorted_final_df = sorted_final_df.where(sorted_final_df['count'] >= 7).dropna()
        print(sorted_final_df)
        last_new_dict = {year_in_dict: sorted_final_df.index.tolist()}
        last_dict.update(last_new_dict)

    print(last_dict)

    years_list = []
    retunrns_list = []
    bench_list = []

    finish_returns_df = pd.DataFrame()

    benchmark = yf.download('^STI')['Close']

    for year in last_dict.keys():
        print(year)
        top_list = last_dict[year]

        print(top_list)

        yahoo_data = yf.download(convert_yahoo(top_list))
        yahoo_data = yahoo_data.bfill().ffill()['Close']

        print(year)

        company_return = (yahoo_data[str(year):str(year)].iloc[-1] - yahoo_data[str(year):str(year)].iloc[0]) / \
                         yahoo_data[str(year):str(year)].iloc[0]
        bench_return = (benchmark[str(year):str(year)].iloc[-1] - benchmark[str(year):str(year)].iloc[0]) / \
                       benchmark[str(year):str(year)].iloc[0]

        years_list.append(year)
        retunrns_list.append(np.mean(company_return))
        bench_list.append(np.mean(bench_return))

        # weights calc

        assets = convert_yahoo(top_list)

        print_df = pd.DataFrame()
        try:
            print_df['Company'] = company_return.index.tolist()
            print_df['Returns'] = company_return.values
        except:
            print_df['Company'] = top_list[0]
            print_df['Returns'] = company_return

        print_df['Year'] = [year] * len(print_df)
        print_df['Benchmark'] = [bench_return] * len(print_df)
        print_df['Overtaking'] = np.where(print_df['Returns'] > bench_return, 1, 0)
        try:
            finish_returns_df = pd.concat([finish_returns_df, print_df])
        except:
            pass

        print(print_df)

    finish_df = pd.DataFrame({'Year': years_list,
                              'Portfolio Return': retunrns_list,
                              'Benchmark': bench_list})

    print(finish_df)
    print(finish_returns_df.set_index(['Year', 'Company']))

    finish_returns_df.to_csv("data/Weights_DF.csv")
