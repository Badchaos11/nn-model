import sys

import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.moment_helpers as mh
import yfinance as yf
from cvxopt import matrix
from cvxopt.solvers import qp
from pandas_datareader import data as pd_data
from pypfopt import BlackLittermanModel
from pypfopt import EfficientFrontier, CLA
from pypfopt import black_litterman, risk_models
from pypfopt.exceptions import OptimizationError

pd.options.mode.chained_assignment = None


class BlackLittermanCalc:

    def __init__(self, assets: list, benchmark_ticker: str, returns: dict, lookback: int, max_size: float,
                 min_size: float, test_year=2021, opt_delta=2, risk_free=0.008, budget=2e5, plot_res=False):
        self.__tickers = sorted(assets)
        self.__benchmark_ticker = benchmark_ticker
        self.__lookback = lookback
        self.__max_size = max_size
        self.__min_size = min_size
        self.__risk_free = risk_free
        self.__test_year = test_year
        self.__opt_delta = opt_delta
        self.__budget = budget
        self.__plot_res = plot_res
        self.__currencies = ['CADUSD=X', 'AUDUSD=X', 'TRYUSD=X', 'CHFUSD=X', 'MYRUSD=X', 'MXNUSD=X',
                             'GBPUSD=X', 'SGDUSD=X', 'HKDUSD=X', 'ZARUSD=X', 'ILSUSD=X', 'JPYUSD=X',
                             'RUBUSD=X', 'EURUSD=X']
        self.__views = returns

    def __load_prices(self):
        #ty = datetime.datetime.strptime(self.__test_year, "%Y-%m-%d").date()
        ed = str(self.__test_year - 1) + '-12-30'
        st = str(self.__test_year - self.__lookback) + '-01-01'
        data = yf.download(self.__tickers, st, ed)['Close']

        self.__data = data

    def __load_market_price(self):
        market_prices = yf.download(self.__benchmark_ticker, period="max")["Adj Close"]
        ed = str(self.__test_year - 1) + '-12-30'
        market_prices = market_prices[:ed]
        self.__market_prices = market_prices

    def __load_mkt_caps(self):
        mcaps = pd_data.get_quote_yahoo(self.__tickers)['marketCap']
        missing_mcap_symbols = mcaps[mcaps.isnull()].index
        for symbol in missing_mcap_symbols:
            print('attempting to find market cap info for', symbol)
            data = yf.Ticker(symbol)
            if data.info['quoteType'] == 'ETF' or data.info['quoteType'] == 'MUTUALFUND':
                mcap = data.info['totalAssets']
                print('adding market cap info for', symbol)
                mcaps.loc[symbol] = mcap
            else:
                print('Failed to find market cap for', symbol)
                sys.exit(-1)
        self.__mkt_caps = mcaps

    def __load_mean_views(self):
        mu = {}
        for symbol in sorted(self.__tickers):
            mu[symbol] = self.__views[symbol][1]
        self.__mu = mu

    def __load_full_data(self):
        self.__load_prices()
        self.__load_market_price()
        self.__load_mkt_caps()

    def __calc_omega(self):
        variances = []
        for symbol in sorted(self.__tickers):
            view = self.__views[symbol]
            lb, ub = view[0], view[2]
            std_dev = (ub - lb) / 2
            variances.append(std_dev ** 2)
        omega = np.diag(variances)
        self.__omega = omega

    def __calc_quantity(self, weight_type):
        wgts = pd.read_csv(f'models/portfolio_weight_results.csv').set_index('Unnamed: 0')
        wgts = wgts[weight_type]
        weighted_budget = [self.__budget * wgts[i] for i in range(len(wgts))]
        return weighted_budget

    # def __count_fama(self):
    #
    #     ed = str(self.__test_year - 1) + '-12-30'
    #     st = str(self.__test_year - self.__lookback) + '-01-01'
    #
    #     dat = self.__data
    #     data_bench = yf.download(self.__benchmark_ticker, st, ed)['Close']
    #
    #     data = dat.pct_change().dropna()
    #     bench = data_bench.pct_change().dropna()
    #
    #     ff_ratios = pd.read_excel("models/F-F_Research_Data_5_Factors_2x3_daily.xlsx")
    #     ff_ratios['Date'] = pd.to_datetime(ff_ratios['Unnamed: 0'], format='%Y%m%d')
    #     ff_ratios = ff_ratios.set_index('Date')
    #     ff_ratios = ff_ratios.drop(columns=['Unnamed: 0'])
    #     ff_ratios = ff_ratios.loc[data.index[0]: data.index[-1]]
    #     for i in range(len(ff_ratios)):
    #         ff_ratios['Mkt-RF'][i] = float(ff_ratios['Mkt-RF'][i][:-1])
    #         ff_ratios['SMB'][i] = float(ff_ratios['SMB'][i][:-1])
    #         ff_ratios['HML'][i] = float(ff_ratios['HML'][i][:-1])
    #         ff_ratios['RMW'][i] = float(ff_ratios['RMW'][i][:-1])
    #         ff_ratios['CMA'][i] = float(ff_ratios['CMA'][i][:-1])
    #
    #     bench_df = pd.DataFrame()
    #     bench_df['Benchmark'] = bench
    #
    #     tiker_with_factors = ff_ratios.merge(data, how='right', on=['Date']).bfill(axis='rows')
    #     pre_Y = bench_df.merge(tiker_with_factors, how='right', on=['Date']).bfill(axis='rows')
    #     pre_Y['Mkt-RF'] = (pre_Y['Benchmark'] - pre_Y['RF']) * 100
    #     X = pre_Y[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] / 100  # Mkt-RF =  Benchmark - RF
    #
    #     Y = data
    #     print('Треню модель')
    #     # Create a regression model
    #     reg = sm.OLS(Y.astype(float), X.astype(float)).fit()
    #
    #     regression_df = reg.params
    #     regression_df = regression_df.T
    #     regression_df['Tickers'] = data.columns.tolist()
    #     regression_df = regression_df.set_index('Tickers')
    #     regression_df = regression_df.T
    #
    #     research_data_per_year = pd.read_csv('models/F-F_Research_Data_5_Factors_2x3.csv')
    #     research_data_per_year = research_data_per_year.rename({'Unnamed: 0': 'Date'}, axis=1)
    #     research_data_per_year['Date'] = research_data_per_year['Date'].astype('str')
    #     research_data_per_year = research_data_per_year.set_index('Date') / 100
    #
    #     low = []
    #     mid = []
    #     high = []
    #     tik = []
    #
    #     bench_year_return = (data_bench[-1] - data_bench[0]) / data_bench[0]
    #
    #     for company in regression_df.columns.tolist():
    #         calk_df = regression_df[company]
    #         table_data_year = research_data_per_year.loc[str(self.__test_year - 1)]
    #         try:
    #             mid_pref = calk_df['Mkt-RF'] * bench_year_return + calk_df['SMB'] * table_data_year['SMB'] + calk_df[
    #                 'HML'] * table_data_year['HML'] + \
    #                        calk_df['RMW'] * table_data_year['RMW'] + calk_df['CMA'] * table_data_year['CMA']
    #
    #             mid.append(mid_pref)
    #             low.append(mid_pref - pre_Y[company].std())
    #             high.append(mid_pref + pre_Y[company].std())
    #             tik.append(company)
    #         except:
    #             mid.append(0)
    #             low.append(0)
    #             high.append(0)
    #             tik.append(company)
    #
    #     daa = pd.DataFrame()
    #     daa['ticker'] = tik
    #     daa['low_pred_ret'] = low
    #     daa['pred_ret'] = mid
    #     daa['hig_pred_ret'] = high
    #
    #     out = daa.set_index('ticker').T.to_dict('list')
    #     self.__views = out

    def __calculate_black_litterman(self):

        delta = black_litterman.market_implied_risk_aversion(self.__market_prices)
        covar = risk_models.risk_matrix(self.__data, method='oracle_approximating')
        market_prior = black_litterman.market_implied_prior_returns(self.__mkt_caps, risk_aversion=delta,
                                                                    cov_matrix=covar)
        self.__calc_omega()
        bl = BlackLittermanModel(covar, pi="market", market_caps=self.__mkt_caps, risk_aversion=delta,
                                 absolute_views=self.__mu, omega=self.__omega)
        rets_bl = bl.bl_returns()
        covar_bl = bl.bl_cov()
        self.__rets_bl = rets_bl
        self.__covar_bl = covar_bl
        self.__market_prior = market_prior

    def __kelly_optimise(self):
        M = self.__rets_bl.to_numpy()
        C = self.__covar_bl.to_numpy()

        n = M.shape[0]
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        G = matrix(0.0, (n, n))
        G[::n + 1] = -1.0
        h = matrix(0.0, (n, 1))

        try:
            max_pos_size = float(self.__max_size)
        except KeyError:
            max_pos_size = None
        try:
            min_pos_size = float(self.__min_size)
        except KeyError:
            min_pos_size = None
        if min_pos_size is not None:
            h = matrix(min_pos_size, (n, 1))

        if max_pos_size is not None:
            h_max = matrix(max_pos_size, (n, 1))
            G_max = matrix(0.0, (n, n))
            G_max[::n + 1] = 1.0
            G = matrix(np.vstack((G, G_max)))
            h = matrix(np.vstack((h, h_max)))

        S = matrix((1.0 / ((1 + self.__risk_free) ** 2)) * C)
        q = matrix((1.0 / (1 + self.__risk_free)) * (M - self.__risk_free))
        sol = qp(S, -q, G, h, A, b)
        kelly = np.array([sol['x'][i] for i in range(n)])
        kelly = pd.DataFrame(kelly, index=self.__covar_bl.columns, columns=['Weights'])
        kelly = kelly.round(3)
        kelly.columns = ['Kelly']
        return kelly

    def __max_quad_utility_weights(self):
        print('Begin max quadratic utility optimization')
        returns, sigmas, weights, deltas = [], [], [], []
        # for delta in np.arange(1, 10, 1):
        #     ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
        #                            weight_bounds=(self.__min_size, self.__max_size))
        #     ef.max_quadratic_utility(delta)
        #     ret, sigma, __ = ef.portfolio_performance()
        #     weights_vec = ef.clean_weights()
        #     returns.append(ret)
        #     sigmas.append(sigma)
        #     deltas.append(delta)
        #     weights.append(weights_vec)
        #opt_delta = float(input('Enter the desired point on the efficient frontier: '))
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.max_quadratic_utility(self.__opt_delta)
        opt_weights = ef.clean_weights()
        opt_weights = pd.DataFrame.from_dict(opt_weights, orient='index')
        opt_weights.columns = ['Max Quad Util']
        self.__sigmas = sigmas
        self.__deltas = deltas
        self.__returns = returns
        return opt_weights, ef

    def __min_volatility_weights(self):
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.min_volatility()
        weights = ef.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['Min Vol']
        return weights, ef

    def __max_sharpe_weights(self):
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.max_sharpe()
        weights = ef.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['Max Sharpe']
        return weights, ef

    def __cla_max_sharpe_weights(self):
        cla = CLA(self.__rets_bl, self.__covar_bl,
                  weight_bounds=(self.__min_size, self.__max_size))
        cla.max_sharpe()
        weights = cla.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['CLA Max Sharpe']
        return weights, cla

    def __cla_min_vol_weights(self):
        cla = CLA(self.__rets_bl, self.__covar_bl,
                  weight_bounds=(self.__min_size, self.__max_size))
        cla.min_volatility()
        weights = cla.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['CLA Min Vol']
        return weights, cla

    @staticmethod
    def __plot_heatmap(df, title, x_label, y_label):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25, left=0.25)
        heatmap = ax.pcolor(df, edgecolors='w', linewidths=1)
        cbar = plt.colorbar(heatmap)
        ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
        ax.set_xticklabels(df.columns)  # , rotation=45)
        ax.set_yticklabels(df.index)

        for y, idx in enumerate(df.index):
            for x, col in enumerate(df.columns):
                plt.text(x + 0.5, y + 0.5, '%.2f' % df.loc[idx, col], horizontalalignment='center',
                         verticalalignment='center', )

        plt.gca().invert_yaxis()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def __plot_max_quad_r(self):
        fig, ax = plt.subplots()
        ax.plot(self.__sigmas, self.__returns)
        for i, delta in enumerate(self.__deltas):
            ax.annotate(str(delta), (self.__sigmas[i], self.__returns[i]))
        plt.xlabel('Volatility (%) ')
        plt.ylabel('Returns (%)')
        plt.title('Efficient Frontier for Max Quadratic Utility Optimization')
        plt.show()

    def __plot_black_litterman(self):
        rets_df = pd.DataFrame([self.__market_prior, self.__rets_bl, pd.Series(self.__mu)],
                               index=["Prior", "Posterior", "Views"]).T
        rets_df.plot.bar(figsize=(12, 8), title='Black-Litterman Expected Returns');
        self.__plot_heatmap(self.__covar_bl, 'Black-Litterman Covariance', '', '')
        corr_bl = mh.cov2corr(self.__covar_bl)
        corr_bl = pd.DataFrame(corr_bl, index=self.__covar_bl.index, columns=self.__covar_bl.columns)
        self.__plot_heatmap(corr_bl, 'Black-Litterman Correlation', '', '')

    def calculate_weights(self):
        print('Начинаю расчитывать веса для портфеля')
        print('Загружаю данные')
        self.__load_full_data()
        print(self.__data)
        print('Рассчитываю модель Фама-Френча с 5 параметрами')
        # self.__count_fama()
        print(self.__views)
        self.__load_mean_views()
        print('Рассчитыаю модель Блэка-Литтермана')
        self.__calculate_black_litterman()
        print('Расчет по критерию Kelly')
        kelly_w = self.__kelly_optimise()
        print('Расчет по Max Quad Utility')
        try:
            max_quad_util_w, max_quad_util_ef = self.__max_quad_utility_weights()
        except OptimizationError:
            OptimizationError("cant optimize< maybe lets do less tryes")
        print('Расчёт для минимальной волатильности')
        try:
            min_vol_w, min_vol_ef = self.__min_volatility_weights()
        except OptimizationError:
            OptimizationError("something went wrong")
        print('Расчёт для максимальногео рейтинга Шарпа')
        try:
            max_sharpe_w, max_sharpe_ef = self.__max_sharpe_weights()
        except OptimizationError:
            pass
        print('Расчет для максимального рейтинга Шарпа по CLA')
        try:
            cla_max_sharpe_w, cla_max_sharpe_cla = self.__cla_max_sharpe_weights()
        except IndexError:
            IndexError("there is no enough data")
        print('Расчёт для минимальной волатильности по CLA')
        try:
            cla_min_vol_w, cla_min_vol_cla = self.__cla_min_vol_weights()
        except IndexError:
            IndexError("tehre is no enough data")
        print('Заполнение весов')
        try:
            weights_df = pd.merge(kelly_w, max_quad_util_w, left_index=True, right_index=True)
        except:
            weights_df = kelly_w
        try:
            weights_df = pd.merge(weights_df, max_sharpe_w, left_index=True, right_index=True)
        except:
            pass
        try:
            weights_df = pd.merge(weights_df, cla_max_sharpe_w, left_index=True, right_index=True)
        except: pass

        try:
            weights_df = pd.merge(weights_df, min_vol_w, left_index=True, right_index=True)
        except: pass
        try:
            weights_df = pd.merge(weights_df, cla_min_vol_w, left_index=True, right_index=True)
        except: pass
        weights_df.to_csv(f'models/portfolio_weight_results.csv')

        if self.__plot_res:
            print('Построение графиков')
            self.__plot_heatmap(weights_df, 'Portfolio Weighting (%)', 'Optimization Method', 'Security')
            self.__plot_black_litterman()
            self.__plot_max_quad_r()
        print('Расчёт окончен. Можно переходить к анализу')
        print('******************************')
        self.__weights = weights_df

    def portfolio_calculate(self, weights_type: str) -> list:
        ed = str(self.__test_year) + '-12-30'
        st = str(self.__test_year) + '-01-01'
        df = yf.download(self.__tickers, st, ed, progress=False)['Close'].dropna()
        index_df = yf.download(self.__benchmark_ticker, st, ed, progress=False)['Close'].dropna()
        try:
            b = self.__calc_quantity(weights_type)
            cum_benchmark_returns = index_df.pct_change().dropna().cumsum()
            for_pr = pd.DataFrame()
            for i, name in enumerate(self.__tickers):
                if b[i] != 0:
                    for_pr[name] = df[name]
            portfolio_returns = for_pr.pct_change().dropna()
            portfolio_returns = portfolio_returns.mean(axis=1)
            index_returns = index_df.pct_change().dropna()

            first_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[2]))
            current_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[-1]))
            growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

            shp = ep.stats.sharpe_ratio(portfolio_returns, annualization=252)
            sort = ep.stats.sortino_ratio(portfolio_returns, annualization=252)
            shp_b = ep.stats.sharpe_ratio(index_returns, annualization=252)
            sort_b = ep.stats.sortino_ratio(index_returns, annualization=252)
            inf = ep.stats.excess_sharpe(portfolio_returns,
                                         index_returns)
            dd = ep.stats.max_drawdown(portfolio_returns)
            dd_b = ep.stats.max_drawdown(index_returns)

            """
            cumprod_ret = ((df * quantity).sum(axis=1).pct_change().dropna() + 1).cumprod() * 100
            cumprod_market_ret = (index_df.pct_change().dropna() + 1).cumprod() * 100
            cumprod_ret.index = pd.to_datetime(cumprod_ret.index)
            trough_index = (np.maximum.accumulate(cumprod_ret) - cumprod_ret).idxmax()
            peak_index = cumprod_ret.loc[:trough_index].idxmax()
            maximum_drawdown = 100 * (cumprod_ret[trough_index] - cumprod_ret[peak_index]) / cumprod_ret[peak_index]
            """
            print(f"Результат анализа портфеля по {weights_type}")
            print('********************************')
            print('Количество акций в портфеле:')
            bb = 0
            for i in range(len(self.__tickers)):
                x = int(b[i] // df[self.__tickers[i]][2])
                print(f"Акций компании {self.__tickers[i]} в портфеле: {x}")
            print('********************************')
            print('Первоначальная стоимость портфеля', round(first_sum_portfolio))
            print('Текущая стоимость портфеля', round(current_sum_portfolio))
            print('***Результат анализа портфеля***')
            print(f"Доходность портфеля: {round(growth_portfolio * 100, 2)}%")
            print(f"Доходность портфеля годовая: {round(((growth_portfolio * 100) / len(df)) * 252, 2)}%")
            print(f"Sharpe ratio: {round(shp, 4)}")
            print(f"Sortino ratio: {round(sort, 4)}")
            print(f"Information ratio: {round(inf, 4)}")
            print(f"Max Drawdown: {round((dd * 100), 2)}%")
            print("*******************************")
            print("****Результат анализа рынка****")
            print(f"Доходность индекса: {round((cum_benchmark_returns[-1] * 100), 2)}%")
            print(f"Доходность индекса годовая: {round(((cum_benchmark_returns[-1] * 100) / len(df)) * 252, 2)}%")
            print(f"Sharpe ratio for benchmark: {round(shp_b, 4)}")
            print(f"Sortino ratio for benchmark: {round(sort_b, 4)}")
            print(f"Max Drawdown benchmark: {round((dd_b * 100), 2)}%")
            print('*******************************')

            return [weights_type, round(growth_portfolio * 100, 2), round(shp, 4), round(sort, 4), round((dd * 100), 2)]
        except:
            print('Такого типа весов несуществует, попробуйте один из этих: \n'
                  'Kelly, Max Quad Util, Max Sharpe, CLA Max Sharpe ,Min Vol ,CLA Min Vol')
            return []
