from models.FamaFrench5 import BlackLittermanCalc
from nn.functions import train_model
from utils.add_returns import extend_df
import pandas as pd


if __name__ == "__main__":
    """В папку data необходимо положить файл для нейросети. В 12 строке указывается его имя.
    Дальнейшие действия идут автоматически."""
    print("Starting")
    # Первый этап. Чтение данных для тренировки нейросети и предсказания доходностей.
    print("Read for NN")
    root_df = pd.read_csv('data/QQQ.csv')
    root_df = root_df.set_index('Unnamed: 0')
    print("Starting NN")
    # train_model(root_df)  # Функция работы с нейросетью: тренировка, получение файлов.
    # Второй этап. Объединение полученных данных. Добавление предсказанных доходностей к лучшим компаниям.
    print('Read data to extend')
    extender = pd.read_csv("data/Returns_DF.csv")
    extandable = pd.read_csv("data/Weights_DF.csv")
    print("Extending")
    extend_df(extender, extandable)  # Функция для расширения.
    # Третий этап. Развесовка портфелей различными методами и проверка их доходностей.
    to_model = pd.read_csv("data/Extended_for_BL.csv")
    to_model = to_model.drop(columns=["Returns", "Benchmark", "Overtaking"])
    list_testers = ['Kelly', 'Max Quad Util', 'Max Sharpe', 'CLA Max Sharpe', 'Min Vol']  # Список методов развесовки.
    result_df = pd.DataFrame()  # Датафрэйм для записи.
    # Проход по каждому году для каждого метода.
    for year in to_model.Year.unique().tolist():
        if year == 2017:
            print("Temporary problems")
            continue
        # Получение параметров для развесовки портфеля
        one_year_data = to_model.where(to_model.Year == year).dropna()
        one_year_data = one_year_data.drop(columns=["Year"])
        views = one_year_data.set_index("Company").T.to_dict("list")
        assets = one_year_data.Company.tolist()
        backtest_year = int(year)
        # Класс равесовки
        model = BlackLittermanCalc(assets=assets, benchmark_ticker="QQQ", returns=views, lookback=1,
                                   max_size=0.35, min_size=0.0, test_year=backtest_year)
        model.calculate_weights()  # Расчёт весов

        res_list = []
        # Проход по методам развесовки для оценки доходности
        for name in list_testers:
            result = model.portfolio_calculate(name)
            res_list.append(result)
        # Создание датафрэйма для составления общего
        to_concat = pd.DataFrame(res_list, columns=["Weights Type", "Returns", "Sharpe", "Sortino", "Max Drawdown"])
        s = ', '.join(assets)
        to_concat.insert(0, "Companies", s)
        to_concat.insert(1, "Year", backtest_year)
        # Объединение датафрэйма за 1 год с общим
        result_df = pd.concat([result_df, to_concat])
    # Отбрасываем пустые строки, если етоды не отработают как надо
    result_df = result_df.dropna()
    result_df = result_df.set_index(["Year", "Companies"])
    # Сохранение результата
    result_df.to_csv("data/Final_Result.csv")
    print(result_df)





