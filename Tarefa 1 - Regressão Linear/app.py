"""
Todo: Atividade de Regressão Linear

Abaixo existe um arquivo de dados. Ele tem 100 medidas (linhas) de 8 preditores (as primeiras oito colunas, "x1...x8" ) 
e os resultados da medida da resposta (última coluna "y").  Estes dados foram gerados por simulação e portanto existe uma 
relação pre-estabelecida que pode ser capturada por regressão linear. 

O objetivo de vocês é descobrir qual é esta relação.
As seguintes situações podem ocorrer:

! 1) Nem todas as variáveis são relevantes, algumas não predizem y e devem ser descartadas. 

! 2) Pode haver colinearidade entre as variáveis e portanto algumas variáveis devem ser descartadas.

! 3) Pode haver interação entre duas variáveis e isto deve ser levado em conta.

! 4) Pode haver não linearidade do tipo polinomial em alguma variável.
"""
import statsmodels.regression.linear_model as sm
from plots import heatmap
import numpy as np
import pandas as pd
import os


def spec_values(row: pd.Series):
    """Função apenas para converter uma linha em valores."""
    values = row[-1].tolist()
    y = values.pop(-1)

    mean = np.mean(values)
    std = np.std(values)
    return mean, std, y, values

    
def view_data(dataframe: pd.DataFrame):
    """Apenas para observar."""
    for row in dataframe.iterrows():
        mean, std, y, values = spec_values(row)
        print(f'Mean: {mean:5f} | Std.: {std:5f} | Y: {y:5f}')

        for j in values:
            if mean + std >= j >= mean - std:
                print(f'Out: {j}')

            else:
                print(f'In.: {j}')


def corr_matrix(dataframe: pd.DataFrame):
    corr = dataframe.corr()
    return corr


def train_data_df(dataframe: pd.DataFrame):
    # train_data, test_data
    test_data, train_data = np.split(dataframe.sample(frac=1, random_state=42), [int(0.2 * len(dataframe))])
    return test_data, train_data


if __name__ == "__main__":
    
    os.system('cls')

    df = pd.read_csv('Tarefa 1 - Regressão Linear\dados3.csv')
    df = df.drop(columns=['Unnamed: 0'])
    
    corr = corr_matrix(df)
    # heatmap(corr).show() # x4 -> descartada

    df = df.drop(columns=['x4'])
    # dados para testar e também treinar
    test_data, train_data = train_data_df(df)
    
    col_names = df.drop(columns='y').columns
    res, predict_x = [], []
    for k, x in enumerate(col_names):
        res.append(sm.OLS.from_formula(formula= 'y ~ ' + x, data=train_data).fit())
        predict_x.append(res[k].predict(test_data))

        # print(res[k].t_test(np.identity(2)))
        # print('\n'*3)
    

    # utilizando o x1
    res_x2 = []
    predict_x2 = []
    for i in range(1, len(df.columns) - 1):
        # y ~ x1 + x2
        formula = 'y ~ x1 + ' + str(col_names[i])
        res_x2.append(sm.OLS.from_formula(formula=formula, data=train_data).fit())
        predict_x2.append(res_x2[i-1].predict(test_data))
        
        # print(res_x2[i-1].t_test(np.identity(3)))
        # print('\n'*3)


    # utilizando o x7
    res_x7, predict_x7 = [], []
    for i in range(1, len(df.columns) - 1):
        if i is not 5:
            formula = 'y ~ x1 + x7 + ' + str(col_names[i])
            res_x7.append(sm.OLS.from_formula(formula = formula, data = train_data).fit())

            # ! array virando 0 -> coloquei no stack overflow o erro
            predict_x7.append(res_x7[i-1].predict(test_data))

            # print(res_x7[i - 1].t_test(np.identity(4)))
            # print('\n'*3)
        
        else:
            res_x7.append(0)
            

    # utilizando o x8
    res_x8, predict_x8 = [], []
    for i in range(1, len(df.columns) - 3):
        formula = 'y ~ x1 + x7 + x8 + ' + str(col_names[i])
        res_x8.append(sm.OLS.from_formula(formula = formula, data = train_data).fit())
        predict_x8.append(res_x8[i-1].predict(test_data))
        
        # print(formula)
        # print(res_x8[i-1].summary())
        # print('\n'*3)


    # utilizando o x6
    res_x6, predict_x6 = [], []
    for i in range(1, len(df.columns) - 4):
        formula = 'y ~ x1 + x7 + x8 + x6 + ' + str(col_names[i])
        res_x6.append(sm.OLS.from_formula(formula = formula, data = train_data).fit())
        predict_x6.append(res_x6[i-1].predict(test_data))
        
        print(formula)
        print(res_x6[i-1].summary())
        print('\n'*3)

