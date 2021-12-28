
import pandas_datareader as web
import datetime

train_start_date = datetime.datetime(2015, 1, 1)
train_end_date = datetime.datetime(2017, 12, 31)
valid_start_date = datetime.datetime(2018, 1, 1)
valid_end_date = datetime.datetime(2018, 12, 31)
test_start_date = datetime.datetime(2019, 1, 1)
test_end_date = datetime.datetime(2019, 12, 31)

def collect_data_to_csv(stock):
    #use yahoo finance API to get data
    #train data
    df_train = web.DataReader(stock, 'yahoo', train_start_date, train_end_date)
    train_file = 'data/{}.csv'.format(stock)
    df.to_csv(train_file)
    df_valid = web.DataReader(stock, 'yahoo', valid_start_date, valid_end_date)
    valid_file = 'data/{}_2018.csv'.format(stock)
    df.to_csv(valid_file)
    df_test = web.DataReader(stock, 'yahoo', test_start_date, test_end_date)
    test_file = 'data/{}_2019.csv'.format(stock)
    df.to_csv(test_file)
    return "Done!"