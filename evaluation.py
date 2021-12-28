"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> <stock-name> [--k-pos=<k-positive>] [--k-neg=<k_negative>]
  [--test] [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --k-pos=<k-positive>            satisfaction function exponent when profit is >= 0. [default: 1]
  --k-neg=<k-negative>            satisfaction function exponent when profit is < 0 [default: 1]
  --test                          specifies whether it is a test mode
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

import os
import coloredlogs
import pandas as pd
import csv

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)


def eval_main(eval_stock, stock, k_pos, k_neg, window_size, model_name, debug, test):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(stock, window_size, pretrained=True, model_name=model_name)
        profit, _, output_lst = evaluate_model(agent, data, window_size, debug, k_pos, k_neg, test)
        #profit, _ = evaluate_model(agent, data, window_size, debug, k_pos, k_neg, test)
        show_eval_result(model_name, profit, initial_offset)
        #if testing output csv 
        if test:
            filename = "csv_output/{}_{}_{}.csv".format(agent.stock, int(k_pos*10), int(k_neg*10))
            headers = ['StockName', 'Time', 'Action', 'Profit'] 
            df = pd.DataFrame(output_lst, columns=headers)
            df.to_csv(filename, index=False)
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(stock, window_size, pretrained=True, model_name=model)
                profit = evaluate_model(agent, data, window_size, debug, k_pos, k_neg)
                show_eval_result(model, profit, initial_offset)
                del agent


# if __name__ == "__main__":
#     args = docopt(__doc__)

#     eval_stock = args["<eval-stock>"]
#     stock = args["<stock-name>"]
#     k_pos = float(args["--k-pos"])
#     k_neg = float(args["--k-neg"])
#     test = bool(args["--test"])
#     window_size = int(args["--window-size"])
#     model_name = args["--model-name"]
#     debug = args["--debug"]

#     coloredlogs.install(level="DEBUG")
#     switch_k_backend_device()

#     try:
#         main(eval_stock, stock, k_pos, k_neg, window_size, model_name, debug, test)
#     except KeyboardInterrupt:
#         print("Aborted")
