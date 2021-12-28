"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> <stock-name> [--strategy=<strategy>]
    [--k-pos=<k-positive>] [--k-neg=<k_negative>] [--test]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --k-pos=<k-positive>            satisfaction function exponent when profit is >= 0. [default: 1]
  --k-neg=<k-negative>            satisfaction function exponent when profit is < 0 [default: 1]
  --test                          specifies whether it is a test mode 
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def train_main(train_stock, val_stock, stock, window_size, batch_size, ep_count,
         k_pos=1, k_neg=1, strategy="t-dqn", model_name="model_debug", 
         pretrained=False, debug=False, test=False):
    """ Refer to original trading-bot for more details
    @misc{github,
          author={pskrunner14/trading-bot},
          title={GitHub},
          year={2019},
          url={https://github.com/pskrunner14/trading-bot.git},}
    """
    agent = Agent(stock, window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, k_pos=k_pos, k_neg=k_neg,
                                   ep_count=ep_count, batch_size=batch_size,
                                   window_size=window_size)
        val_result, _, _= evaluate_model(agent, val_data, window_size, debug, k_pos, k_neg, test)
        show_train_result(train_result, val_result, initial_offset)


# if __name__ == "__main__":
#     args = docopt(__doc__)

#     train_stock = args["<train-stock>"]
#     val_stock = args["<val-stock>"]
#     stock = args["<stock-name>"]
#     strategy = args["--strategy"]
#     k_pos = float(args["--k-pos"])
#     k_neg = float(args["--k-neg"])
#     test = bool(args["--test"])
#     window_size = int(args["--window-size"])
#     batch_size = int(args["--batch-size"])
#     ep_count = int(args["--episode-count"])
#     model_name = args["--model-name"]
#     pretrained = args["--pretrained"]
#     debug = args["--debug"]

#     coloredlogs.install(level="DEBUG")
#     switch_k_backend_device()

#     try:
#         main(train_stock, val_stock, stock, window_size, batch_size,
#              ep_count, k_pos, k_neg, strategy=strategy, model_name=model_name, 
#              pretrained=pretrained, debug=debug, test=test)
#     except KeyboardInterrupt:
#         print("Aborted!")
