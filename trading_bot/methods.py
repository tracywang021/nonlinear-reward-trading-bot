import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)

def satisfaction(x, k_pos, k_neg):
    #customized satisfaction function
    try:
        if x>=0:
            return x**k_pos
        else:
            return -((-x)**k_neg)
    except Exception as err:
        print("Error in satisfaction: " +  err)

def train_model(agent, episode, data, k_pos, k_neg, ep_count=100, batch_size=32, window_size=10):
        """ Refer to original trading-bot for more details
    @misc{github,
          author={pskrunner14/trading-bot},
          title={GitHub},
          year={2019},
          url={https://github.com/pskrunner14/trading-bot.git},}
    """
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = satisfaction(delta, k_pos, k_neg)
            #reward = delta #max(delta, 0)
            total_profit += delta

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 50 == 0:
        agent.save(k_pos, k_neg)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug, k_pos, k_neg, test):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    output_lst = [] #used to output csv file
    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):        
        tmp_output = [] #init output for data t
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            history.append((data[t], "BUY"))
            #update tmp_output
            tmp_output.append(agent.stock) #append stockname
            tmp_output.append(t) #append time
            tmp_output.append("BUY") #append action
            tmp_output.append(None) #append profit for this time
            #update output_lst
            output_lst.append(tmp_output)
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            #reward = delta #max(delta, 0)
            reward = satisfaction(delta, k_pos, k_neg)
            total_profit += delta
            history.append((data[t], "SELL"))
            #update tmp_output
            tmp_output.append(agent.stock) #append stockname
            tmp_output.append(t) #append time
            tmp_output.append("SELL") #append action
            tmp_output.append(delta) #append profit for this time
            #update output_lst
            output_lst.append(tmp_output)
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))
        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history, output_lst
