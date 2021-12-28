from train import train_main
from evaluation import eval_main



def main():
    stocks = {'GOOG': {'train':'data/GOOG.csv', 'valid':'data/GOOG_2018.csv',
                        'test':'data/GOOG_2019.csv'},
              'AAPL': {'train':'data/AAPL.csv', 'valid':'data/AAPL_2018.csv',
                        'test':'data/AAPL_2019.csv'},
              'JNJ': {'train':'data/JNJ.csv', 'valid':'data/JNJ_2018.csv',
                        'test':'data/JNJ_2019.csv'}}
    k = [(1,1), (1.5,0.5), (1.5,0.7), (1.5,0.9), (1.7,0.5), (1.7,0.7),
          (1.7,0.9), (1.9,0.5), (1.9,0.7), (1.9,0.9), (0.5,1.5), (0.5,1.7),
          (0.5,1.9),(0.7,1.5), (0.7,1.7), (0.7,1.9), (0.9,1.5), (0.9,1.7),
          (0.9,1.9)]
    #loop through all stocks
    for stock in stocks:
        print(stock)
        #loop through all pairs of hyperparameters
        for k_pos, k_neg in k:
            print("kpositive", k_pos, "knegative", k_neg)
            train = stocks[stock]['train']
            valid = stocks[stock]['valid']
            test = stocks[stock]['test']
            print("Training")
            train_main(train, valid, stock, window_size=10, batch_size=32,
                       ep_count=50, k_pos=k_pos, k_neg=k_neg, strategy="t-dqn", model_name="model_debug", 
                       pretrained=False, debug=False, test=False)
            print("Testing")
            model_name = "{}_{}_{}".format(stock, int(k_pos*10), int(k_neg*10))
            eval_main(test, stock, k_pos=k_pos, k_neg=k_neg, window_size=10,
                            model_name=model_name, debug=False, test=True)
    print("Done!")
    return None



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted!")
