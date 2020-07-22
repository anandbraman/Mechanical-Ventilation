import csv
import sys


class CSVTracker():
    def __init__(self, filepath):
        self.filepath = filepath
        self.fields = ['model_type', 'model_id', 'epochs', 'learning_rate',
                       'hidden_size', 'loss', 'precision',
                       'recall', 'f1_score', 'auroc', 'accuracy']

    def get_fields(self):
        return self.fields

    def is_model_unique(self, model_id):
        '''
        Uses model id to determine if hyperparam tuning experiment has been
        performed before
        Input: 
            Model ID, a combination of relevant parameters + model type
            ex: Model - LSTM, Epochs - 250, LR - 0.001, HS - 8
            model_id = LSTM_250_001_8
        
        Output:
            None
        '''
        with open(self.filepath, newline='') as f:
            reader = csv.DictReader(f, fieldnames=self.fields)
            for row in reader:
                if row['model_id'] == str(model_id):
                    print('It looks like you have trained this model before.')
                    print('Would you like to train this model again?')
                    ans = input('Enter yes or no: ')
                    ans = str.lower(ans)
                    if ans == 'yes':
                        return
                    elif ans == 'no':
                        sys.exit(0)
        
    def record_experiment(self, params_results: dict):
        '''
        appends a line to csv

        Input:
            parameters and results of experiment

        Output:
            None

        '''
        with open(self.filepath, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(params_results)

            