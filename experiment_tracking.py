import gspread
import sys


class ExperimentTracker():
    def __init__(self, client_secret: str, spreadsheet: str,
                 worksheet: str = 'Sheet1'):
        self.gc = gspread.service_account(client_secret)
        self.spreadsheet = spreadsheet
        self.worksheet = worksheet
        self.sheet = self.gc.open(self.spreadsheet).worksheet(self.worksheet)
        # apparently sheets start index is 1
        self.param_lst = self.sheet.row_values(1)

    def get_param_lst(self):
        return self.param_lst

    def record_experiment(self, lst):
        '''
        pass a list of parameters and results to be recorded
        make sure to keep the order the same across scripts or you'll be 
        causing your own data entry problems
        '''
        
        self.sheet.append_row(lst)

    def unique_params(self, model_id, col_name: str):
        '''
        str, int -> None
        model id and the model id column name in the spreadsheet
        This method opens up the spreadsheet and checks if a model with this id
        has been trained before. I set model id to be a combo of the model type 
        and the tuning params 
        If the combo has been tried, the it will ask if you want to 
        train this model again anyway. If yes, the model trains, if no,
        the script quits
        '''
        cell = self.sheet.find(col_name)
        col = cell.col
        prev_ids = self.sheet.col_values(col)
        
        if model_id in prev_ids:
            print('It looks like you have trained this model before.')
            print('Would you like to train this model again?')
            ans = input('Enter yes or no: ')
            ans = str.lower(ans)

            if ans == 'yes':
                return
            elif ans == 'no':
                sys.exit(0)
        else: 
            return

