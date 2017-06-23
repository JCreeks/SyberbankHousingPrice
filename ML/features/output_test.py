import pandas as pd
import data_utils
_,test,_=data_utils.load_data()
print(test['id'].head())