import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 
import numpy as np 
import datetime 
from sklearn.ensemble import RandomForestRegressor


stores_data = pd.read_csv('store.csv')

#comptetion distance has only three missing values we can replace these three null values with median
stores_data['CompetitionDistance'].fillna(stores_data['CompetitionDistance'].median(), inplace = True)

#Filling all null values with zero in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear'
stores_data['CompetitionOpenSinceMonth'].fillna(0, inplace = True )
stores_data['CompetitionOpenSinceYear'].fillna(0, inplace = True)

promo_columns = ['Promo2SinceWeek','Promo2SinceYear','PromoInterval']
for col in promo_columns:
  stores_data[col].fillna(0, inplace = True)

#function to replace a,b,c in assortment with real meaningful names
def assortment_name_conversion(x):
  if x =='a':
    return 'Basic'
  if x=='b':
    return 'Extra'
  if x=='c':
    return 'Extended'

#applying above defined function to change the values of assortment feature accordingly


stores_data['Assortment'] = stores_data['Assortment'].apply(assortment_name_conversion)

encoder_dict = dict({'Assortment': {'Basic': 0, 'Extended': 1, 'Extra': 2},
                 'PromoInterval': {'0': 0,'Feb,May,Aug,Nov': 1,'Jan,Apr,Jul,Oct': 2,'Mar,Jun,Sept,Dec': 3},
                 'StoreType': {'a': 0, 'b': 1, 'c': 2, 'd': 3}})

app = Flask(__name__)


model = pickle.load(open('final_rfmodel.sav', 'rb'))


#defining a function to get number of weeks since the promo2 has started from current sales date
def promo2_time_conversion(week,year,date):
  if week != 0:
    promo_since = str(year) + "-" + str(week)   #to store year and week together in a variable promo_since
    #considering promo2 started from 1st weekday of given year and month and coverting promo_since format from string to datetime
    promo_since = datetime.datetime.strptime(promo_since + "-1", "%Y-%W-%w")   
    return ((datetime.datetime.strptime(date,"%Y-%m-%d") - promo_since)/7).days   #returning no. of weeks between promo_since and given date 
  else:
    return 0 

#defining a function to get number of months since the competition store has started to current sales date
def competition_time_conversion(month,year,date):
  if month != 0 and year!=0:
    comp_since = str(year) + "-" + str(month)   #to store year and month together in a variable comp_since
    comp_since = datetime.datetime.strptime(comp_since + "-1", "%Y-%m-%d")   #coverting comp_since format from string to datetime
    return ((datetime.datetime.strptime(date,"%Y-%m-%d") - comp_since)/30).days   #returning no. of months between comp_since and given date 
  else:
    return 0 

#class for final predictions
class Rossman:

  #function to clean out the data i.e. null or missing value treatment
  def data_cleaning(self,data):   
    #merging data with stores data 
    data = pd.DataFrame(data, index = [0])
    data['Store'] = data['Store'].astype('int64')
    final_data = pd.merge(data,stores_data, on = 'Store', how = 'inner')

    #moving a current date to date column if date is not present in a row
    final_data['Date']  = final_data['Date'].apply(lambda x : date.today().strftime('%Y-%m-%d') if pd.isna(x) or x=='' else x)

    #getting a day of week from date to fill if day of week is not present 
    final_data['DayOfWeek'] = final_data['DayOfWeek'].apply(lambda x : x.weekday if pd.isna(x) else x)

    #for all other features null value tratment has already done during training and will replicate same here 
    
    return final_data

  
  def feature_engineering(self,final_data):
    #extracting month from date column
    final_data['month'] = pd.DatetimeIndex(final_data['Date']).month
    
    #converting promo2_weeks from Promo2SinceWeek and Promo2SinceYear using previousluy defined function in block 55
    final_data['promo2_weeks'] = final_data.apply(lambda x:promo2_time_conversion(int(x['Promo2SinceWeek']),
                                                                               int(x['Promo2SinceYear']),x['Date']),axis=1)
    
     #converting comp_months from comptetionopensincemonth and comptetionopensinceyear using previousluy defined function in block 67
    final_data['comp_months'] = final_data.apply(lambda x:competition_time_conversion(int(x['CompetitionOpenSinceMonth']),
                                                                                              int(x['CompetitionOpenSinceYear']),x['Date']),axis=1)
    
    #selecting the features only needed in our model prediction
    final_data = final_data[['DayOfWeek', 'Customers', 'Promo', 'StoreType', 'Assortment', 'CompetitionDistance', 'PromoInterval', 'month',
                             'promo2_weeks', 'comp_months']]
    
   
    #label encoding using previously defined label encoder to ensure same encoding has been applied
    for feature in encoder_dict.keys():
        mapping_dict = encoder_dict[feature]
        value = final_data[feature].values[0]
        value = str(value)
        final_data[feature] = mapping_dict[value]
        
    
    #returning prediction ready data 
    return final_data
    
  def prediction(self,data,final_data):

    #predicting output 
    output = model.predict(final_data)
    
    #creating a dataframe that will store predictor and predicted variable 
    predicted_df = data
    predicted_df['Predicted Sales'] = output

    #for feature in encode_features:
      #predicted_df[feature] = encoder_dict[feature].inverse_transform(predicted_df[feature]) 

    #returning the created dataframe
    return predicted_df 

def rossman_prediction(unit):

    #pipeline for predicting rossman sales
    pipeline = Rossman()

    #checking if whole data becomes null after treating and if not go further  
    if len(unit) >0:
      data = pipeline.data_cleaning(unit)
      final_data = pipeline.feature_engineering(data)
      prediction  = pipeline.prediction(data,final_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.items()]
    features = dict(features)
    if features['Open']!= '0' and int(features['Customers'])>0:
        prediction = rossman_prediction(features)
        op= prediction['Predicted Sales'].values[0]
        date = prediction['Date'].values[0]
        output = round(op, 2)
    else:
        output = 0
    output_dict = features
    output_dict['Sales'] = output


    return render_template('predict.html', output_dict=output_dict)
    
if __name__ == "__main__":
    app.run(debug=True)
