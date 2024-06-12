import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from sqlalchemy import create_engine
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import math
import shap
import random
import warnings
warnings.filterwarnings("ignore")
# JavaScript Important for the interactive charts later on
shap.initjs()
#Change working directory to save plots
os.chdir('C:/Users/obriene/Projects/ED/Ambulance Wait Times Explainable AI/Plots')

#############################################################################
#Create the engines
cl3_engine = create_engine('mssql+pyodbc://@cl3-data/DataWarehouse?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
realtime_engine = create_engine('mssql+pyodbc://@dwrealtime/RealTimeReporting?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
#sql querys
att_query = """SET NOCOUNT ON
SELECT NCAttendanceId AS NCAttendanceID, HospitalNumber, PatientAgeOnArrival, PatientGenderIPM,
       AmbulanceArrivalDateTime, AmbulanceCallSign, IsNewAttendance,
       MainDiagnosisECDSGroup1, IsInjury, TriageCategory, ClinicalFrailtyScore,
       ActualDischargeDestinationDescription, AdmitPrvspRefno
FROM [DataWarehouse].[ED].[vw_EDAttendance]
WHERE AmbulanceArrivalDate IS NOT NULL
AND AmbulanceArrivalDate BETWEEN '01-apr-2023' AND '31-mar-2024'
ORDER BY AmbulanceArrivalDateTime desc
"""

loc_query = """SET NOCOUNT ON
SELECT loc.NCAttendanceID, loc.Bed,
       datediff(minute,loc.startdatetime, loc.enddatetime) as AmbHandoverTime,
       loc.LocationOrder, loc2.LocationDescription as NextLocation,
       loc2.Bed AS NextLocationBed
FROM [DataWarehouse].[ED].[vw_EDAttendanceLocationHistory] loc
LEFT JOIN [DataWarehouse].[ED].[vw_EDAttendanceLocationHistory] loc2
ON loc.NCAttendanceId = loc2.NCAttendanceId
AND loc.EndDateTime = loc2.StartDateTime
WHERE (loc.StartDateTime BETWEEN '01-apr 2023 00:00:00' AND '31-mar-2024 23:59:59')
AND (loc.LocationOrder = '1' OR loc.LocationOrder = '2')
AND (loc.Bed = '999 Waiting Area' OR loc.Bed = 'PRE-REG STROKE')
"""

occ_query = """SET NOCOUNT ON
SELECT Date, Data
FROM [RealTimeReporting].[PCM].[operational_dashboard_snapshot]
WHERE Measure_Description = 'In Department'
AND (Date BETWEEN '01-apr 2023 00:00:00' AND '31-mar-2024 23:59:59')
ORDER BY date desc"""

#Read querys into dataframes
att_df = pd.read_sql(att_query, cl3_engine)
loc_df = pd.read_sql(loc_query, cl3_engine)
occ_df = pd.read_sql(occ_query, realtime_engine).rename(columns={'Data':'EDOccupancy'})
print('Data read from SQL')
#Close the connections
cl3_engine.dispose()
realtime_engine.dispose()

#######################################################################################################
#Some patients are pre-reg stroke, which means they have an extra event before arriving in an ambulance
#where they are booked in ahead due to the severity of strokes.  Where this is the first event, we will
#want to take the event after it for ambulance handover time if that event is 999 Waiting Area.

#Filter out anywhere that the second event is pre-reg stroke
filter_loc = loc_df.loc[~((loc_df['Bed'] == 'PRE-REG STROKE')
                          & (loc_df['LocationOrder'] == 2))]
#Group by incident ID and sum the location order
sum = (filter_loc.groupby('NCAttendanceID', as_index=False)['LocationOrder']
       .sum().rename(columns={'LocationOrder':'Sum'}))
#Merge this sum back onto original dataframe
filter_loc = filter_loc.merge(sum, on='NCAttendanceID')
#We then only want to keep anywhere where bed is 999 waiting are and the sum is 1 (first location)
#Or 3 (first location was pre-reg stroke and second is 999 waiting area).  We have already filtered
#out other bed locations, so anything where sum is 2 had a different first bed location
filter_loc = (filter_loc.loc[(filter_loc['Bed'] == '999 Waiting Area')
                            & ((filter_loc['Sum'] == 1) | (filter_loc['Sum'] == 3))]
                            .drop(['Bed', 'LocationOrder', 'Sum'], axis=1))

#Merge this location data onto the attendance data to get full dataframe.
df = att_df.merge(filter_loc, on='NCAttendanceID', how='inner')

#Get capacity data merged onto df by rounding times to the nearest hour
df['Date'] = df['AmbulanceArrivalDateTime'].dt.round('H')
occ_df['Date'] = occ_df['Date'].dt.round('H')
df = df.merge(occ_df, on='Date', how='left')
df = df.sort_values(by='Date')
df['EDOccupancy'] = df['EDOccupancy'].interpolate(method='linear').round().astype(int)
df = df.drop('Date', axis=1)
##################################################################################################
#remove outliers
df = df.loc[((df['AmbHandoverTime'] >= 0) & (df['AmbHandoverTime'] <= 2880))
            & ~(df['NextLocation'].isin(['ED Paeds', 'ED Paediatrics']))].copy()
#Encode required columns
df['PatientGenderIPM'] = df['PatientGenderIPM'].str.extract(r'(\d+)').astype(int)
df['IsNewAttendance'] = np.where(df['IsNewAttendance'] == 'Y', 1, 0)
df['TriageCategory'] = df['TriageCategory'].fillna(0).astype(int)
df['ClinicalFrailty'] = (np.where(df['ClinicalFrailtyScore'].str.extract(r'(\d+)').astype(float) >= 5,
                                  1, 0))
df['MentalHealth'] = np.where(df['MainDiagnosisECDSGroup1'] == 'Psych / tox / D+A', 1, 0)
df['Admitted'] = np.where(df['AdmitPrvspRefno'] > 0, 1, 0)
df['NextLocation'] = df['NextLocation'].replace({'ED Ambulatory|ED AMBULATORY' : 1,
                                                 'Majors HOLD|ED Majors Hold|ED Majors' : 2,
                                                 'ED Resus' : 3}, regex=True).fillna(0)

#################################################################################################
#Split train/test and fit model
y = df['AmbHandoverTime']
X = df[['PatientAgeOnArrival', 'PatientGenderIPM', #'AmbulanceArrivalDateTime',
        'IsNewAttendance', 'IsInjury', 'TriageCategory', 'NextLocation', 'ClinicalFrailty',
        'MentalHealth', 'Admitted', 'EDOccupancy']]

#Feature selection
# parameters to be tested on GridSearchCV
params = {"alpha":np.arange(0.00001, 10, 500)}
# Number of Folds and adding the random state for replication
kf = KFold(n_splits=5,shuffle=True, random_state=42)
# Initializing the Model
lasso_model = Lasso()
# GridSearchCV with model, to find best alpha paramter.
lasso_cv = GridSearchCV(lasso_model, param_grid=params, cv=kf)
lasso_cv.fit(X, y)
# calling the model with the best parameter
print(lasso_cv.best_params_)
alpha_param = float(lasso_cv.best_params_['alpha'])
lasso1 = Lasso(alpha=alpha_param)
lasso1.fit(X, y) 
lasso1_coef = np.abs(lasso1.coef_)

# plotting the Column Names and Importance of Columns. 
plt.bar(X.columns, lasso1_coef)
plt.xticks(rotation=30)
plt.grid()
plt.title("Feature Selection Based on Lasso")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('Lasso Feature Selection.png')
plt.close()

# Subsetting the features which has more than 15 importance.
feature_subset = np.array(X.columns)[lasso1_coef>15]
X = X[feature_subset].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

#Fitting the model
#Grid search for best parameters (this is slow, took around 2 hours).
#params = {"learning_rate": np.arange(0, 0.5, 0.05),
 #         "max_depth": np.arange(3, 11, 1),
  #        "min_child_weight": np.arange(0, 11, 1),
   #       "gamma": np.arange(0, 0.03, 0.01),
    #      "colsample_bytree": np.arange(0.6, 1.01, 0.1)}
# Number of Folds and adding the random state for replication
#kf = KFold(n_splits=5,shuffle=True, random_state=42)
# Initializing the Model
#model = xgb.XGBRegressor()
# GridSearchCV with model, to find best alpha paramter.
#XGB_cv = GridSearchCV(model, param_grid=params, cv=kf)
#XGB_cv.fit(X, y)
#print(XGB_cv.best_params_)

#Fit the model
model = xgb.XGBRegressor(learning_rate=0.02, max_depth=4, min_child_weight=5,
                         gamma=0.01, colsample_bytree=0.8)
model.fit(X_train, y_train)

#Print model metrics
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
train_RMSE =  root_mean_squared_error(y_train, y_train_pred)
test_RMSE = root_mean_squared_error(y_test, y_pred)
train_R2 = r2_score(y_train, y_train_pred)
test_R2 = r2_score(y_test, y_pred)
print('Training RMSE: ' + str(train_RMSE))
print('Testing RMSE: ' + str(test_RMSE))
print('Training R2: ' + str(train_R2))
print('Testing R2: ' + str(test_R2))

# evaluate model using RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

# Check Actual Vs Predictions
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
x_max = 50 * math.ceil(y_test.max() / 50)
y_max = 50 * math.ceil(y_pred.max() / 50)
ax.scatter(y_test, y_pred)
ax.set_title('Actual Vs Predicted Ambulance Handover Values')
ax.set_ylabel('Predicted Ambulance Handover')
ax.set_xlabel('Actual Ambulance Handover')
ax.plot([0, x_max], [0, y_max], color='r', linestyle='-', linewidth=2)
plt.grid()
plt.tight_layout()
plt.savefig('Actual Vs Predicted Ambulance Handover Values.png')
plt.close()

# Obtain shap values
shap_values = shap.Explainer(model).shap_values(X_test)
# Summary - SHAP value bar chart
shap.summary_plot(shap_values,
                  X_test,
                  plot_type="bar",
                  show=False)
plt.savefig('SHAP interaction values bar.png')
plt.close()
# Summary - Beeswarm plot
shap.summary_plot(shap_values,
                  X_test,
                  show=False)
plt.savefig('SHAP interaction values beeswarm.png')
plt.close()
# Summary - Violin plot
shap.summary_plot(shap_values,
                  X_test,
                  plot_type="violin",
                  show=False)
plt.savefig('SHAP interaction values violin.png')
plt.close()

# Now to create a dependence plot for each
for e, i in enumerate(X_test.columns):
   shap.dependence_plot(e, shap_values, X_test, show=False)
   plt.savefig(f'{i} dependence plot.png')
   plt.close()

# compute SHAP values (different to above)
train_explainer = shap.Explainer(model, X_train)
test_explainer = shap.Explainer(model, X_test)
train_shap_values = train_explainer(X)

#Create some waterfall plots for random ids
for idx in random.sample(X_train.index.tolist(), 10):
    fig = plt.figure()
    shap.plots.waterfall(train_shap_values[idx], show=False)
    plt.savefig(f'SHAP waterfall plot id {str(idx)}.png', bbox_inches='tight')
    plt.close()

#Force plot for id 0
idx=0
shap.force_plot(test_explainer.expected_value, # base_value i.e. expected value i.e. mean of predictions
                shap_values[idx,:], # shap_values i.e. matrix of SHAP values 
                X_test.iloc[idx,:], matplotlib=True, show=False) # features i.e. should be the same as shap_values, above
plt.savefig('SHAP Force Plot.png', bbox_inches='tight')
plt.close()

#Code to get average difference in ambulance wait time for each variable
df = X_train.join(pd.DataFrame(train_shap_values.values,
                               columns=['SHAP_' + col for col in X_train.columns]))

outputs = []
for col in X_train.columns:
    temp = df[[col, 'SHAP_'+col]]
    temp = temp.groupby(col, as_index=False).mean()
    temp['column'] = col
    outputs += temp.values.tolist()

output_df = pd.DataFrame(outputs, columns=['Value', 'Average Difference',
                                           'Feature'])[['Feature', 'Value', 'Average Difference']]


all_scenarios = (df.drop_duplicates(subset=[col for col in df.columns if 'SHAP' in col])
                 .sort_values(by=['Admitted', 'IsInjury', 'MentalHealth', 'ClinicalFrailty', 'NextLocation'])
                 .set_index(['Admitted', 'IsInjury', 'MentalHealth', 'ClinicalFrailty', 'NextLocation']))

#with pd.ExcelWriter('Difference in wait times by feature values.xlsx') as writer:
 #   output_df.to_excel(writer, sheet_name='Average SHAP', index=False)
  #  all_scenarios.to_excel(writer, sheet_name='All Scnarios')
