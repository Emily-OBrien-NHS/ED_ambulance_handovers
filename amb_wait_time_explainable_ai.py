import pandas as pd
import numpy as np
from numpy import absolute
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import xgboost as xgb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sklearn
print("Scikit-Learn Version : {}".format(sklearn.__version__))

import shap
from shap import Explanation
print("SHAP Version : {}".format(shap.__version__))

# JavaScript Important for the interactive charts later on
shap.initjs()

#############################################################################
#Create the engine
cl3_engine = create_engine('mssql+pyodbc://@cl3-data/DataWarehouse?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
#sql query
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

#Read query into dataframe
att_df = pd.read_sql(att_query, cl3_engine)
loc_df = pd.read_sql(loc_query, cl3_engine)
#Close the connection
cl3_engine.dispose()

#Some patients are pre-reg stroke, which means they have an extra event before arriving in an ambulance
#where they are booked in ahead due to the severity of strokes.  Where this is the first event, we will
#want to take the event after it for ambulance handover time if that event is 999 Waiting Area.

#Filter out anywhere that the second event is pre-reg stroke
filter_loc = loc_df.loc[~((loc_df['Bed'] == 'PRE-REG STROKE') & (loc_df['LocationOrder'] == 2))]
#Group by incident ID and sum the location order
sum = filter_loc.groupby('NCAttendanceID', as_index=False)['LocationOrder'].sum().rename(columns={'LocationOrder':'Sum'})
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
############################################################################
#remove outliers
df = df.loc[((df['AmbHandoverTime'] >= 0) & (df['AmbHandoverTime'] <= 2880))
            & ~(df['NextLocation'].isin(['ED Paeds', 'ED Paediatrics']))].copy()
#Encode required columns
df['PatientGenderIPM'] = df['PatientGenderIPM'].str.extract('(\d+)').astype(int)
df['IsNewAttendance'] = np.where(df['IsNewAttendance'] == 'Y', 1, 0)
df['TriageCategory'] = df['TriageCategory'].fillna(0).astype(int)
df['ClinicalFrailty'] = (np.where(df['ClinicalFrailtyScore'].str.extract('(\d+)').astype(float) >= 5,
                                  1, 0))
df['MentalHealth'] = np.where(df['MainDiagnosisECDSGroup1'] == 'Psych / tox / D+A', 1, 0)
df['Admitted'] = np.where(df['AdmitPrvspRefno'] > 0, 1, 0)
df['NextLocation'] = df['NextLocation'].replace({'ED Ambulatory|ED AMBULATORY' : 1,
                                                 'Majors HOLD|ED Majors Hold|ED Majors' : 2,
                                                 'ED Resus' : 3}, regex=True).drop_duplicates()

#Split train/test and fit model
y = df['AmbHandoverTime']
X = df[['PatientAgeOnArrival', 'PatientGenderIPM', #'AmbulanceArrivalDateTime',
        'IsNewAttendance',
        'IsInjury', 'TriageCategory', 'NextLocation', 'ClinicalFrailty', 'MentalHealth',
        'Admitted']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

y_pred = model.predict(X_test)
RMSE = mean_squared_error(y_test, y_pred)
print('RMSE: ' + str(RMSE))

# Check Actual Vs Predictions
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
x_points = y_test
y_points = y_pred
ax.scatter(x_points, y_points)
ax.set_title('Actual Vs Predicted Ambulance Handover Values')
ax.set_ylabel('Predicted Ambulance Handover')
ax.set_xlabel('Actual Ambulance Handover')
ax.plot([0, 1750], [0, 1750], color='r', linestyle='-', linewidth=2)
plt.grid()
plt.show()

# Obtain shap values
shap_values = shap.Explainer(model).shap_values(X_test)
# Obtain shap interaction values
shap_interaction_values = shap.Explainer(model).shap_interaction_values(X_test)
shap.summary_plot(shap_values,
                  X_test,
                  plot_type="bar")
# Summary - Beeswarm plot
shap.summary_plot(shap_values,
                  X_test)
# Summary - Violin plot
shap.summary_plot(shap_values,
                  X_test,
                  plot_type="violin")

# Now to create a dependence plot for each...
# Remember - Y-axis - is SHAP value for respective feature value
# X-axis - is the freature's value
for e, i in enumerate(X_test.columns):
    shap.dependence_plot(e, shap_values, X_test)

# compute SHAP values
# when variable `shap_values` was created above it used slightly different params...
# shap_values = shap.Explainer(model).shap_values(X_test)
explainer2 = shap.Explainer(model, X_train)
shap_values2 = explainer2(X)
# idx of value to check
idx = 0
shap.plots.waterfall(shap_values2[idx])

#Force plot
e = shap.Explainer(model, X_test)
# See how the predicted value above compares to average predicted value below
y_pred = model.predict(X_test)
# Inspecting a single record using `shap.force_plot`
idx = 0
shap.force_plot(e.expected_value, # base_value i.e. expected value i.e. mean of predictions
                shap_values[idx,:], # shap_values i.e. matrix of SHAP values 
                X_test.iloc[idx,:], matplotlib=True, show=True) # features i.e. should be the same as shap_values, above

# Multiple values
# Interactive plot with 2 different drop downs - left and top
shap.force_plot(e.expected_value,
                shap_values,
                X_test, show=True)