import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
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
SELECT NCAttendanceId AS NCAttendanceID, HospitalNumber, PatientAgeOnArrival,
PatientGenderIPM, AmbulanceArrivalDateTime, AmbulanceCallSign, IsNewAttendance,
       MainDiagnosisECDSGroup1, IsInjury, TriageCategory, ClinicalFrailtyScore,
       ActualDischargeDestinationDescription, AdmitPrvspRefno
FROM [DataWarehouse].[ED].[vw_EDAttendance]
WHERE AmbulanceArrivalDate IS NOT NULL
AND AmbulanceArrivalDate > '01-apr-2023'
ORDER BY AmbulanceArrivalDateTime desc
"""

loc_query = """SET NOCOUNT ON
SELECT loc.NCAttendanceID, loc.Bed, loc.NCBedId, loc.startdatetime,
       datediff(minute, loc.startdatetime, loc.enddatetime) as AmbHandoverTime,
       loc.LocationOrder, loc2.LocationDescription as NextLocation,
       loc2.Bed AS NextLocationBed
FROM [DataWarehouse].[ED].[vw_EDAttendanceLocationHistory] loc
LEFT JOIN [DataWarehouse].[ED].[vw_EDAttendanceLocationHistory] loc2
ON loc.NCAttendanceId = loc2.NCAttendanceId
AND loc.EndDateTime = loc2.StartDateTime
WHERE (loc.StartDateTime > '01-apr 2023 00:00:00')
AND (loc.LocationOrder = '1' OR loc.LocationOrder = '2')
--AND (loc.Bed = '999 Waiting Area' OR loc.Bed = 'PRE-REG STROKE')
AND (loc.NCBedId = '161' or loc.NCBedID = '160')
ORDER BY  loc.StartDateTime DESC
"""

occ_query = """SET NOCOUNT ON
SELECT Date, Data
FROM [RealTimeReporting].[PCM].[operational_dashboard_snapshot]
WHERE Measure_Description = 'In Department'
AND (Date > '01-apr 2023 00:00:00')
ORDER BY date desc"""

#Read querys into dataframes
att_df = pd.read_sql(att_query, cl3_engine)
loc_df = pd.read_sql(loc_query, cl3_engine)
occ_df = pd.read_sql(occ_query, realtime_engine).rename(columns={'Data'
                                                                :'EDOccupancy'})
print('Data read from SQL')
#Close the connections
cl3_engine.dispose()
realtime_engine.dispose()

################################################################################
#Some patients are pre-reg stroke, which means they have an extra event before
#arriving in an ambulance where they are booked in ahead due to the severity of
#strokes.  Where this is the first event, we will want to take the event after
#it for ambulance handover time if that event is 999 Waiting Area.

#Filter out anywhere that the second event is pre-reg stroke (ID=160)
filter_loc = loc_df.loc[~((loc_df['NCBedId'] == 160)
                          & (loc_df['LocationOrder'] == 2))]
#Group by incident ID and sum location order and merge back onto original df
sum = (filter_loc.groupby('NCAttendanceID', as_index=False)['LocationOrder']
       .sum().rename(columns={'LocationOrder':'Sum'}))
filter_loc = filter_loc.merge(sum, on='NCAttendanceID')
#We then only want to keep anywhere where bed is 999 waiting area/in ambulance
#(ID = 161) and the sum is 1 (first location) Or 3 (first location was pre-reg
#stroke and second is 999 waiting area/in ambulance).  We have already filtered
#out other bed locations, so anything where sum is 2 had a different first bed
#location
filter_loc = (filter_loc
              .loc[(filter_loc['NCBedId'] == 161)
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
#Add day of the week and month as columns
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
##################################################################################################
#remove outliers
df = df.loc[((df['AmbHandoverTime'] >= 0) & (df['AmbHandoverTime'] <= 2880))
            & ~(df['NextLocation'].isin(['ED Paeds', 'ED Paediatrics']))
            & ~(df['PatientGenderIPM'].isna())].copy()
#Encode required columns
df['PatientGenderIPM'] = df['PatientGenderIPM'].str.extract(r'(\d+)').astype(int)
df['IsNewAttendance'] = np.where(df['IsNewAttendance'] == 'Y', 1, 0)
df['TriageCategory'] = df['TriageCategory'].fillna(0).astype(int)
df['ClinicalFrailty'] = (np.where(df['ClinicalFrailtyScore']
                                  .str.extract(r'(\d+)').astype(float) >= 5,
                                  1, 0))
df['MentalHealth'] = np.where(
                     df['MainDiagnosisECDSGroup1'] == 'Psych / tox / D+A', 1, 0)
df['Admitted'] = np.where(df['AdmitPrvspRefno'] > 0, 1, 0)
df['NextLocation'] = (df['NextLocation']
                      .replace({'ED Ambulatory|ED AMBULATORY' : 1,
                                'Majors HOLD|ED Majors Hold|ED Majors' : 2,
                                'ED Resus' : 3}, regex=True).fillna(0))
#Split into pre and post MRU opening
pre_MRU = df.loc[df['Date'] < '31-07-2024'].copy().reset_index(drop=True).drop('Date', axis=1)
post_MRU = df.loc[df['Date'] >= '31-07-2024'].copy().reset_index(drop=True).drop('Date', axis=1)
################################################################################
results = []
for data, text in [(pre_MRU, 'Pre MRU'), (post_MRU, 'Post MRU')]:
    #Split train/test and fit model
    y = data['AmbHandoverTime']
    X = data[['PatientAgeOnArrival', 'PatientGenderIPM', 'DayOfWeek', 'Month',
            'IsNewAttendance', 'IsInjury', 'TriageCategory', 'NextLocation',
            'ClinicalFrailty', 'MentalHealth', 'Admitted', 'EDOccupancy']]
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
    plt.savefig(text + '/Lasso Feature Selection.png')
    plt.close()

    # Subsetting the top 6 features
    feature_subset = (pd.DataFrame({'Col':X.columns, 'Coef':lasso1_coef})
                      .sort_values(by='Coef', ascending=False)
                      .head(6)['Col'].tolist())
    #feature_subset = np.array(X.columns)[lasso1_coef>15]
    X = X[feature_subset].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                        test_size=0.25)

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
    results.append([train_RMSE, test_RMSE, train_R2, test_R2, scores.mean(),
                    scores.std()])

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
    plt.savefig(text + '/Actual Vs Predicted Ambulance Handover Values.png')
    plt.close()

    # Obtain shap values
    shap_values = shap.Explainer(model).shap_values(X_test)
    # Summary - SHAP value bar chart
    shap.summary_plot(shap_values,
                    X_test,
                    plot_type="bar",
                    show=False)
    plt.savefig(text + '/SHAP interaction values bar.png')
    plt.close()
    # Summary - Beeswarm plot
    shap.summary_plot(shap_values,
                    X_test,
                    show=False)
    plt.savefig(text + '/SHAP interaction values beeswarm.png')
    plt.close()
    # Summary - Violin plot
    shap.summary_plot(shap_values,
                    X_test,
                    plot_type="violin",
                    show=False)
    plt.savefig(text + '/SHAP interaction values violin.png')
    plt.close()

    # Now to create a dependence plot for each
    for e, i in enumerate(X_test.columns):
        shap.dependence_plot(e, shap_values, X_test, show=False)
        plt.savefig(text + f'/{i} dependence plot.png')
        plt.close()

    # compute SHAP values (different to above)
    train_explainer = shap.Explainer(model, X_train)
    test_explainer = shap.Explainer(model, X_test)
    X_explainer = shap.Explainer(model, X)
    train_shap_values = train_explainer(X_train)
    X_shap_values = X_explainer(X)

    #Create some waterfall plots for random ids
    for idx in random.sample(X.index.tolist(), 10):
        fig = plt.figure()
        shap.plots.waterfall(X_shap_values[idx], show=False)
        plt.savefig(text + f'/SHAP waterfall plot id {str(idx)}.png', bbox_inches='tight')
        plt.close()

    #Force plot for id 0
    idx=0
    shap.force_plot(test_explainer.expected_value, # base_value i.e. expected value i.e. mean of predictions
                    shap_values[idx,:], # shap_values i.e. matrix of SHAP values 
                    X_test.iloc[idx,:], matplotlib=True, show=False) # features i.e. should be the same as shap_values, above
    plt.savefig(text + '/SHAP Force Plot.png', bbox_inches='tight')
    plt.close()

    #Code to get average difference in ambulance wait time for each variable
    data = X_train.join(pd.DataFrame(train_shap_values.values,
                            columns=['SHAP_' + col for col in X_train.columns]))

    outputs = []
    for col in X_train.columns:
        temp = data[[col, 'SHAP_'+col]]
        temp = temp.groupby(col, as_index=False).mean()
        temp['column'] = col
        outputs += temp.values.tolist()

    output_df = pd.DataFrame(outputs, columns=['Value', 'Average Difference',
                                            'Feature'])[['Feature', 'Value', 'Average Difference']]

    all_scenarios = (data.drop_duplicates(subset=[col for col in data.columns if 'SHAP' in col])
                    .sort_values(by=list(feature_subset))
                    .set_index(list(feature_subset)))

    with pd.ExcelWriter(text + ' - difference in wait times by feature values.xlsx') as writer:
        output_df.to_excel(writer, sheet_name='Average SHAP', index=False)
        all_scenarios.to_excel(writer, sheet_name='All Scnarios')

#Put together comparison of models
results = pd.DataFrame(data={'Pre MRU':results[0], 'Post MRU':results[1]},
                       index=['Train RMSE', 'Test RMSE', 'Train R2', 'Test R2',
                              'MAE', 'MAE std'],
                       columns=['Pre MRU', 'Post MRU'])
results.to_excel('Comparisons/model results')

#Data cleaning to make comparison plots
#Add column of pre/post MRU
df['Pre/Post MRU'] = np.where(df['Date'] < '31-07-2024', 'Pre MRU', 'Post MRU')
#Add day of week as string
df['DayOfWeek'] = df['Date'].dt.day_name()
DoW = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DoW_cat_type = CategoricalDtype(categories=DoW, ordered=True)
df['DayOfWeek'] = df['DayOfWeek'].astype(DoW_cat_type)
#Add month as a string
df['Month'] = df['Date'].dt.month_name()
MoY = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
       'September', 'October', 'November', 'December']
MoY_cat_type = CategoricalDtype(categories=MoY, ordered=True)
df['Month'] = df['Month'].astype(MoY_cat_type)

#Bar charts of average amb handover by variables pre vs post MRU
for col in ['PatientGenderIPM', 'IsNewAttendance', 'MainDiagnosisECDSGroup1',
            'IsInjury', 'TriageCategory', 'ClinicalFrailtyScore', 'DayOfWeek',
            'Month']:
    ax = (df.groupby(['Pre/Post MRU', col], as_index=False)
          ['AmbHandoverTime'].mean().pivot(columns='Pre/Post MRU',
                                           index=col, values='AmbHandoverTime')
          .plot(kind='bar'))
    fig = ax.get_figure()
    plt.title('Average Amb Handover Time by' + col + 'Pre and Post MRU')
    plt.tight_layout()
    fig.savefig('Comparisons/' + col)
    plt.close()

#Scatter plots of average amb handover by variables pre vs post MRU
colors = {'Pre MRU': 'MediumVioletRed', 'Post MRU': 'Navy'}
for col in ['PatientAgeOnArrival', 'EDOccupancy']:
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, data in  df.groupby('Pre/Post MRU', sort=False):
        print(kind)
        data.plot(kind='scatter', x=col, y='AmbHandoverTime', label=kind, color=colors[kind], ax=ax)
    plt.title('Average Amb Handover Time by' + col + 'Pre and Post MRU')
    plt.tight_layout()
    fig.savefig('Comparisons/' + col)
    plt.close()
