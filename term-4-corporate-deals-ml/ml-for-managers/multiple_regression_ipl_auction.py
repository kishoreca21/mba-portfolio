import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

np.set_printoptions(precision=4, linewidth=100)


ipl_auction_df = pd.read_csv( 'D:/MDI/Dr. Sangeeta Shah Bharadwaj/Data Analysis/Dataset and reading material/IPL IMB381IPL2013.csv' )

ipl_auction_df.info()

ipl_auction_df.iloc[0:5, 0:10]
ipl_auction_df.iloc[0:5, 13:]

X_features = ipl_auction_df.columns
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL',
  'CAPTAINCY EXP', 'RUNS-S',  'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS', 'AVE-BL', 'ECON', 'SR-BL']
 
ipl_auction_df['PLAYING ROLE'].unique()

pd.get_dummies(ipl_auction_df['PLAYING ROLE'])[0:5]


categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']

ipl_auction_encoded_df = pd.get_dummies( ipl_auction_df[X_features],  columns = categorical_features, drop_first = True )

ipl_auction_encoded_df.columns

X_features = ipl_auction_encoded_df.columns


X = sm.add_constant( ipl_auction_encoded_df )
Y = ipl_auction_df['SOLD PRICE']

train_X, test_X, train_y, test_y = train_test_split( X , Y, train_size = 0.8, random_state = 42 )

ipl_model_1 = sm.OLS(train_y, train_X).fit()
ipl_model_1.summary2()



# Multi-Collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors( X ):
 X_matrix = X.to_numpy()
 vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
 vif_factors = pd.DataFrame()
 vif_factors['column'] = X.columns
 vif_factors['vif'] = vif
 
 return vif_factors
 
vif_factors = get_vif_factors( X[X_features] )
vif_factors



#Checking correlation of columns with large VIFs
columns_with_large_vif = vif_factors[vif_factors.vif > 4].column
plt.figure( figsize = (20,15) )
sns.heatmap( X[columns_with_large_vif].corr(), annot = True );
plt.title( "Figure 4.5 - Heatmap depicting correlation between features");



columns_to_be_removed = ['T-RUNS', 'T-WKTS', 'RUNS-S', 'HS', 'AVE', 'RUNS-C', 'SR-B', 'AVE-BL', 'ECON', 'ODI-SR-B', 'ODI-RUNS-S', 'AGE_2', 'SR-BL']

X_new_features = list( set(X_features) - set(columns_to_be_removed) )

get_vif_factors( X[X_new_features] )



#Building a new model after removing multicollinearity

train_X = train_X[X_new_features]

ipl_model_2 = sm.OLS(train_y, train_X).fit()
ipl_model_2.summary2()



significant_vars = ['COUNTRY_IND', 'COUNTRY_ENG', 'SIXERS', 'CAPTAINCY EXP_1']
train_X = train_X[significant_vars]
ipl_model_3 = sm.OLS(train_y, train_X).fit()
ipl_model_3.summary2()



#Residual Analysis

## P-P plot
def draw_pp_plot( model, title ):
 probplot = sm.ProbPlot( model.resid, fit=True);
 plt.figure( figsize = (8, 6) );
 probplot.ppplot( line='45' );
 plt.title( title );
 plt.show();
 
draw_pp_plot( ipl_model_3,
 "Figure 4.6 - Normal P-P Plot of Regression Standardized Residuals");
 
#Residual Plot
def get_standardized_values( vals ):
return (vals - vals.mean())/vals.std()

def plot_resid_fitted( fitted, resid, title):
 plt.scatter( get_standardized_values( fitted ),
 get_standardized_values( resid ) )
 plt.title( title )
 plt.xlabel( "Standardized predicted values")
 plt.ylabel( "Standardized residual values") 
 plt.show()
 
plot_resid_fitted( ipl_model_3.fittedvalues, ipl_model_3.resid, "Figure 4.7 - Residual Plot")



#Detecting Influencers

k = train_X.shape[1]
n = train_X.shape[0]

print( "Number of variables:", k, " and number of observations:", n)
leverage_cutoff = 3*((k + 1)/n)
print( "Cutoff for leverage value: ", round(leverage_cutoff, 3) )


from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) )
influence_plot( ipl_model_3, ax = ax )
plt.title( "Figure 4.7 - Leverage Value Vs Residuals")
plt.show()


train_X_new = train_X.drop( [23, 28, 83, 88], axis = 0)
train_y_new = train_y.drop( [23, 28, 83, 88], axis = 0)




#Transforming Response Variable

train_y = np.sqrt( train_y )

ipl_model_4 = sm.OLS(train_y, train_X).fit()
ipl_model_4.summary2()

draw_pp_plot( ipl_model_4, "Figure 4.8 - Normal P-P Plot of Regression Standardized Residuals");




#Making predictions on validation set

pred_y = np.power( ipl_model_4.predict( test_X[train_X.columns] ), 2)

#Measuring RMSE
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(pred_y, test_y))

#Measuring R-squared value
np.round( metrics.r2_score(pred_y, test_y), 2 )

