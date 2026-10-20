import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

np.set_printoptions(precision=4, linewidth=100)

credit_df = pd.read_csv( 'D:/MDI/Dr. Sangeeta Shah Bharadwaj/Data Analysis/Dataset and reading material/German Credit Data.csv' )

credit_df.info()

credit_df.iloc[0:5, 1:7]
credit_df.iloc[0:5, 7:]

credit_df['Status'] = np.where(credit_df['Status']==2, 1, 0)

credit_df.Status.value_counts()

X_features = list(credit_df.columns)
X_features.remove('Status')
X_features

#Encoding categorical features
encoded_credit_df=pd.get_dummies(credit_df[X_features],drop_first=True)
list(encoded_credit_df.columns)
##displaying one dummy feature
encoded_credit_df[['Status of existing checking account_A12', 'Status of existing checking account_A13', 'Status of existing checking account_A14']].head(5)


#Defining features and target variable
Y=credit_df.Status
X=sm.add_constant(encoded_credit_df)


# Splitting dataset in train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , Y, test_size = 0.3, random_state = 42 )


#Building logistic regression model
logit_model1=sm.Logit(y_train,X_train).fit()


#Displaying model summary
logit_model1.summary2()


#Model diagnostics#
#Finding significant variables from model summary
def get_sig_vars(lm):
    ##store p-values and corresponding column names in dataframe
    var_pval=pd.DataFrame(lm.pvalues)
    var_pval['vars']=var_pval.index
    var_pval.columns=['pvals','vars']
    ##filter columns where p-values less than 0.05
    return list(var_pval[var_pval.pvals<=0.05]['vars'])
    
sig_vars=get_sig_vars(logit_model1)
sig_vars


logit_model2=sm.Logit(y_train,sm.add_constant(X_train[sig_vars])).fit()
logit_model2.summary2()


y_pred=pd.DataFrame{"actual":y_test, "predited_prob":logit_model2.predict(sm.add_constant(X_train[sig_vars]))})
y_pred.sample(10,random_state=42)


y_pred['predicted']=y_pred.predited_prob.map(lambda x: 1 if x>0.5 else 0)
y_pred.sample(10,random_state=42)



#Creating confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import metrics
def draw_cm(actual,predicted):
cm=metrics.confusion_matrix(actual,predicted,[1,0])
sns.heatmap(cm,annot=True,fmt='.2f',
xticklabels=["Bad credit"."Good credit"],
yticklabels=["Bad credit"."Good credit"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


draw_cm(y_pred.actual,y_pred.predicted)


#Measuring accuracies

print(metrics.classification_report(y_pred.actual,y_pred.predicted))

plt.figure(figsize=(20,15))

#Plotting distribution of predicted probability values for bad credit
sns.distplot(y_pred[y_pred.actual==1]["predited_prob"],kde=False,color='b',label='Bad Credit')

#Plotting distribution of predicted probability values for good credit
sns.distplot(y_pred[y_pred.actual==0]["predited_prob"],kde=False,color='g',label='Good Credit')



#ROC curve

def draw_roc(actual,probs):
    #Obtain fpr,tpr, thresholds
    fpr, tpr, thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(20,15))
    #Plot fpr, tpr values for different thresold values
    plt.plot(fpr,tpr,label='ROC Curve(area=%0.2f)'%auc_score)
    #Drawing 45 degree line
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    #Setting x and y label
    plt.xlabel('False Positive Rate i.e. 1-True negative rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    return fpr,tpr,thresholds

fpr,tpr,thresholds=draw_roc(y_pred.actual,y_pred.predited_prob)


#AUC score
auc_score=metrics.roc_auc_score(y_pred.actual,y_pred.predited_prob)
round(float(auc_score),2)



#Finding optimal classification cut-off

## Youden's index
tpr_fpr=pd.DataFrame({'tpr':tpr,'fpr':fpr,'thresholds':thresholds})
tpr_fpr['difference']=tpr_fpr.tpr-tpr_fpr.fpr
tpr_fpr.sort_values('difference',ascending=False)[0:5]

y_pred['pred_new']=y_pred.predited_prob.map(lambda x: 1 if x>0.27 else 0)

draw_cm(y_pred.actual,y_pred.pred_new)

print(metrics.classification_report(y_pred.actual,y_pred.pred_new))