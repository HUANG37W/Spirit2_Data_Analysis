# %%
import pandas as pd 
import altair as alt
import numpy as np
# import catboost as cb

from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from altair_saver import save
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score

alt.data_transformers.enable('json')
# %%
#Import data
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   

#%%
#Split data
# X, y = load_iris(return_X_y=True)
X = dwellings_ml.drop(['before1980', 'yrbuilt','parcel'], axis = 1 )
y = dwellings_ml.filter(regex = 'before1980')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=76)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=76)

p = y_test.head(10).mean()
print(p)
# %%
# Stories
# Sales price
# Square footage
# Garage size
# Bathrooms

chart1 = (alt.Chart(dwellings_denver.query('yrbuilt > 0 and numbdrm > 0'))
    .encode(
        alt.X('numbdrm', title = "Number of bedrooms"),
        alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d")))
    .mark_boxplot()
    .properties(
        width = 400,
        title = {
            "text": "Many beds back then"
        }
   )
    )
    
    
chart1.save('chart1.png')
chart1



# %%
chart2 = (alt.Chart(dwellings_denver.query('yrbuilt > 0 and numbaths > 0'))
    .encode(
        alt.X('numbaths:O', scale = alt.Scale(zero = False), title = "Number of bathrooms"),
        alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d"))
         )
         .mark_boxplot()
    .properties(
        width = 400,
        title = {"text": "More bathrooms built in modern houses"
        }
    )
    
    )
chart2.save('chart2.png')
chart2

#%%
chart3 = (alt.Chart(dwellings_denver.query('yrbuilt > 0 and -@pd.isnull(arcstyle) '))
    .encode(
        alt.X('arcstyle' , title = "Architectural style"),
        alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d")))
    .mark_boxplot()
    .properties(
        width = 400,
        title = {
            "text": "Some architectural style were more popular after 1980"
        }
    )

)
chart3.save('chart3.png')
chart3


#%%

# chart4 = (alt.Chart(dwellings_denver.query('yrbuilt > 0').dropna())
#     .encode(
#         alt.X('arcstyle' , title = "Architectural style"),
#         alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d")))
#     .mark_boxplot()
#     .properties(
#         width = 400,
#         title = {
#             "text": "Some architectural style was used before the 1980's."
#         }
#     )

# )
# chart4.save('chart4.png')
# chart4


#%%

dwellings_denver['before1980'] = np.where(dwellings_denver['yrbuilt'] < 1980, 1 , 0)


dwell_agg_before =(dwellings_denver
    .query('before1980 == 1')
    .groupby('arcstyle')
    .arcstyle.agg(['count'])
    .reset_index()
    .rename({'count':'before1980Count'}, axis = 'columns')

)

dwell_agg_after = (dwellings_denver
    .query('before1980 == 0')
    .groupby('arcstyle')
    .arcstyle.agg(['count'])
    .reset_index()
    .rename({'count':'after1980Count'}, axis = 'columns')
    
    )

dwell_agg_comb = pd.merge(dwell_agg_after,dwell_agg_before)

dwell_agg_comb['afterPercentage'] = dwell_agg_comb['after1980Count']/(dwell_agg_comb['after1980Count'] + dwell_agg_comb['before1980Count'])
dwell_agg_comb['beforePercentage'] = dwell_agg_comb['before1980Count']/(dwell_agg_comb['after1980Count'] + dwell_agg_comb['before1980Count'])
#%%

# chart6 = (alt.Chart(dwell_agg_comb.encode(
#         alt.X('arcstyle' , title = "Architectural style"),
#         alt.Y('after1980Count' , title = "Year Built", axis = alt.Axis(format="d")))
#     .mark_bar()))
# chart6
#%%
# dwell_agg_comb = pd.merge(dwell_agg_after,dwell_agg_before)



#%%
dwell_agg_comb1 = pd.melt(dwell_agg_comb, id_vars = ['arcstyle'],
    value_vars = ['afterPercentage' , 'beforePercentage'],
    var_name = 'Before/After',
    value_name = 'Percentage')
# ARC_SORT =['ONE-STORY', 'ONE AND HALF STORY','TRI-LEVEL WITH BASEMENT','TRI-LEVEL','BI-LEVEL','MIDDLE UNIT', 'END UNIT','THREE-STORY','TWO AND HALF-STORY','TWO-STORY','SPLIT LEVEL']
# (alt.Chart(dwell_agg_comb).encode(
#     alt.X('arcstyle', sort = ARC_SORT),
#     alt.Y('Percentage'),
#     alt.Color('Before/After')
# ).mark_bar())


dwell_agg_comb1

chart4 = alt.Chart(dwell_agg_comb1).mark_bar(color='firebrick').encode(
    alt.Y('Percentage', axis=alt.Axis(title='Proportion')),
    alt.X('arcstyle', axis=alt.Axis(title='Architectural style'))
) .properties(
        width = 600,
        title = {
            "text": "Ratio of various architectural styles of houses built after 1980 versus houses built before 1980"
        }
    )

chart4.save('chart4.png')
chart4


# chart0 = alt.Chart(dwell_agg_comb1).mark_boxplot(color='firebrick').encode(
#     alt.Y('yrbuilt:N', axis=alt.Axis(title='Year Built')),
#     alt.X('nbhd:O', axis=alt.Axis(title='Neighborhood'))
# ) .properties(
#         width = 600,
#         title = {
#             "text": "Popularity of some neighborhoods over others"
#         }
#     )

# chart0.save('chart4.png')
# chart0

#%%

d = dwellings_denver.query('yrbuilt > 0').groupby('nbhd').filter(lambda x : len(x)>200)
d

chart0 = alt.Chart(d).mark_boxplot().encode(
    alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d")),
    alt.X('nbhd:O', axis=alt.Axis(title='Neighborhood'))
) .properties(
        title = {
            "text": "Popularity of some neighborhoods over others"
        }
)

chart0.save('chart0.png')
chart0

#%%

# chart0 = alt.Chart(dwellings_denver.query('yrbuilt > 0').groupby('nbhd')).mark_boxplot(color='firebrick').encode(
#     alt.Y('yrbuilt', scale = alt.Scale(zero = False), title = "Year Built", axis = alt.Axis(format="d")),
#     alt.X('nbhd:O', axis=alt.Axis(title='Neighborhood'))
# ) .properties(
#         width = 600,
#         title = {
#             "text": "Popularity of some neighborhoods over others"
#         }
#     )

# chart0.save('chart0.png')
# chart0



# chart5 = (alt.Chart(dwell_agg_comb).encode(
#     alt.X('arcstyle'),
#     alt.Y('Percentage'),
#     alt.Color('Before/After')
# ).mark_bar())

# chart5

# ARC_SORT =['ONE-STORY', 'ONE AND HALF STORY','TRI-LEVEL WITH BASEMENT','TRI-LEVEL','BI-LEVEL','MIDDLE UNIT', 'END UNIT','THREE-STORY','TWO AND HALF-STORY','TWO-STORY','SPLIT LEVEL']
# (alt.Chart(dwell_agg_comb).encode(
#     alt.X('arcstyle', sort = ARC_SORT),
#     alt.Y('Percentage'),
#     alt.Color('Before/After')
# ).mark_bar())








# %%
#Lets try a tree model
#GQ2
#tree.DecisionTreeClassifier

tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
y_pred_tree = tree_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred_tree))
print("Accuracy:", metrics.accuracy_score(y_test,y_pred_tree))

#%%


feature_dat = pd.DataFrame({
    "values": tree_clf.feature_importances_,
    "features": X_train.columns
})

chart5 = alt.Chart(feature_dat.query('values > .02')).encode(
    alt.X('values'),
    alt.Y('features', sort = "-x")).mark_bar()

chart5.save('chart5.png')
chart5


# %%
print(metrics.accuracy_score(y_test,y_pred_tree))

#%%
print(metrics.classification_report(y_test, y_pred_tree))

# %%
#look at model performance
print(metrics.confusion_matrix(y_test, y_pred_tree))
metrics.plot_confusion_matrix(tree_clf, X_test, y_test)

#####################end##############
# %%
#GradientBoostingClassifier
boost =  GradientBoostingClassifier(random_state=76)
boost.fit(X_train, y_train)
y_pred_boost = boost.predict(X_test)
print(metrics.classification_report(y_test, y_pred_boost))
print("Accuracy:", metrics.accuracy_score(y_test,y_pred_boost))
print(metrics.classification_report(y_test, y_pred_boost))

feature_dat2 = pd.DataFrame({
    "values": boost.feature_importances_,
    "features": X_train.columns
})


chart8 = alt.Chart(feature_dat2.query('values > .02')).encode(
    alt.X('values'),
    alt.Y('features', sort = "-x")).mark_bar()

chart8.save('chart8.png')
chart8

# %%


# %%
metrics.accuracy_score(y_test,y_pred_boost)


#%%
print(metrics.confusion_matrix(y_test, y_pred_boost))
metrics.plot_confusion_matrix(boost, X_test, y_test)

# feature_dat = pd.DataFrame({
#     "values": tree_clf.feature_importances_,
#     "features": X_train.columns
# })

# alt.Chart(feature_dat.query('values > .02')).encode(
#     alt.X('values'),
#     alt.Y('features', sort = "-x")).mark_bar()

# # %%
# #look at model performance
# print(metrics.confusion_matrix(y_test, y_pred))
# metrics.plot_confusion_matrix(tree_clf, X_test, y_test)



# # %%
# print(metrics.classification_report(y_test, y_pred))


# %%
boost.fit(X_train, y_train)
y_pred_boost = boost.predict(X_test)

feature_dat_boost = pd.DataFrame({
    "values": boost.feature_importances_,
    "features": X_train.columns
})


chart7 = alt.Chart(feature_dat_boost.query('values > .02')).encode(
    alt.X('values'),
    alt.Y('features', sort = "-x")).mark_bar()

chart7.save('chart7.png')
chart7

# %%

chartBoost = alt.Chart(feature_dat_boost.query('values > .02')).encode(
     alt.X('values'),
     alt.Y('features', sort = "-x")).mark_bar()



chartBoost.save('chartBoost.png')
chartBoost



print(metrics.confusion_matrix(y_test, y_pred_boost))
metricsBoostFit = metrics.plot_confusion_matrix(boost, X_test, y_test)
# metricsBoostFit.save('metricsBoostFit.png')

# %%


# catboostR = CatBoostRegressor(verbose=False)
# catboostR.fit(X_train,y_train)
# train_pred_cat = catboostR.predict(X_train)
# test_pred_cat = catboostR.predict(X_test)
# metrics.classification_report(y_train,train_pred_cat)
# print(metrics.classification_report(y_test,train_pred_cat))




# %%
#GQ3 YEESSSSS

# catboost_clf = pd.Dataframe({"Feature Importances": catboost_clf.feature_importances_})

catboost_clf = CatBoostClassifier(
    verbose=False)
catboost_clf.fit(X_train,y_train)
train_predictions_cat = catboost_clf.predict(X_train)
test_predictions_cat = catboost_clf.predict(X_test)
metrics.classification_report(y_train,train_predictions_cat)
print(metrics.classification_report(y_test,test_predictions_cat))

feature_dat_cat = pd.DataFrame({
    "values": catboost_clf.feature_importances_,
    "features": X_train.columns
})

# %%

chart6 = alt.Chart(feature_dat_cat.query('values > .02')).encode(
    alt.X('values'),
    alt.Y('features', sort = "-x")).mark_bar()

chart6.save('chart6.png')
chart6
# %%

print(metrics.confusion_matrix(y_test, test_predictions_cat))
metricsBoostFit = metrics.plot_confusion_matrix(catboost_clf, X_test, y_test)


# %%
