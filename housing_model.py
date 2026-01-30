import os
import tarfile
import urllib.request  # corrected import

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Function to download and extract the dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")  # fixed: housing_path was misspelled
    urllib.request.urlretrieve(housing_url, tgz_path)  # fixed: add .request
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Download the dataset
fetch_housing_data()

# Load the CSV file using pandas
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Load and display
housing = load_housing_data()
print(housing.head())
housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()

#Creating a Test set
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indice =shuffled_indices[:test_set_size]
    train_indice =shuffled_indices[test_set_size:]
    return data.iloc[train_indice],data.iloc[test_indice]

train_set,test_set = split_train_test(housing,0.2)
print("length of traineing data = ",len(train_set))
print("length of test data = ",len(test_set))

from zlib import crc32
def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier))& 0xffffffff <test_ratio*2**32

def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]

housing_with_id = housing.reset_index()  #adds an index column
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")

housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6.,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn .model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts()    

for set_ in(strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

housing = strat_train_set.copy()

# Making a scatter plo
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"))
plt.legend()


# In[88]:


# Make sure 'housing' DataFrame is defined and loaded properly
corr_matrix = housing.corr(numeric_only=True)  # Use numeric_only=True for safety
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# In[89]:


from pandas.plotting import scatter_matrix
# Corrected "maedian_house_value" to "median_house_value"
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[90]:


housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)


# In[91]:


#making combinations of attributes
housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_hosehold"]=housing["population"]/housing["households"]


# In[92]:


corr_matrix=housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[93]:


housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels =strat_train_set["median_house_value"].copy()


# In[94]:


#Data cleaning
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
imputer.statistics_
#print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)


# In[95]:


# using sckitlearn ordinalEncoder class:
housing_cat =housing[["ocean_proximity"]]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_


# In[96]:


# categorial to one-hot vectors
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
#converting sparse_matrix to array to save memory
housing_cat_1hot.toarray()
cat_encoder.categories_


# In[47]:


"""Till Here, we have :
1. Upload the dataset from the github respoistory which contains a zip file , from which we extract csv file of housing dataset
2.Then we use some functions(info(),describe()) to know statistically about our data and also make a histogram
3. We made a test set and write some functions to fix our data whenever we refresh our dataset
4.We visualized our data using various types of plots(scattter etc.) on the training dat
5.Then,we will make correlations between diffrent attributes and their combinations.
6.After correlations we perform Data Cleaning(fil the missing values using Simple iMPUTER CLASS)
7. NOW, to fix the categorial class we use ordinal Encode and One Hotencoder classes to convert it to numeric and then in a binary formart
"""  


# In[97]:


#Custom Transformers
from sklearn.base import BaseEstimator , TransformerMixin
rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[::,population_ix] / X[:,households_ix]
        population_per_household = X[:,population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[98]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define the numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])

# Apply the pipeline to the numeric data
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[99]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),("cat",OneHotEncoder(),cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)


# In[101]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# Make sure labels are a 1D array
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions",lin_reg.predict(some_data_prepared))  


# In[102]:


print("Labels:",list(some_labels))


# In[104]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[106]:


from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
housing_ppredictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_mse = np.sqrt(tree_mse)
tree_mse


# In[110]:


tree_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# In[ ]:





# In[112]:


from sklearn.model_selection import cross_val_score
import numpy as np

# Perform cross-validation
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

# Convert scores to RMSE


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Example model
forest_reg = RandomForestRegressor(random_state=42)

# --- 1️⃣ Hyperparameter Tuning with GridSearchCV ---
param_grid = [
    {'n_estimators': [50, 100, 200],
     'max_features': [4, 6, 8],
     'max_depth': [None, 10, 20]}
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print("Best parameters (GridSearchCV):", grid_search.best_params_)

# --- 2️⃣ Hyperparameter Tuning with RandomizedSearchCV ---
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=50, high=200),
    'max_features': randint(low=2, high=8),
    'max_depth': [None, 10, 20, 30]
}

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5,
                                scoring='neg_mean_squared_error',
                                random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

print("Best parameters (RandomizedSearchCV):", rnd_search.best_params_)

# --- 3️⃣ Final Model from Grid Search ---
final_model = grid_search.best_estimator_

# --- 4️⃣ Test on Unseen Data ---
X_test_prepared = full_pipeline.transform(housing_test)  # assuming housing_test exists
y_test = housing_test_labels

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE on test set:", final_rmse)

# --- 5️⃣ Save Model & Pipeline ---
joblib.dump(final_model, "final_model.pkl")
joblib.dump(full_pipeline, "data_pipeline.pkl")

print("Model and pipeline saved successfully!")
