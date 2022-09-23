### Requires Libraries and Methods
from sklearn.preprocessing import MinMaxScaler
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ProximityForest
from sklearn.model_selection import train_test_split
from sktime.datasets import load_arrow_head
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path
import os, csv, re, math, time
import datetime as dt

#################################################################################################

### 1.) Load a priori clustered data from the clustering script -> It forms the basis to train the classification model
df_hdf = pd.read_hdf("data.h5", "df_data")

#################################################################################################

### 2.) Presets: Required to create a directories for classification and to put data into it
###     "input_data" - directory contains new input files to classify
###     "done_classification" - directory contains the preprocssed data -> Actually not required because data will hold in memory, thus no export of preprocessed data are not required
cwd = os.getcwd()

# Check exisiting working directory, if not create one
if not os.path.exists(cwd + "/" + "working_dir"):
    os.makedirs("working_dir")
    print("Working directory successfully created")
    
    """
    if not os.path.exists(cwd + "/" + "working_dir" + "/" + "done_classification"):
        os.makedirs(cwd + "/" + "working_dir" + "/" + "done_classification")
        print("Done directory successfully created")
    else:
        print("Done directory already exists")
    """
    
    if not os.path.exists(cwd + "/" + "working_dir" + "/" + "input_data"):
        os.makedirs(cwd + "/" + "working_dir" + "/" + "input_data")
    else:
        print("Raw data directory already exists")
else:
    print("Working directory already exists, but check existing done and raw directory!")
    
    """
    if not os.path.exists(cwd + "/" + "working_dir" + "/" + "done_classification"):
        os.makedirs(cwd + "/" + "working_dir" + "/" + "done_classification")
        print("Done directory successfully created")
    else:
        print("Done directory already exists")
    """    
    
    if not os.path.exists(cwd + "/" + "working_dir" + "/" + "input_data"):
        os.makedirs(cwd + "/" + "working_dir" + "/" + "input_data")
        print("Raw data directory successfully created")
    else:
        print("Raw data directory already exists")
		
#################################################################################################

### 3.) Prepare (new) input data, thus some data preprocessing are necessary to standardize data

# Define path variable
# First path variable contains raw data   -  Annotation: Cluster all available data -> move all out data crop folder into raw_data folder; per default only the data from the residuals folder are used
WORK_PATH = Path(cwd + "/" + "working_dir" + "/" + "input_data")
#DONE_PATH = Path(cwd + "/" + "working_dir" + "/" + "done_classification")

# List all files within working directory and its sub-directories
tmp_list = []
tmp_part = []        # Total time series (length in total) = 900 = (12*60 ) + (15*12)
df_list = []         # 900 Data frames -> each data frames represent one year; length of each Data frames (or time series) are 365

filename_only = []

for root, dirs, files in sorted(os.walk(WORK_PATH)):
    for file in files:
        if file.endswith('.txt'):
            tmp_list.append(os.path.join(root, file))
            
# Clear file list, which contains log files like checkpoint in name
regex = re.compile(r'-checkpoint.txt')
file_list = [i for i in tmp_list if not regex.search(i)]

# Process raw data files and convert them to a manageable data format, like csv
# At first check if the list of files is empty, if not then continue else skip because no files exists to work with
if file_list:
    for i, elem in enumerate(file_list):
        raw_files = pd.read_csv(file_list[i], sep="\s+", header=3, usecols=[0,1,6])  #1 Read all files, but only Jahr, Tag and BOF (%nFK)

        #2 Alternativ to slicing data (filtering) -> Remove meta information 
        df_file = raw_files.loc[raw_files["Jahr"] != "Station:", "Jahr":"BOF"]
        df_file = raw_files.loc[raw_files["Jahr"] != "Flexibilisierung:", "Jahr":"BOF"]
        df_file = raw_files.loc[raw_files["Jahr"] != "Hauptfrucht:", "Jahr":"BOF"]
        # Special treatment for Jahr, Tag and BOF
        df_file = raw_files.loc[raw_files["Jahr"].str.contains("Jahr|mm") == False, "Jahr":"BOF"]
        
        #3 After removing superfluous information reset index to align them 
        df_file.dropna(inplace=True)
        df_file.reset_index(drop=True, inplace=True)

        #4 Rename columns caption via index
        df_file = df_file.rename(columns={df_file.columns[2]: "BOF (%nFK)"})

        #5 Check the number of days of a year, if the year contains more than 365 days then remove the day 366 -> harmonise data length
        indexNames = df_file.loc[df_file["Tag"] == "366", "Jahr":"BOF (%nFK)"].index
        df_file = df_file.drop(indexNames)

        #6 dtype conversion, because per default all read data are of type object (or string)
        # Use int32/float32 instead of int64/float64 by default to save memory
        df_file['Jahr'] = df_file['Jahr'].astype('int32')
        df_file['Tag'] = df_file['Tag'].astype('int32')
        # Special treatment for the col "BOF (%nFK)" which objects are comma separation instead dot 
        # => relevant for conversion
        df_file['BOF (%nFK)'] = df_file['BOF (%nFK)'].apply(lambda x: x.replace(',', '.')).astype('float32')

        #7 Combine cols "Jahr" and "Tag" and then convert it to datetime format, ... 
        # ... whereby Day 1, 2019 can be translated to jan 1st 2019
        # a.) Create a new col 'Date'
        df_file['Date'] = df_file['Jahr'] * 1000 + df_file['Tag']
        
        # b.) Convert current date format YYYY-DD to actual date format YYYY-MM-DD
        df_file['Date'] = pd.to_datetime(df_file['Date'], format='%Y%j')
        
        #8 insert column using insert(position, column_name, first_column) function              
        df_file.insert(0, 'Date', df_file.pop('Date'))

        #9 Omit superfluous columns
        df_file = df_file.drop(columns=["Jahr", "Tag"])
        
        #10 Set Date as (time) index
        df_file.set_index('Date', inplace=True)
        df_file.sort_index(inplace=True)
        
        #11 Partition dataframe by years -> Required to treat cluster daily based data for a year as data point (thus clustering whole year)
        for j in df_file.groupby(pd.DatetimeIndex(df_file.index).year):
            tmp_part.append(j)
        
        #12 Filenames without extension .txt
        #base = os.path.basename(file_list[i])
        #tmp = os.path.splitext(base)
        #filename_only.append(tmp[0])
        
        #12 Export processed data -> Not requried, but can be used to check, if the the data are processed correctly, but for now its will be commented
        #export_csv = df_file.to_csv(str(DONE_PATH) + "/" + filename_only[i] + ".csv", sep=";", index=True, header=True, encoding="utf-8")
else:
    print("List is empty, hence no files in directory to process")
    

#12  Get dataframe from the tuple -> extract from partition only the data frames, e.g. [(1961, dataframe)]
for k, df in enumerate(tmp_part):
    df_list.append(df[1])
    

#################################################################################################

### 4.) Convert each time series from pd.dataframe from to pd.series
df_list_series = []

for i, series in enumerate(df_list):
    df_tmp = df_list[i].squeeze()
    df_list_series.append(df_tmp)

	
### 5.) Create label 'year'
year_tmp = []

for i, elem in enumerate(tmp_part):
    year_tmp.append(elem[0])

df_data = pd.DataFrame({"Jahr": year_tmp, "Series": df_list_series})

#################################################################################################

### 6.) Required for classsification based on the clustered data! (not for the new input data) 

X_data = df_hdf["Series"]
X_data = X_data.to_frame()                       # Take from the hdf5 file the first column, which contains the BOF (%nFK) and convert from series to data frame type

y_data = df_hdf["Labels"]
y_data = y_data.to_numpy()                       # Similar for the column with assigned labels, but to numpy array type 

# Split data in train and test samples -> Ratio 70:30 (rule of thumb -> adjustable depending on preference via parameter "test_size")
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30)

#################################################################################################

### 7.) Required for the new input data, which are classified 

X_classify = df_data["Series"]
X_classify = X_classify.to_frame()


#################################################################################################


### 8.) Apply Classifiers

# A.) KNN - Tuning parameter: n_neighbors=n  -> tends to overfitting and sensitve to outliers, not good for large data sets -> not recommended!
knn_classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1 ,distance="euclidean").fit(X_train, y_train)

knn_pred = knn_classifier.predict(X_test)

# Measure accuracy
accuracy_score(y_test, knn_pred)
confusion_matrix(y_test, knn_pred)


# The following classifiers are using the concept of ensemble learning
# That means: 
#    -> Instead one strong learner (decision tree) there are n-weak learners, which are trained on subset of the given data sets
#    -> n-weak learners or decision tress are trained and buil up to a meta model with result
#    -> Pro: Prevent under-/ overfitting and is not subject under random influence, thus  yield better classification result
#    -> Cons: requires higher computational ressources

#################################################################################################

# B.) TimeSeriesForestClassifier - Tuning parameter: n_estimators=n (Number of estimators to build for the ensemble)

#B.1  Define and call classification model and train via fit()-function -> fit() takes the first argument the train data and as second the target variable, which contains the labels that belongs to the train data
forest_classifier = TimeSeriesForestClassifier(n_estimators=5).fit(X_train, y_train)

#B.2 Once the training is complete, we can now see how well the model predicts the "new" data, which in this case is the previously omitted or split data.  
#forest_pred = forest_classifier.predict(X_test)

# Accuracy Score
forest_score = forest_classifier.score(X_test, y_test)

#################################################################################################

### 9.) Classify new input data
forest_pred_input_data = forest_classifier.predict(X_classify)

### 10.) Assign classification result to the input data
df_data.insert(loc=2, column="class", value=forest_pred_input_data)

### 11.) Export result in hdf5 and csv (where the latter is more for checking -> optional)
# Export via hdf will yield a warning, but can be ignored
export_class_hdf = df_data.to_hdf("data_class.hdf5", key="df_data", mode="w" ) 
export_class_csv = df_data.to_csv("data_class.csv", sep=";", index=True, header=True, encoding="utf-8")

#################################################################################################

# Optional: C.) ProximityForest (distanced-based) - Calculation takes longer than the previous two
# Tuning parameter: n_estimators=n (The number of trees in the forest), max_depth=4 (maximum depth of the tree) --> Playground 
prox_classifier = ProximityForest(random_state=22, n_estimators=5, max_depth=4, 
                                  distance_measure="euclidean").fit(X_data, y_data)

# Accuracy Score
prox_score = prox_classifier.score(X_test, y_test)