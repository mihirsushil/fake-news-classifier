import random
from unicodedata import digit
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# columns that are going to be used 
cols = ["id", "label", "statement", "true_counts", "mostly_true_counts",
        "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]

df = pd.read_csv("train.csv")
df = df[cols]

# dropping columns that are empty 
df = df.dropna(axis = 1, how = 'all')

# binary classification  
fake_set = {0,1,3}
real_set = {2,4,5}

# making it into a binary lablel 
binary_label = []

for v in df["label"]:
        if v in [2,4,5]:
                binary_label.append(1)
        else:
                binary_label.append(0)

df["binary_label"] = binary_label

# cleaning the data 
X = df['statement'].str.lower().str.strip()
y = df['binary_label']

# spitting into two main sets(training and val,test)
X_train, X_temp, y_train, y_temp = train_test_split(X,y, 
test_size = 0.2, stratify=y,   # stratify -> ensure class balance 
random_state=42 ) 

X_val, X_test, y_val,y_test = train_test_split(X_temp, y_temp,  # using the temp 0.2 of set 
test_size=0.5, stratify = y_temp, 
random_state=42)


# making the pipeline between vecotorizing it and logitical regression  
model = make_pipeline(TfidfVectorizer(ngram_range = (1,2), min_df = 3), 
LogisticRegression(max_iter = 1000, class_weight = "balanced"))

# training it using the train set(0.8)
model.fit(X_train, y_train )
y_val_pred = model.predict(X_val)
print(classification_report(y_val,y_val_pred, target_names=["Fake","Real"], digits = 3 ))

# training it using traing and val set(0.9)
model.fit(pd.concat([X_train,X_val]), pd.concat([y_train,y_val]))
y_test_pred = model.predict(X_test)
print(classification_report(y_test,y_test_pred, target_names=["Fake", "Real"], digits = 3))



