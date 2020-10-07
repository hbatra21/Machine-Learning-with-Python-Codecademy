from typing import List, Any

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create your df here:
df = pd.read_csv("profiles.csv")

# print(df.job.head())

# plt.hist(df.age, bins=20)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.xlim(16, 80)
# plt.show()

# print(df.sign.value_counts())
# print(df.diet.value_counts())

df.fillna({'diet': 'nothing'}, inplace=True)
# find label and data
labels = df['diet']
print(labels.head())

# drink mapping
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}

df["drinks_code"] = df.drinks.map(drink_mapping)

# smokes mapping
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}

df["smokes_code"] = df.smokes.map(smokes_mapping)

# drugs mapping
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}

df["drugs_code"] = df.drugs.map(drugs_mapping)

# print(df.smokes.value_counts())
# print(df.drugs.value_counts())

# combine each essay into one string
essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

df["essay_len"] = all_essays.apply(lambda x: len(x))

df["avg_word_length"] = df['essay_len'].apply(lambda x: np.mean(x))
# normalize data
feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
print(feature_data.head())

feature_data.fillna({'smokes_code': 0, 'drinks_code': 0, 'drugs_code': 0}, inplace=True)
print(feature_data.isna().any())
print(feature_data.corr())

# classification

regressor = KNeighborsRegressor(n_neighbors=5, weights="uniform")

training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data, labels,
                                                                                      test_size=0.2, random_state=10000)
regressor.fit(training_data, training_labels)

accuracies = []
for k in range(1, 201):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    score = classifier.score(validation_data, validation_labels)
    accuracies.append(score)

k_list: list(range(1, 201))

# plot a graph of k
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel("Validation Accuracy")
plt.title("k")
plt.show()
