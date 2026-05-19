import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))
df[['Act1','Act2','Act3','Act4','Act5','Act6']] = df.Activities.str.split(',',expand=True)
df[['App1','App2','App3','App4','App5','App6','App7','App8','App9','App10','App11','App12','App13','App14','App15']] = df.Apps.str.split(',',expand=True)

# Convert act/app columns to numeric 1/0 (fix: use int instead of string "0")
for col in ['Act1','Act2','Act3','Act4','Act5','Act6',
            'App1','App2','App3','App4','App5','App6','App7','App8',
            'App9','App10','App11','App12','App13','App14','App15']:
    df[col] = df[col].notna().astype(int)

df = df.drop(columns=["Activities", "Apps"])

ovrScore = []

for idx in df.index:
    score = 0
    if df["Age"][idx] == 0:
        score += 10
    if df["Act1"][idx] == 1:
        score += 10
    if df["Act2"][idx] == 1:
        score += 10
    if df["Act3"][idx] == 1:
        score += 10
    if df["Act4"][idx] == 1:
        score += 10
    if df["Act5"][idx] == 1:
        score += 10
    if df["Act6"][idx] == 1:
        score += 10
    if df["App1"][idx] == 1:
        score += 10
    if df["App2"][idx] == 1:
        score += 10
    if df["App3"][idx] == 1:
        score += 10
    if df["App4"][idx] == 1:
        score += 10
    if df["App5"][idx] == 1:
        score += 10
    if df["App6"][idx] == 1:
        score += 10
    if df["App7"][idx] == 1:
        score += 10
    if df["App8"][idx] == 1:
        score += 10
    if df["App9"][idx] == 1:
        score += 10
    if df["App10"][idx] == 1:
        score += 10
    if df["App11"][idx] == 1:
        score += 10
    if df["App12"][idx] == 1:
        score += 10
    if df["App13"][idx] == 1:
        score += 10
    if df["App14"][idx] == 1:
        score += 10
    if df["App15"][idx] == 1:
        score += 10
    if df["Screen Time"][idx] == 1:
        score += 10
    if df["Screen Time"][idx] == 2:
        score += 20
    if df["Screen Time"][idx] == 3:
        score += 30
    if df["Screen Time"][idx] == 4:
        score += 40
    if df["Technology affect"][idx] == 1:
        score += 5
    if df["Technology affect"][idx] == 0:
        score -= 5
    if df["Improve Addiction"][idx] == 0:
        score += 5
    if df["Improve Addiction"][idx] == 1:
        score -= 5

    ovrScore.append(score)


df["Target"] = 0
df["Overall_score"] = ovrScore
for idx in df.index:
    if ovrScore[idx] >= 105:
        df["Target"][idx] = 1
    if ovrScore[idx] < 105:
        df["Target"][idx] = 0

df = df.drop(columns=["Overall_score"])

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :25], df.iloc[:, 25], random_state=42, test_size=0.20, shuffle=True
)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
print(f"Model accuracy on test set: {score*100:.1f}%")

# Save fresh model (overwrite)
pickle.dump(knn, open("knn.pkl", "wb"))
print("Model saved to knn.pkl")