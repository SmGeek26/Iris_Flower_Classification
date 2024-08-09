import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("F:/Machine Learning/Iris_Classification/IRIS.csv")

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with scaling and SVM
model = make_pipeline(StandardScaler(), SVC(probability=True))

# Train model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')