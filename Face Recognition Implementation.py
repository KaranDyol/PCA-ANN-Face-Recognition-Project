import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the files we just created
train = pd.read_csv('train_faces.csv')
test = pd.read_csv('test_faces.csv')

# 2. Normalize and Split data
X_train, y_train = train.iloc[:, 1:].values / 255.0, train.iloc[:, 0].values
X_test, y_test = test.iloc[:, 1:].values / 255.0, test.iloc[:, 0].values

# 3. PCA (Dimensionality Reduction)
# This extracts the "Eigenfaces" from the pixels
print("Extracting 100 Eigenfaces using PCA...")
pca = PCA(n_components=100, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. ANN (Artificial Neural Network)
# A network of 512 neurons to classify the PCA features
print("Training the ANN 'Brain' for recognition...")
clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=500, activation='relu', random_state=42)
clf.fit(X_train_pca, y_train)

# 5. Result and Evaluation
y_pred = clf.predict(X_test_pca)
print("\nFinal Internship Project Summary:")
print(f"Face Recognition Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))