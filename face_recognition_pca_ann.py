import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- STEP 1: GENERATE DATASET ---
# According to your PDF, we need a face database. 
# We create CSVs with 5 people and 4096 pixels (64x64 images).
def create_dataset():
    print("Creating face database...")
    data = []
    for person_id in range(5):
        for _ in range(25):
            pixels = np.random.randint(0, 200, 4096) + (person_id * 15)
            pixels = np.clip(pixels, 0, 255)
            data.append([person_id] + pixels.tolist())
    
    df = pd.DataFrame(data, columns=['Label'] + [f'Pixel_{i}' for i in range(4096)])
    df.to_csv('train_faces.csv', index=False)
    df.sample(20).to_csv('test_faces.csv', index=False)
    print("Files 'train_faces.csv' and 'test_faces.csv' generated.")

# --- STEP 2: PCA & ANN IMPLEMENTATION ---
def run_recognition():
    # Load and Normalize
    train = pd.read_csv('train_faces.csv')
    test = pd.read_csv('test_faces.csv')
    
    X_train, y_train = train.iloc[:, 1:].values / 255.0, train.iloc[:, 0].values
    X_test, y_test = test.iloc[:, 1:].values / 255.0, test.iloc[:, 0].values

    # PCA (Feature Extraction)
    # Reduces 4096 dimensions to 100 Eigenfaces
    print("Applying PCA to find Eigenfaces...")
    pca = PCA(n_components=100, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # ANN (Classification)
    # Using a 1024-neuron hidden layer as the 'brain'
    print("Training Artificial Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(1024,), max_iter=500, random_state=42)
    mlp.fit(X_train_pca, y_train)

    # Evaluation
    y_pred = mlp.predict(X_test_pca)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Save models for GitHub upload
    joblib.dump(pca, 'pca_model.pkl')
    joblib.dump(mlp, 'ann_model.pkl')

if __name__ == "__main__":
    create_dataset()
    run_recognition()