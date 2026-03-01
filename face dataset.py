import pandas as pd
import numpy as np

def create_face_dataset(file_name, n_samples=20, n_people=5):
    data = []
    for person_id in range(n_people):
        for _ in range(n_samples):
            # Generate dummy pixel data (0-255) with a pattern unique to each person
            pixels = np.random.randint(0, 200, 4096) + (person_id * 10)
            pixels = np.clip(pixels, 0, 255)
            # Add label as the first column
            data.append([person_id] + pixels.tolist())
            
    cols = ['Label'] + [f'Pixel_{i}' for i in range(4096)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(file_name, index=False)
    print(f"File created: {file_name}")

# Create both files
create_face_dataset('train_faces.csv', n_samples=20) # 100 images
create_face_dataset('test_faces.csv', n_samples=5)   # 25 images