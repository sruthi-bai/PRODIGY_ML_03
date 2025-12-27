import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog


cat_folder = r"E:\cat"
dog_folder = r"E:\dog"

data = []
labels = []


def extract_features(image):
    if image is None or image.size == 0:
        return None

    
    image = cv2.resize(image, (128, 128))  

   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), visualize=False)

    
    hist_r = cv2.calcHist([image], [2], None, [16], [0,256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [16], [0,256]).flatten()
    hist_b = cv2.calcHist([image], [0], None, [16], [0,256]).flatten()

   
    features = np.concatenate([hog_features, hist_r, hist_g, hist_b])
    return features


def load_images_from_folder(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping invalid image: {img_path}")
            continue
        feat = extract_features(image)
        if feat is not None:
            data.append(feat)
            labels.append(label)


load_images_from_folder(cat_folder, 0)  
load_images_from_folder(dog_folder, 1)  


X = np.array(data)
y = np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))


def predict_image(img_path):
    image = cv2.imread(img_path)
    if image is None or image.size == 0:
        print(f"Error: Could not read image at path:\n{img_path}")
        return
    features = extract_features(image)
    if features is None:
        print("Error: Could not extract features.")
        return
    features = features.reshape(1, -1)
    pred = model.predict(features)[0]
    label="CAT " if pred == 0 else "DOG "
    cv2.putText(image, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    img_name = os.path.basename(img_path)
    output_folder = r"E:/predictions"
    os.makedirs(output_folder, exist_ok=True)  
    cv2.imwrite(os.path.join(output_folder, img_name), image)



test_image_path = r"E:\testimage"
for img in os.listdir(test_image_path):
    print("File Name:",img)
    img_path=os.path.join(test_image_path,img)
    predict_image(img_path)
