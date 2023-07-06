import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

def crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face
    else:
        return None

max_size = 1280
path = "C:/Users/matbl/OneDrive/Obrazy/do_uczenia"
X = []
Y = []
for emotion in os.listdir(path):
    for file in os.listdir(path+"/"+emotion):
        image = cv2.imread(path+"/"+emotion+"/"+file)
        cropped_faces = crop_faces(image)
        if cropped_faces is not None:
            X.append(cropped_faces)
            Y.append(emotion)
            print("działa")
for i in range(len(X)):
    X[i] = resize(X[i], (int(np.sqrt(max_size)), int(np.sqrt(max_size))), anti_aliasing=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.vstack([face.reshape(-1) for face in X_train]))
X_test_scaled = scaler.transform(np.vstack([face.reshape(-1) for face in X_test]))

pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clf = SVC()
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność predykcji: {:.2f}".format(accuracy))

test_image = cv2.imread("C:/Users/matbl/OneDrive/Obrazy/test/test3.jpg")
cropped_faces = crop_faces(test_image)
cropped_faces= resize(cropped_faces, (int(np.sqrt(max_size)), int(np.sqrt(max_size))), anti_aliasing=True)
img_pca=pca.transform(scaler.transform(cropped_faces.flatten().reshape(1, -1)))
#cv2.imshow("Obraz", img_pca)
#cv2.waitKey(0)
predicted_emotions = clf.predict(img_pca)
print("Predykcje dla nowego obrazu: {}".format(predicted_emotions))
