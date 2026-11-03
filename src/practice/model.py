import cv2

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

image = cv2.iamread("image")

image.cvtColor(image, cv2.COLOR_BGR2GRAY)

rezize_image = cv2.resize(image, 200,200)
cv2.imwrite(rezize_image)

data =  load_iris
X = data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
) 

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Predictions:", prediction)
print("Actual:", y_test)
print("Accuracy:", accuracy)