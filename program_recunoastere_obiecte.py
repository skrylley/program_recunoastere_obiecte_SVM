# Tema:     Colocviu
# Grupa:    323AC
# Echipa:   Mateescu Catalin
#           Dobrin Razvan-Andrei


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Setam calea catre folderul de antrenare
train_path = "train"

# Initializam lista pentru imagini si etichetele lor
images = []
labels = []

# Iteram prin toate imaginile din folderul de antrenare
for image_path in os.listdir(train_path):
    # Incarcam imaginea
    image = cv2.imread(os.path.join(train_path, image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))
    image = image.reshape(image.shape[0]*image.shape[1])
    # Verificam daca imaginea este o masina sau un camion
    if "car" in image_path:
        label = 0
    elif "truck" in image_path:
        label = 1
    else:
        continue
    
    # Adaugam imaginea si eticheta la listele respective
    images.append(image)
    labels.append(label)

# Convertim lista de imagini intr-un numpy array
images = np.array(images)

# Folosim algoritmul "problema celor mai mici patrate" pentru a antrena un model de regresie liniar
def train_model(images, labels):
    model = LinearRegression()
    model.fit(images, labels)
    return model

# Utilizare
model = train_model(images, labels)

# Utilizam modelul antrenat pentru a prezice eticheta unei noi imagini
#inp = input('Introduceti un numar conform index poza: ')
#test_image = cv2.imread('test/test_' + inp +'.png', cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread('test/test_2' +'.png', cv2.IMREAD_GRAYSCALE)
# Afisare imagine
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(test_image,cmap="gray")


plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(model)

plt.show()


test_image = cv2.resize(test_image, (300, 300))
test_image = test_image.reshape(test_image.shape[0]*test_image.shape[1])
predicted_label = model.predict([test_image])

# Afisam eticheta prezisa
if predicted_label <= 0.5 :
    print("Imaginea este o masina")
else:
    print("Imaginea este un camion")

#print(predicted_label)