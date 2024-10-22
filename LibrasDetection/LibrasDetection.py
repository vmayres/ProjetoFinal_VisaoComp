#Bibliotecas
import os

# Direcionamento do diretorio do Dataset.
dataset_path = r'C:\Users\victor\OneDrive\Documentos\GitHub\ProjetoFinal_VisaoComp\Dataset'

#Carrega a lista de labels
label_dict = {}
for i, d in enumerate(sorted(os.listdir(dataset_path))):
    label_dict[d] = i

# print(label_dict)
'''
{'acontecer': 0, 'aluno': 1, 'amarelo': 2, 'america': 3, 'aproveitar': 4, 'bala': 5, 'banco': 6, 'banheiro': 7, 'barulho': 8, 'cinco': 9, 'conhcer': 10, 'espelho': 11, 'esquina': 12, 'filho': 13, 'maca': 14, 'medo': 15, 'ruim': 16, 'sapo': 17, 'vacina': 18, 'vontade': 19}
'''

#Calcula a quantidade de classes
num_classes = len(label_dict)

data = []
labels = []

print("[INFO] loading images...")
# loop over the input images
for imagePath in tqdm(imagePaths):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    #Converts Image instance to a Numpy array
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = label_dict[imagePath.split(os.path.sep)[-2]]
    labels.append(label)