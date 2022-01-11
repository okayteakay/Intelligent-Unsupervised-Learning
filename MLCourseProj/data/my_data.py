import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class my_data_train:
  def __init__(self):
    self.data = []
    self.targets =[]
    self.classes = ['C1', 'C2', 'C3', 'C4', 'C5']

    path_to_data = '/content/drive/MyDrive/Data/SCAN_mydata/my_data/'
    path_to_csv = '/content/drive/MyDrive/Data/SCAN_mydata/classes.csv'
    df = pd.read_csv(path_to_csv)

    for index, row in df.iterrows():
      print(row)
      name = row[1]
      classes = row[2]
      img = Image.open(path_to_data + name)
      self.data.append(img)
      self.targets.append(classes)


  def __getitem__(self, index):
          """
          Args:
              index (int): Index
          Returns:
              dict: {'image': image, 'target': index of target class, 'meta': dict}
          """
          img, target = self.data[index], self.targets[index]
          img_size = (img.shape[0], img.shape[1])
          # img = Image.fromarray(img)
          class_name = self.classes[target]        

          out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
          
          return out

class my_data_val:
  def __init__(self):
    self.data = []
    self.targets =[]
    self.classes = ['C1', 'C2', 'C3', 'C4', 'C5']

    path_to_data = '/content/drive/MyDrive/Data/SCAN_mydata/my_data/'
    path_to_csv = '/content/drive/MyDrive/Data/SCAN_mydata/classes.csv'
    df = pd.read_csv(path_to_csv)

    for index, row in df.iterrows():
      name = row[0]
      classes = row[1]
      img = Image.open(path_to_data + name)
      self.data.append(img)
      self.target.append(classes)
    
    _, _, self.data, self.target = train_test_split(self.data, self.target, test_size = 0.2, random_state = 37)


  def __getitem__(self, index):
          """
          Args:
              index (int): Index
          Returns:
              dict: {'image': image, 'target': index of target class, 'meta': dict}
          """
          img, target = self.data[index], self.targets[index]
          img_size = (img.shape[0], img.shape[1])
          # img = Image.fromarray(img)
          class_name = self.classes[target]        

          out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
          
          return out



