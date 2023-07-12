import pandas as pd
import numpy as np

def perf_infClass():
  path = '../csv/infectedClassification.csv'
  df = pd.read_csv(path)

  # Filter the dataset for rows where the "result" column is "trophozoite"
  fa_df = df[['filename', 'result', 'label']]

  TP = 0
  TN = 0
  FP = 0
  FN = 0

  for i in range (len(fa_df)):
    if fa_df['result'][i] == 'Parasitized':
      # print(fa_df['filename'][i], 'adalah Parasitized dengan label', fa_df['label'][i])
      if fa_df['label'][i] == 'Parasitized':
        TP += 1
      else:
        FN += 1
    else:
      # print('\t\t\t',fa_df['filename'][i], 'adalah Uninfected dengan label', fa_df['label'][i])
      if fa_df['label'][i] == 'Parasitized':
        FP += 1
      else:
        TN += 1

  print('TP: ', TP, 'TN: ', TN, 'FP: ', FP, 'FN: ', FN)

  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP)
  sensitivity = TP / (TP + FN)
  specificity = TN / (TN + FP)
  fonescore = (2*sensitivity*precision) / (sensitivity+precision)

  print("Accuracy:", accuracy)
  print("Precision: ", precision)
  print("Sensitivity (Recall):", sensitivity)
  print("Specificity:", specificity)
  print("F1-Score: ", fonescore)

def perf_stdClass():
  path = '../csv/result.csv'
  path2 = '../csv/labelling.csv'
  df = pd.read_csv(path)
  df2 = pd.read_csv(path2)

  # Filter the dataset for rows where the "result" column is "trophozoite"
  fa_df = df[['filename', 'prediction']]
  fa_df2 = df2[['filename', 'label']]

  TP = 0
  TN = 0
  FP = 0
  FN = 0

  sama = 0
  tsama = 0
  data = []
  for i in range (len(fa_df)):
    if fa_df['prediction'][i] == fa_df2['label'][i]:
      data.append([fa_df['filename'][i], fa_df2['filename'][i], fa_df['prediction'][i], fa_df2['label'][i], 'sama'])
      # print(fa_df['filename'][i], 'sama')

      sama += 1

    else:
      data.append([fa_df['filename'][i], fa_df2['filename'][i], fa_df['prediction'][i], fa_df2['label'][i], 'tidak sama'])
      # print('\t\t', fa_df['filename'][i], 'tidak sama')
      tsama += 1

  return data, sama, tsama

  

  # print('TP: ', TP, 'TN: ', TN, 'FP: ', FP, 'FN: ', FN)

  # accuracy = (TP + TN) / (TP + TN + FP + FN)
  # sensitivity = TP / (TP + FN)
  # specificity = TN / (TN + FP)

  # print("Accuracy:", accuracy)
  # print("Sensitivity (Recall):", sensitivity)
  # print("Specificity:", specificity)

if __name__ == '__main__':
  print("Klasifikasi Terinfeksi")
  perf_infClass()

  print("\nKlasifikasi Stadium")
  data, sama, tsama = perf_stdClass()

  glcm_df = pd.DataFrame(data)

  print('jumlah sama', sama)
  print('jumlah tsama', tsama)

  print('accuracy = ', (sama)/(sama+tsama))