from toCSV import createCSV_Infected, createCSV_Stadium, createCSV_result
from secondClassification import getDataFrame, getData
from distanceComparison import getDataFrame

# Parasitized Classification
createCSV_Infected()
dir = '../Data/Fase/test/*'
# getDataFrame(dir)

# print(getData(dir))
createCSV_Stadium()

# Stadium Classification
# print(stadium_classification(dir, 'canberra'))
# print(getDataFrame2(dir, 'canberra'))
createCSV_result(dir, 'canberra')

'''

1. 600 i 600 t, kita count object, tentukan TP i TP t
2. dari TP dan TN, ambil nilai glcm dan nilai bentuk lalu masukkan basis pengetahuan
3. nanti data folder uji dibandingkan dengan basis pengetahuan menggunakan canberra atau svm?
4. tentukan TP FN dari data folder uji

'''