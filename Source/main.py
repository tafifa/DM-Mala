from databaseCreator import createDatabase, createDatabase2
from dataProcessor import getData, getglcm, getglcm2, getDataFrame
from distanceComparator import distanceComparison, distanceComparison2

# createDatabase()
dir = '../Data/test/*'
# print(getData(dir))
# createDatabase2()
# print(getData(dir))
distanceComparison(dir, 'canberra')
# distanceComparison2(dir)

'''
6/7/2023 01.32

hasil Tropozoit image
hasil Tropozoit image2
hasil Tropozoit image3
hasil Tropozoit image4
hasil Tropozoit image5
hasil Tropozoit image6

walaupun sudah pakai segmentasi tapi masih menunjukkan hasil klasifikasi yang salah
perlu ditelaah dibagian codingan distance comparison ataupun dari dataset yang dipilih
ada kemungkinan kecil masih ada miss di bagian segmentation

'''