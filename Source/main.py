from toCSV import createCSV_Infected, createCSV_Stadium, createCSV_result

dir = '../Data/Stadium/Test/*'

# Parasitized Classification
createCSV_Infected()

# print(getData(dir))
createCSV_Stadium()

# Stadium Classification
createCSV_result(dir, 'canberra')

### UNTUK HASILNYA CEK DI FOLDER CSV

'''

1. 600 i 600 t, kita count object, tentukan TP i TP t
2. dari TP dan TN, ambil nilai glcm dan nilai bentuk lalu masukkan basis pengetahuan
3. nanti data folder uji dibandingkan dengan basis pengetahuan menggunakan canberra atau svm?
4. tentukan TP FN dari data folder uji

'''