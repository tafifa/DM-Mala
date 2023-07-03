# import python file from directory
import createDataBase as cDB
import getData as t
import distanceComparison as d

### analyze distance comparison for image from path
# cDB.createDBAll()

pathTest = '../Data/cell_images/dummy/test/*'
metric = 'dice'

positive, negative = d.distanceComparison(pathTest, metric)

# print("\nHasil Prediksi adalah", "Positif" if positive > negative else "Negatif")
