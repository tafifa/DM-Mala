from databaseCreator import createDatabase, createDatabase2
from dataProcessor import getData, getglcm, getglcm2, getDataFrame
from distanceComparator import distanceComparison, distanceComparison2

# createDatabase()
dir = '../Data/test/*'
# print(getData(dir))
# createDatabase2()
# print(getData(dir))
# distanceComparison(dir, 'canberra')
distanceComparison2(dir)
# print("Hello")