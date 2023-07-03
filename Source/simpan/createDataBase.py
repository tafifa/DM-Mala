import getData as t
import pandas as pd
import os

def createDB(path):
    df = t.getDataFrame(path)

    label = os.path.basename(os.path.dirname(path))

    df.to_csv(f'./csv/KBase_{label}.csv')

def createDBAll():
    # PARASITIZED
    dfO = t.getDataFrame('../Data/cell_images/Parasitized/*')

    # UNINFECTED
    dfM = t.getDataFrame('../Data/cell_images/Uninfected/*')
    # print(dfO, dfM)

    df_concat = pd.concat([dfO, dfM], ignore_index=True)
    df_concat.to_csv('./csv/dummy/KBase.csv')

if __name__ == "__main__":
    createDBAll()
    # createDB('../Data/cell_images/Parasitized/*')
    # createDB('../Data/cell_images/Uninfected/*')