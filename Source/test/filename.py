import glob
import pandas as pd
from pathlib import Path

dir = '.../Data/Fase/test/*'

path = glob.glob(dir)
data = []

for item in path:
  filename = Path(item).stem

  data.append(filename)

label = 'filename'

glcm_df = pd.DataFrame(data)

glcm_df.to_csv('../csv/filename.csv')