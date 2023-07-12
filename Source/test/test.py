import numpy as np

# Crosstabulation matrix
crosstab = np.array([[21, 0, 0], [2, 14, 3], [1, 1, 8]])

# Calculate the observed agreement (Po)
po = np.trace(crosstab) / np.sum(crosstab)

# Calculate the expected agreement by chance (Pe)
row_totals = np.sum(crosstab, axis=1)
col_totals = np.sum(crosstab, axis=0)
pe = np.sum(row_totals * col_totals) / np.sum(crosstab)**2

# Calculate Cohen's kappa value (κ)
kappa = (po - pe) / (1 - pe)

print("Cohen's kappa value:", kappa)
