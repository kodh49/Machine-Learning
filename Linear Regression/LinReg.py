import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

xval = [1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012]
y_raw = [338.7, 341.2, 344.4, 347.2, 351.5, 354.2, 356.3, 358.6, 362.4, 366.5, 369.4, 373.2, 377.5, 381.9, 385.6, 389.9, 393.8]
yval = []

for x in range(len(y_raw)):
    yval.append(np.log(y_raw[x]))

data = pd.DataFrame(
    {'income': xval,
    'ulcer_rate': yval
    }
)
pd.set_option('display.max_rows', None)
print(data)

model = sm.OLS.from_formula("ulcer_rate ~ income", data)
result = model.fit()
summ = result.summary()
print(summ)
fig = sm.graphics.abline_plot(model_results=result)
ax = fig.axes[0]
ax.scatter(xval, yval)
plt.xlabel('Year')
plt.ylabel('CO2 Lvl in ppm')
plt.show()

print(len(summ.tables))