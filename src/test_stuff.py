import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
# plot histogram with training time
pos = [0, 1, 2]
time = {}
time["cb"] = 3.07
time["xgb"] = 10.89
time["rf"] = 5.17



plt.figure()
df = pd.DataFrame(list(time.items()), columns=["Model", "Time"]).sort_values("Time")
print(df)

plt.bar(df["Model"], df["Time"], width=0.25)
plt.xticks(
    pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
)

plt.show()
"""
