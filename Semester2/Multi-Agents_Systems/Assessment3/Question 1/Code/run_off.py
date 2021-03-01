import pandas as pd
import numpy as np

def table(n):
  table = np.arange(1, n+1)
  np.random.shuffle(table)
  return table

n = 12

votes = pd.DataFrame( [table(n), table(n), table(n),
                       table(n), table(n), table(n),
                       table(n), table(n), table(n),
                       table(n), table(n), table(n),
                       table(n), table(n), table(n),
                       table(n), table(n), table(n)], columns = ["participant0", "participant1", "participant2",
                                                                 "participant3", "participant4",
                                                                 "participant5", "participant6", "participant7",
                                                                 "participant8", "participant9",
                                                                 "participant10", "participant11"])

dataframe_table = votes.copy()
length = votes.columns.shape[0]
iterations = np.arange(1, length-1)

for i in iterations:
  min_votes = dataframe_table[dataframe_table == dataframe_table.min(axis = 1).min()].count().idxmax()
  dataframe_table = dataframe_table.drop([min_votes], axis = 1)
  dataframe_table = dataframe_table.rank(axis = 1, method='first') + i

winner = dataframe_table[votes == length].count().idxmax()
print("The Winner is: ", winner)


