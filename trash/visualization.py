import pickle
import matplotlib.pyplot as plt
import pandas as pd

with open('/home/t-kasanishi/arXiv-bot/abst.pkl', 'rb') as f:
    abst_sim_list = pickle.load(f)
with open('/home/t-kasanishi/arXiv-bot/title.pkl', 'rb') as f:
    title_sim_list = pickle.load(f)
abst_sim_list = [x.item() for x in abst_sim_list]
title_sim_list = [x.item() for x in title_sim_list]

s1 = pd.Series(title_sim_list)
s2 = pd.Series(abst_sim_list)
print(s1.corr(s2, method='spearman'))

plt.scatter(title_sim_list, abst_sim_list)
plt.savefig("/home/t-kasanishi/arXiv-bot/title_abst.png")