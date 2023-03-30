import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size = 100000

l1 = []
l1n = 0.5
l2 = []
l2n = 0.5
l3 = []
l3n = 0.5

l1m = np.random.randint(-500, 501, 100000) / 5000.
l2m = np.random.randint(-500, 501, 100000) / 5000.
l3m = np.random.randint(-500, 501, 100000) / 5000.

for i in range(size):
    l1n += l1m[i]
    l1n = min(max(l1n, 0), 1)
    l1.append(l1n)
    l2n += l2m[i]
    l2n = min(max(l2n, 0), 1)
    l2.append(l2n)
    l3n += l3m[i]
    l3n = min(max(l3n, 0), 1)
    l3.append(l3n)

l1 = (np.array(l1) / 3) + 0 / 3
l2 = (np.array(l2) / 3) + 1 / 3
l3 = (np.array(l3) / 3) + 2 / 3
print(l1)

c = ["a", "b", "c"]

df = pd.DataFrame(list(zip(l1, l2, l3)), columns=c)

plot_length = size / 10

savepath = "tesplots"
if not os.listdir().__contains__(savepath):
    os.mkdir(savepath)

subpath = "subs"
if not os.listdir(savepath).__contains__(subpath):
    os.mkdir(savepath + "/" + subpath)

plot = df.plot(kind="line", linewidth=0.1, figsize=(10, 10), ylim=(-1, 1),
               use_index=True,
               legend=True, subplots=True)
fgr = plot[0].get_figure()
fgr.savefig(savepath + "/test_separated.pdf")

plot = df.plot(kind="line", linewidth=0.1, figsize=(10, 10), ylim=(-1, 1),
               use_index=True,
               legend=True, subplots=False)
fgr = plot.get_figure()
fgr.savefig(savepath + "/test.pdf")

plt.close('all')

for i in range(10):
    plot = df.plot(kind="line", linewidth=0.1, figsize=(10, 10), ylim=(-1, 1),
                   xlim=(i * plot_length, (i + 1) * plot_length), use_index=True,
                   legend=True, subplots=True)
    fgr = plot[0].get_figure()
    fgr.savefig(savepath + "/" + subpath + "/test_separated_" + str(i) + ".pdf")

    plot = df.plot(kind="line", linewidth=0.1, figsize=(10, 10), ylim=(-1, 1),
                   xlim=(i * plot_length, (i + 1) * plot_length), use_index=True,
                   legend=True, subplots=False)
    fgr = plot.get_figure()
    fgr.savefig(savepath + "/" + subpath + "/test_" + str(i) + ".pdf")

    plt.close('all')

df.to_csv(savepath + "/test.csv")

plt.close('all')

print(plot)
print(df)
