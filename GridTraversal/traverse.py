import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

minVal = 0
maxVal = 10
map = np.random.randint(1, maxVal, size=(maxVal, maxVal))

# Initialize auxiliary arrays
distMap = np.ones((maxVal, maxVal), dtype=int) * np.Infinity
distMap[0, 0] = 0
originMap = np.ones((maxVal, maxVal), dtype=int) * np.nan
visited = np.zeros((maxVal, maxVal), dtype=bool)
finished = False
x, y = int(0), int(0)
count = 0

# Loop until reaching the target cell
while not finished:
    # Checks the adjacent nodes to the current one and moves to the smallest in the direction of the goal.
    if x < maxVal - 1:
        if distMap[x + 1, y] > map[x + 1, y] + distMap[x, y] and not visited[x + 1, y]:
            distMap[x + 1, y] = map[x + 1, y] + distMap[x, y]
            originMap[x + 1, y] = np.ravel_multi_index([x, y], (maxVal, maxVal))

    if x > 0:
        if distMap[x - 1, y] > map[x - 1, y] + distMap[x, y] and not visited[x - 1, y]:
            distMap[x - 1, y] = map[x - 1, y] + distMap[x, y]
            originMap[x - 1, y] = np.ravel_multi_index([x, y], (maxVal, maxVal))

    if y < maxVal - 1:
        if distMap[x, y + 1] > map[x, y + 1] + distMap[x, y] and not visited[x, y + 1]:
            distMap[x, y + 1] = map[x, y + 1] + distMap[x, y]
            originMap[x, y + 1] = np.ravel_multi_index([x, y], (maxVal, maxVal))

    if y > 0:
        if distMap[x, y - 1] > map[x, y - 1] + distMap[x, y] and not visited[x, y - 1]:
            distMap[x, y - 1] = map[x, y - 1] + distMap[x, y]
            originMap[x, y - 1] = np.ravel_multi_index([x, y], (maxVal, maxVal))

    visited[x, y] = True
    distMapTemp = distMap
    distMapTemp[np.where(visited)] = np.Infinity

    # now we find the shortest path so far

    minPos = np.unravel_index(np.argmin(distMapTemp), np.shape(distMapTemp))
    x, y = minPos[0], minPos[1]
    if x == maxVal - 1 and y == maxVal - 1:
        finished = True
    count = count + 1

# Auxiliary map
mapTemp = map.astype(float)
x, y = maxVal - 1, maxVal - 1
path = []
mapTemp[int(x), int(y)] = np.nan

# Backtrack to plot path
while x > 0.0 or y > 0.0:
    path.append([int(x), int(y)])
    xxyy = np.unravel_index(int(originMap[int(x), int(y)]), (maxVal, maxVal))
    x, y = xxyy[0], xxyy[1]
    mapTemp[int(x), int(y)] = np.nan
    path.append([int(x), int(y)])

# Visualize and output
cMap = mpl.cm.get_cmap("binary").copy()  # I get an error without mpl....copy()
cMap.set_bad(color='purple')
fig, plot = plt.subplots(figsize=(10, 10))
plot.matshow(mapTemp, cmap=cMap, vmin=0, vmax=15)
for i in range(maxVal):
    for j in range(maxVal):
        c = map[j, i]
        plot.text(i, j, str(c), va='center', ha='center')

print('Path Length = ' + str(distMap[maxVal - 1, maxVal - 1]))
plt.show()
