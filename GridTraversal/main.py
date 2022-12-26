import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, initialPlot = plt.subplots(figsize = (10, 10))
minVal = 0
maxVal = 10
map = np.random.randint(1, maxVal, size = (maxVal, maxVal))

current_cmap = plt.cm.Blues
cmap = mpl.cm.get_cmap("Blues").copy()
cmap.set_bad(color = 'red')
initialPlot.matshow(map, cmap = plt.cm.Blues, vmin = 0, vmax = maxVal * 2)

# Initialize auxiliary arrays

distMap = np.ones((maxVal, maxVal), dtype = int) * np.Infinity
distMap[0, 0] = 0
originMap = np.ones((maxVal, maxVal), dtype = int) * np.nan
visited = np.zeros((maxVal, maxVal), dtype = bool)
finished = False
x, y = int(0), int(0)
count = 0

# Loop Dijkstra until reaching the target cell

while not finished:
    # move to x + 1, y
    if x < maxVal - 1:
        if distMap[x + 1, y] > map[x + 1, y] + distMap[x, y] and not visited[x + 1, y]:
            distMap[x + 1, y] = map[x + 1, y] + distMap[x, y]
            originMap[x + 1, y] = np.ravel_multi_index([x, y], (maxVal, maxVal))
    # move to x - 1, y
    if x > 0:
        if distMap[x - 1, y] > map[x - 1, y] + distMap[x, y] and not visited[x - 1, y]:
            distMap[x - 1, y] = map[x - 1, y] + distMap[x, y]
            originMap[x - 1, y] = np.ravel_multi_index([x, y], (maxVal, maxVal))
    # move to x, y + 1
    if y < maxVal - 1:
        if distMap[x, y + 1] > map[x, y + 1] + distMap[x, y] and not visited[x, y + 1]:
            distMap[x, y + 1] = map[x, y + 1] + distMap[x, y]
            originMap[x, y + 1] = np.ravel_multi_index([x, y], (maxVal, maxVal))
    # move to x, y - 1
    if y > 0:
        if distMap[x, y - 1] > map[x, y - 1] + distMap[x, y] and not visited[x, y - 1]:
            distMap[x, y - 1] = map[x, y - 1] + distMap[x, y]
            originMap[x, y - 1] = np.ravel_multi_index([x, y], (maxVal, maxVal))

    visited[x, y] = True
    distmapTemp = distMap
    distmapTemp[np.where(visited)] = np.Infinity

    # now we find the shortest path so far

    minPos = np.unravel_index(np.argmin(distmapTemp), np.shape(distmapTemp))
    x, y = minPos[0], minPos[1]
    if x == maxVal - 1 and y == maxVal - 1:
        finished = True
    count = count + 1

# Start backtracking to plot the path

matTemp = map.astype(float)
x, y = maxVal - 1, maxVal - 1
path = []
matTemp[int(x), int(y)] = np.nan

while x > 0.0 or y > 0.0:
    path.append([int(x), int(y)])
    xxyy = np.unravel_index(int(originMap[int(x), int(y)]), (maxVal, maxVal))
    x, y = xxyy[0], xxyy[1]
    matTemp[int(x), int(y)] = np.nan
path.append([int(x), int(y)])

# Output and visualization of the path

current_cmap = plt.cm.Blues
cmap = mpl.cm.get_cmap("Blues").copy()
cmap.set_bad(color='red')
fig, newPlot = plt.subplots(figsize=(8, 8))
newPlot.matshow(matTemp, cmap=plt.cm.Blues, vmin=0, vmax=20)
for i in range(maxVal):
    for j in range(maxVal):
        c = map[j, i]
        newPlot.text(i, j, str(c), va='center', ha='center')

print('The path length is: ' + str(distMap[maxVal - 1, maxVal - 1]))

newPlot = plt.imshow(map, cmap)
newPlot = plt.show()