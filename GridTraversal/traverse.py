import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

minVal = 0
maxVal = 10
map = np.random.randint(minVal, maxVal, size=(maxVal, maxVal))

# Auxiliary arrays and variables
distMap = np.ones((maxVal, maxVal), dtype=int) * np.Infinity
distMap[0, 0] = 0
originMap = np.ones((maxVal, maxVal), dtype=int) * np.nan
visited = np.zeros((maxVal, maxVal), dtype=bool)
finished = False
xPos, yPos = int(0), int(0)
count = 0

# Loop until reaching the target cell
while not finished:
    # Checks the adjacent nodes to the current one and moves to the smallest in the direction of the goal.
    if xPos < maxVal - 1:
        if distMap[xPos + 1, yPos] > map[xPos + 1, yPos] + distMap[xPos, yPos] and not visited[xPos + 1, yPos]:
            distMap[xPos + 1, yPos] = map[xPos + 1, yPos] + distMap[xPos, yPos]
            originMap[xPos + 1, yPos] = np.ravel_multi_index([xPos, yPos], (maxVal, maxVal))

    if xPos > 0:
        if distMap[xPos - 1, yPos] > map[xPos - 1, yPos] + distMap[xPos, yPos] and not visited[xPos - 1, yPos]:
            distMap[xPos - 1, yPos] = map[xPos - 1, yPos] + distMap[xPos, yPos]
            originMap[xPos - 1, yPos] = np.ravel_multi_index([xPos, yPos], (maxVal, maxVal))

    if yPos < maxVal - 1:
        if distMap[xPos, yPos + 1] > map[xPos, yPos + 1] + distMap[xPos, yPos] and not visited[xPos, yPos + 1]:
            distMap[xPos, yPos + 1] = map[xPos, yPos + 1] + distMap[xPos, yPos]
            originMap[xPos, yPos + 1] = np.ravel_multi_index([xPos, yPos], (maxVal, maxVal))

    if yPos > 0:
        if distMap[xPos, yPos - 1] > map[xPos, yPos - 1] + distMap[xPos, yPos] and not visited[xPos, yPos - 1]:
            distMap[xPos, yPos - 1] = map[xPos, yPos - 1] + distMap[xPos, yPos]
            originMap[xPos, yPos - 1] = np.ravel_multi_index([xPos, yPos], (maxVal, maxVal))

    visited[xPos, yPos] = True
    distMapTemp = distMap
    distMapTemp[np.where(visited)] = np.Infinity

    # Find the shortest path
    minPath = np.unravel_index(np.argmin(distMapTemp), np.shape(distMapTemp))
    xPos, yPos = minPath[0], minPath[1]
    if xPos == maxVal - 1 and yPos == maxVal - 1:
        finished = True
    count = count + 1

# Auxiliary for backtracking
mapTemp = map.astype(float)
xPos, yPos = maxVal - 1, maxVal - 1
path = []
mapTemp[int(xPos), int(yPos)] = np.nan

# Backtrack to plot path
while xPos > 0.0 or yPos > 0.0:
    path.append([int(xPos), int(yPos)])
    xxyy = np.unravel_index(int(originMap[int(xPos), int(yPos)]), (maxVal, maxVal))
    xPos, yPos = xxyy[0], xxyy[1]
    mapTemp[int(xPos), int(yPos)] = np.nan
    path.append([int(xPos), int(yPos)])

# Visualize and output
cMap = mpl.cm.get_cmap("binary").copy()  # I get an error without mpl....copy()
cMap.set_bad(color='purple')
fig, plot = plt.subplots(figsize=(10, 10))
plot.matshow(mapTemp, cmap=cMap, vmin=0, vmax=15)
for i in range(maxVal):
    for j in range(maxVal):
        c = map[j, i]
        plot.text(i, j, str(c), va='center', ha='center')

# Display sum of path and show figure.
print('Path Length = ' + str(distMap[maxVal - 1, maxVal - 1]))
plt.show()
