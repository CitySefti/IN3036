import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

class Traverse:

    def __init__(self, cellMin, cellMax, gridWidth, gridHeight):
        self.cellMin = cellMin
        self.cellMax = cellMax
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

    def run(self):
        width = self.gridWidth
        height = self.gridHeight
        map = np.random.randint(self.cellMin, self.cellMax, size=(width, height))

        # Auxiliary arrays and variables
        path = []
        distMap = np.ones((width, height), dtype=int) * np.Infinity
        distMap[0, 0] = 0
        originMap = np.ones((width, height), dtype=int) * np.nan
        visited = np.zeros((width, height), dtype=bool)
        finished = False
        xPos = int(0)
        yPos = int(0)
        count = 0
        sum = 0
        mapTemp = map.astype(float)

        start = time.time()
        # Loop until reaching the target cell
        while not finished:
            # Checks the adjacent nodes to the current one and moves to the smallest in the direction of the goal.
            if xPos > 0:
                if distMap[xPos - 1, yPos] > map[xPos - 1, yPos] + distMap[xPos, yPos] and not visited[xPos - 1, yPos]:
                    distMap[xPos - 1, yPos] = map[xPos - 1, yPos] + distMap[xPos, yPos]
                    originMap[xPos - 1, yPos] = np.ravel_multi_index([xPos, yPos], (width, height))

            if xPos < width - 1:
                if distMap[xPos + 1, yPos] > map[xPos + 1, yPos] + distMap[xPos, yPos] and not visited[xPos + 1, yPos]:
                    distMap[xPos + 1, yPos] = map[xPos + 1, yPos] + distMap[xPos, yPos]
                    originMap[xPos + 1, yPos] = np.ravel_multi_index([xPos, yPos], (width, height))

            if yPos > 0:
                if distMap[xPos, yPos - 1] > map[xPos, yPos - 1] + distMap[xPos, yPos] and not visited[xPos, yPos - 1]:
                    distMap[xPos, yPos - 1] = map[xPos, yPos - 1] + distMap[xPos, yPos]
                    originMap[xPos, yPos - 1] = np.ravel_multi_index([xPos, yPos], (width, height))

            if yPos < height - 1:
                if distMap[xPos, yPos + 1] > map[xPos, yPos + 1] + distMap[xPos, yPos] and not visited[xPos, yPos + 1]:
                    distMap[xPos, yPos + 1] = map[xPos, yPos + 1] + distMap[xPos, yPos]
                    originMap[xPos, yPos + 1] = np.ravel_multi_index([xPos, yPos], (width, height))

            visited[xPos, yPos] = True
            distMapTemp = distMap
            distMapTemp[np.where(visited)] = np.Infinity

            # Find the shortest path
            minPath = np.unravel_index(np.argmin(distMapTemp), np.shape(distMapTemp))
            xPos = minPath[0]
            yPos = minPath[1]
            if xPos == width - 1 and yPos == height - 1:
                finished = True
            count = count + 1

        end = time.time()
        total = end - start

        # Auxiliary for backtracking
        xPos = width - 1
        yPos = height - 1
        mapTemp[int(xPos), int(yPos)] = np.nan

        # Backtrack to plot path
        while xPos > 0.0 or yPos > 0.0:
            path.append([int(xPos), int(yPos)])
            xxyy = np.unravel_index(int(originMap[int(xPos), int(yPos)]), (width, height))
            xPos, yPos = xxyy[0], xxyy[1]
            mapTemp[int(xPos), int(yPos)] = np.nan
            path.append([int(xPos), int(yPos)])

        # Visualize and output
        cMap = mpl.cm.get_cmap("binary").copy()  # I get an error without mpl....copy()
        cMap.set_bad(color='purple')
        fig, plot = plt.subplots(figsize=(width, height))
        plot.matshow(mapTemp, cmap=cMap, vmin=0, vmax=15)
        for i in range(height):
            for j in range(width):
                c = map[j, i]
                plot.text(i, j, str(c), va='center', ha='center')
                sum = sum + map[j, i]


        # Total Dimensions and sum
        print('Dimensions Total = ' + str(width * height))
        print('Sum of all cells = ' + str(sum))
        # Display sum of path and show figure.
        print('Path Length = ' + str(distMap[width - 1, height - 1]))
        print('Took ' + str(total) + ' seconds')
        plt.show()
