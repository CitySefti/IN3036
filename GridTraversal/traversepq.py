import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import heapq


class TraversePQ:

    def __init__(self, cellMin, cellMax, mapWidth, mapHeight):
        self.cellMin = cellMin
        self.cellMax = cellMax
        self.mapWidth = mapWidth
        self.mapHeight = mapHeight

    def run(self):
        # Create Map
        width = self.mapWidth # Need to do this for width and height as I got errors when I try to use them via self.
        height = self.mapHeight
        map = np.random.randint(self.cellMin, self.cellMax, size=(width, height))

        # initiate all auxiliary arrays and variables
        path = []
        heap = []
        pathCount = 0
        distMap = np.ones((width, height), dtype=int) * np.Infinity
        distMap[0, 0] = 0
        originMap = np.ones((width, height), dtype=int) * np.nan
        visited = np.zeros((width, height), dtype=bool)
        finished = False
        count = 0
        mapSum = 0
        mapTemp = map.astype(float)

        heapq.heappush(heap, (0, (0, 0)))

        # Start timer for finding the shortest path specifically
        start = time.time()

        # Will loop until the end goal is reached and shortest path is found.
        while heap:
            # extract cell with minimum distance
            dist, (xPos, yPos) = heapq.heappop(heap)

            # Checks the adjacent cells and compares to the current one, moves current position accordingly.
            if xPos > 0:
                if distMap[xPos - 1, yPos] > map[xPos - 1, yPos] + distMap[xPos, yPos] and not visited[xPos - 1, yPos]:
                    distMap[xPos - 1, yPos] = map[xPos - 1, yPos] + distMap[xPos, yPos]
                    originMap[xPos - 1, yPos] = np.ravel_multi_index([xPos, yPos], (width, height))
                    heapq.heappush(heap, (distMap[xPos - 1, yPos], (xPos - 1, yPos)))

            if xPos < width - 1:
                if distMap[xPos + 1, yPos] > map[xPos + 1, yPos] + distMap[xPos, yPos] and not visited[xPos + 1, yPos]:
                    distMap[xPos + 1, yPos] = map[xPos + 1, yPos] + distMap[xPos, yPos]
                    originMap[xPos + 1, yPos] = np.ravel_multi_index([xPos, yPos], (width, height))
                    heapq.heappush(heap, (distMap[xPos + 1, yPos], (xPos + 1, yPos)))

            if yPos > 0:
                if distMap[xPos, yPos - 1] > map[xPos, yPos - 1] + distMap[xPos, yPos] and not visited[xPos, yPos - 1]:
                    distMap[xPos, yPos - 1] = map[xPos, yPos - 1] + distMap[xPos, yPos]
                    originMap[xPos, yPos - 1] = np.ravel_multi_index([xPos, yPos], (width, height))
                    heapq.heappush(heap, (distMap[xPos, yPos - 1], (xPos, yPos - 1)))

            if yPos < height - 1:
                if distMap[xPos, yPos + 1] > map[xPos, yPos + 1] + distMap[xPos, yPos] and not visited[xPos, yPos + 1]:
                    distMap[xPos, yPos + 1] = map[xPos, yPos + 1] + distMap[xPos, yPos]
                    originMap[xPos, yPos + 1] = np.ravel_multi_index([xPos, yPos], (width, height))
                    heapq.heappush(heap, (distMap[xPos, yPos + 1], (xPos, yPos + 1)))

            visited[xPos, yPos] = True
            count = count + 1

        # Gets the current time after shortest path is found and calculates elapsed time.
        end = time.time()
        total = end - start

        # Adjust the auxiliaries related to backtracking
        xPos = width - 1
        yPos = height - 1
        mapTemp[int(xPos), int(yPos)] = np.nan

        # Backtrack to plot path
        while xPos > 0.0 or yPos > 0.0:
            pathCount = pathCount + 1
            path.append([int(xPos), int(yPos)])
            xy = np.unravel_index(int(originMap[int(xPos), int(yPos)]), (width, height))
            xPos = xy[0]
            yPos = xy[1]
            mapTemp[int(xPos), int(yPos)] = np.nan

        # Visualize and output
        cMap = mpl.cm.get_cmap("binary")
        cMap.set_bad(color='purple')
        fig, plot = plt.subplots(figsize=(10, 10))
        plot.matshow(mapTemp, cmap=cMap, vmin=0, vmax=30)
        for i in range(height):
            for j in range(width):
                mapSum = mapSum + map[j, i]
                c = map[j, i]
                plot.text(i, j, str(c), va='center', ha='center')

        # Total Dimensions and sum
        print("Map Stats")
        print('Total number of Cells = ' + str(width * height))
        print('Sum of all Cells = ' + str(mapSum))
        # Display sum of path and show figure.
        print(" ")
        print("Shortest Path:")
        print('Path Length = ' + str(distMap[width - 1, height - 1]))
        print('Cells in Path = ' + str(pathCount + 1))
        print('Took ' + str(total) + ' seconds')
        plt.show()