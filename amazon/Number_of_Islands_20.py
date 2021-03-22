from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        result = 0
        checkList = [[False for i in range (len(grid[0]))] for j in range (len(grid))]
        for i in range (len(grid)):
            for j in range(len(grid[i])):
                if i == len(grid) - 1:
                    if j == len(grid[i]) - 1:
                        if (not checkList[i][j] and grid[i][j] == "1"):
                            result += 1
                            # checkList[i][j+1] = True
                    else:
                        if not checkList[i][j] and grid[i][j] == "1":
                            if (i == 0):
                                if grid[i][j-1] == 0 and grid[i+1][j] == 0:
                                    result += 1
                                    checkList[i][j+1] = True
                            if (i != 0 and j == len(grid[i]) - 1):
                                if grid[i-1][j] == 0 and grid[i][j-1] and grid[i][j+1] == 0:
                                    result += 1
                                    checkList[i][j+1] = True
                        elif checkList[i][j] and grid[i][j] == "1":
                            checkList[i][j+1] = True
                        else:
                            checkList[i][j] = 0
                else:
                    if j == len(grid[i]) - 1:
                        if (not checkList[i][j] and grid[i][j] == "1"):
                            if (i == 0 and j == len(grid[i]) - 1):
                                if grid[i][j-1] == 0 and grid[i+1][j] == 0:
                                    result += 1
                                    checkList[i+1][j] = True
                            if (i != 0 and j == len(grid[i]) - 1):
                                if grid[i-1][j] == 0 and grid[i+1][j]  and grid[i][j-1]== 0:
                                    result += 1
                                    checkList[i+1][j] = True
                        elif checkList[i][j] and grid[i][j] == "1":
                            checkList[i+1][j] = True
                        else:
                            checkList[i][j] = 0
                    else:
                        if (not checkList[i][j] and grid[i][j] == "1"):
                            if (i == 0 and j == 0):
                                if grid[i][j+1] == 0 and grid[i+1][j] == 0:
                                    result += 1
                                    checkList[i][j+1] = True
                                    checkList[i+1][j] = True
                            if (i != 0 and j == 0):
                                if grid[i][j+1] == 0 and grid[i+1][j] and grid[i-1][j] == 0:
                                    result += 1
                                    checkList[i][j+1] = True
                                    checkList[i+1][j] = True
                            if (i != 0 and j != 0):
                                if grid[i][j+1] == 0 and grid[i+1][j] and grid[i-1][j] and grid[i][j-1] == 0:
                                    result += 1
                                    checkList[i][j+1] = True
                        elif checkList[i][j] and grid[i][j] == "1":
                            checkList[i][j+1] = True
                            checkList[i+1][j] = True
                        else:
                            checkList[i][j] = 0
        return result


def numIslands(grid):
    result = 0
    checkList = [[False for i in range(len(grid[0]))]
                    for j in range(len(grid))]
    m = len(grid)
    n = len(grid[0])
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if i == len(grid) - 1:
                if j == len(grid[i]) - 1:
                    if ( not checkList[i][j]  and grid[i][j] == "1"):
                        result += 1
                        # checkList[i][j+1] = True
                else:
                    if not checkList[i][j]  and grid[i][j] == "1":
                        result += 1
                        checkList[i][j+1] = True
                    elif checkList[i][j] and grid[i][j] == "1":
                        checkList[i][j+1] = True
                    else:
                        checkList[i][j] = 0
            else:
                if j == len(grid[i]) - 1:
                    if ( not checkList[i][j]  and grid[i][j] == "1"):
                        result += 1
                        checkList[i+1][j] = True
                    elif checkList[i][j] and grid[i][j] == "1":
                        checkList[i+1][j] = True
                    else:
                        checkList[i][j] = 0
                else:
                    if ( not checkList[i][j] and grid[i][j] == "1"):
                        result += 1
                        checkList[i][j+1] = True
                        checkList[i+1][j] = True
                    elif checkList[i][j] and grid[i][j] == "1":
                        checkList[i][j+1] = True
                        checkList[i+1][j] = True
                    else:
                        checkList[i][j] = 0
    return result


print(numIslands(
    [["1", "1", "1"], ["0", "1", "0"], ["1", "1", "1"]]))
