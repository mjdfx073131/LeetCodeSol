from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        ROWS, COLS = len(grid), len(grid[0])
        count = 0
        visited = set()

        def dfs(row, col):
            if (
                row >= ROWS
                or row < 0
                or col >= COLS
                or col < 0
                or grid[row][col] != "1"
                or (row, col) in visited
            ):
                return

            visited.add((row, col))
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)

        for i in range(ROWS):
            for j in range(COLS):
                if grid[i][j] == "1" and (i, j) not in visited:
                    dfs(i, j)
                    count += 1
        return count


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
