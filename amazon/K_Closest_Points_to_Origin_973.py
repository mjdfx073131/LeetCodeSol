from typing import List
import math
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        euclidean_distance = {}
        result = []
        for point in points:
            dis = (point[0] - 0)**2 + (point[1] - 0)**2
            if(dis not in euclidean_distance):
                euclidean_distance[dis] = []
            euclidean_distance[dis].append(point)
        # print(euclidean_distance)
        i = 0
        while True:
            for point in euclidean_distance[sorted(euclidean_distance.keys())[i]]:
                if len(result) < k:
                    result.append(point)
                    if (len(result) == k):
                        return result
                else:
                    return result
            i += 1

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        result = []
        maxDis = 0
        for point in points:
            dis = (point[0] - 0)**2 + (point[1] - 0)**2
            if len(result) < k:
                if dis > maxDis:
                    maxDis = dis
                    result.append(point)
                else:
                    result.insert(0, point)
            else:
                if dis < maxDis:
                    result.insert(0, point)
                    result.pop()
        return result




