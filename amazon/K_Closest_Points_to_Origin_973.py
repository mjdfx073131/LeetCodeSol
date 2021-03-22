from typing import List

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        dist = lambda i: points[i][0]**2 +points[i][1]**2

        results = []
        for i in range(len(points)):
            points[i] = (points[i][0]**2 + points[i][1]**2,
                         points[i])  # (distance, point) tuple
        self.quickSelectHelper(points, 0, len(points)-1, K)
        for i in range(K):
            results.append(points[i][1])
        return results

    def quickSelectHelper(self, arr, start, end, K):
        if start >= end:
            return
        i, j = start + 1, end
        while i <= j:
            # compare the first element in tuple (distance)
            while i <= j and arr[i][0] <= arr[start][0]:
                i += 1
            # compare the first element in tuple (distance)
            while i <= j and arr[j][0] >= arr[start][0]:
                j -= 1
            if i < j:
                arr[i], arr[j] = arr[j], arr[i]
        arr[start], arr[j] = arr[j], arr[start]
        if K == j:
            return
        elif K < j:
            self.quickSelectHelper(arr, start, j-1, K)
        else:
            self.quickSelectHelper(arr, j+1, end, K)

