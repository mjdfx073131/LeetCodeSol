class Solution:

    """ 
    SC: O(n)
    TC: O(n)
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        compliment = {}
        for i in range (len(nums)):
            diff = target - nums[i]
            if diff in compliment:
                return [compliment[diff], i]
            else:
                compliment[nums[i]] = i
    