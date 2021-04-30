from typing import List

class Solution:
    def twoSum1(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                if nums[i]+ nums[j] == target:
                    return i,j

    def twoSum2(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for index , num in enumerate(nums):
            hashmap[num] = index
        for i , num in enumerate(nums):
            j = hashmap.get(target - num)
            if i!=j and j is not None:
                return i,j

if __name__ == '__main__':
    s = Solution()
    res = s.twoSum2([1,2,3,4,5], 7)
    print(res)

