# import sys

# def check(nums):
#     for i in range(1, len(nums)):
#         if nums[i] < nums[i - 1]:
#             return False
#         return True

# for line in sys.stdin:
#     if line[0] is '\n':
#         break
#     num = [int(n) for n in line.split(' ')]
#     ans = False
#     for i in range(len(num)):
#         nums = [num[k] for k in range(len(num)) if k != i]
#         print(nums)
#         ans = ans or check(nums)
#         if ans:
#             print('Yes')
#             break
#     if not ans:
#         print('No')
# hash={}
# inp = input()
# size, nums = inp.split('/')
# size = int(size)

# print(size, nums)

# import sys

# for line in sys.stdin:
#     if line[0] is '\n':
#         break
#     num = [int(n) for n in line.split(' ')]
#     for i in range(len(num)):
#         cnt = 0
#         for j in range(i + 1, len(num)):
#             if num[j] > num[i]:
#                 cnt = cnt + 1
#             if cnt >= 2:
#                 print('No')
#     print('Yes')

