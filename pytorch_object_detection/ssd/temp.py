# def sharp(a,b):
#     res = a-(a&b)
#     return res

# print(sharp(2,3))

a = [[1, 2], [1, 3], [2, 4], [3, 4]]
dict = {}
for ix, (i,j) in enumerate(a):
    # if i == ix+1:
    if i not in dict.keys():
        dict[i]=[j]
    else:
        dict[i].append(j)
    # else:
    # if j == 4:
    #     print(i,j)
    # print(i, j)
print(dict)