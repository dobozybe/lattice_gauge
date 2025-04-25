import numpy as np

tuplelist = []
def make_tuples(tuple, depths,outputlist):
        if depths == []:
            outputlist.append(tuple.copy())
            return
        for j in range(depths[0]):
            tuple[-len(depths)] = j
            make_tuples(tuple, depths[1:],outputlist)

make_tuples(np.zeros(2),[4,4],tuplelist)
sum = 0
for tuple in tuplelist:
    print(sum,":",tuple)
    sum+=1

def get_tuple(coordinates, tuple_depths):
    coords = coordinates[::-1]
    depths = tuple_depths[::-1]
    runningsum = 0
    place_value = 1
    for i in range(len(depths[1:])):
        place_value *= depths[i]
        runningsum += coords[i+1] * place_value
    runningsum+=coords[0]
    return tuplelist[runningsum]

print("and the output is")
print(get_tuple([2,3],[4,4]))