import time

starttime = time.time()

for i in range(1000000000):
    j = i+1

print("total:", time.time()-starttime)