# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:10:34 2019

@author: SSB
"""

from pyspark import SparkContext, SparkConf
import sys
import math
from pyspark.mllib.recommendation import ALS, Rating
import time

conf=SparkConf().setAppName("Model Based CollaborativeFiltering (ALS)")
sc=SparkContext(conf=conf)


trainingFile=sys.argv[1]
testFile=sys.argv[2]
outPutFile=sys.argv[3]


startTime= time.time()

trainingSet=sc.textFile(trainingFile).map(lambda x: x.split(','))
trainingSet= trainingSet.filter(lambda x: x[0]!= "user_id")

testSet=sc.textFile(testFile).map(lambda x: x.split(','))
testSet= testSet.filter(lambda x: x[0]!= "user_id")

rdd_completeData = sc.union([trainingSet,testSet])

list_userId_distinct = rdd_completeData.map(lambda x : x[0]).distinct().collect()

list_businessId_distinct = rdd_completeData.map(lambda x : x[1]).distinct().collect()

numberOfUsers= len(list_userId_distinct)

numberOfBusinesses= len(list_businessId_distinct)


dict_users=dict()
for usernum in range(numberOfUsers):
    dict_users[list_userId_distinct[usernum]]=usernum
    
dict_businesses = dict()
for biznum in range(numberOfBusinesses):
    dict_businesses[list_businessId_distinct[biznum]] = biznum
    

rdd_Rating = trainingSet.map(lambda x: Rating(int(dict_users[x[0]]), int(dict_businesses[x[1]]), float(x[2])))

rank=3
iterations=10
model=ALS.train(rdd_Rating,rank,iterations,0.1)


rdd_test_Rating=testSet.map(lambda x: Rating(int(dict_users[x[0]]), int(dict_businesses[x[1]]), float(x[2])))

rdd_test_userId_businessId=rdd_test_Rating.map(lambda x: (x[0],x[1]))

predictions= model.predictAll(rdd_test_userId_businessId).map(lambda x: ((x[0],x[1]),x[2]))


ratesAndPreds = rdd_test_Rating.map(lambda x: ((x[0],x[1]),x[2])).join(predictions)

mean_squared_error=ratesAndPreds.map(lambda x: ((x[1][0] - x[1][1])*(x[1][0] - x[1][1]))).mean()
print("Root mean_squared_error = "+str(math.sqrt(mean_squared_error)))


list_ratesAndPreds = ratesAndPreds.collect()

w=open(outPutFile,'w')
w.write("user_id, business_id, prediction")
for result in list_ratesAndPreds:
    w.write("\n")
    w.write(list_userId_distinct[result[0][0]]+","+list_businessId_distinct[result[0][1]]+","+str(result[1][1]))

w.close()

endTime= time.time()
print("Duration: "+str(endTime-startTime))


