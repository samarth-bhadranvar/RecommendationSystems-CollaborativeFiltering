# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:06:14 2019

@author: SSB
"""

from pyspark import SparkContext, SparkConf
import sys
import math


conf=SparkConf().setAppName("Item Based CollaborativeFiltering")
sc=SparkContext(conf=conf)


trainingFile=sys.argv[1]
testFile=sys.argv[2]
outPutFile=sys.argv[3]

cosineSimilarity_matrix= {}

def normalize(x):
    business_id=x[0]
    ratings_list=x[1]
    print("ratings_list=")
    print(ratings_list)
    
    ratingsTotal=0.0
    ratingList=[]
    ratings_list_size=len(ratings_list)
    for rating in ratings_list:
        ratingsTotal+=float(rating)
        ratingList.append(float(rating))
        
    rating_avg=float(ratingsTotal/ratings_list_size)
    
    return (business_id,(ratingList,rating_avg))
        

def convertFormat(x):
    businessId=x[0][0]
    userId_list=x[0][1]
    ratings_list=x[1][0]
    avgRating=x[1][1]
    
    x_formatted_list=[]
    for i in range(0,len(userId_list)):
        x_formatted_list.append(((userId_list[i],businessId),(ratings_list[i],avgRating)))
    return x_formatted_list


def get_cosine_similarities(x, list_business_usersList_ratingsList_ratingAvg, list_userIdi_bussinessId_ratingi_avgRating):
    testBusinessId=x[0]
    businessRatedByTestUser=x[1]
    
    if(testBusinessId in list_business_usersList_ratingsList_ratingAvg and businessRatedByTestUser in list_business_usersList_ratingsList_ratingAvg):
        userList_forTestBusinessId = list_business_usersList_ratingsList_ratingAvg[testBusinessId][0]
        
        userList_forBRBTU = list_business_usersList_ratingsList_ratingAvg[businessRatedByTestUser][0]
        
        corated_users_set= set(userList_forTestBusinessId) & set(userList_forBRBTU)
        
        N=4000
        
        neighborhood_count=0
        num=0
        den1=0
        den2=0
        cosineSimilarity=0
        
        for corated_user in corated_users_set:
            if neighborhood_count < N:
                ratingForTestBusinessId=float(list_userIdi_bussinessId_ratingi_avgRating[(corated_user,testBusinessId)][0])
                ratingForBRBTU=float(list_userIdi_bussinessId_ratingi_avgRating[(corated_user,businessRatedByTestUser)][0])
                
                num=num+ratingForTestBusinessId*ratingForBRBTU
                den1=den1+ratingForTestBusinessId*ratingForTestBusinessId
                den2=den2+ratingForBRBTU*ratingForBRBTU
                neighborhood_count=neighborhood_count+1
            else:
                break
            
        if num != 0:
            cosineSimilarity=float(num/math.sqrt(den1*den2))
            
        global cosineSimilarity_matrix
        cosineSimilarity_matrix[x]=cosineSimilarity
        
        return cosineSimilarity
    else:
        return 0
        
    
        
    
def getPredictedRatings(x,list_user_businessList,list_business_usersList_ratingsList_ratingAvg,list_userIdi_bussinessId_ratingi_avgRating):
    
    testUserId=x[0]
    testBusinessId=x[1]
    
    
    
    list_businessesRatedByTestUser=list_user_businessList[testUserId]
    
    total_cosineSimilarities=[]
    cosineSimilarities = []
    neighborhood_count=0

    for businessRatedByTestUser in list_businessesRatedByTestUser:
        cosine_similarity=0
        if businessRatedByTestUser != testBusinessId:
            if (testBusinessId,businessRatedByTestUser) in cosineSimilarity_matrix:
                cosine_similarity=cosineSimilarity_matrix[(testBusinessId,businessRatedByTestUser)]
            elif (businessRatedByTestUser,testBusinessId) in cosineSimilarity_matrix:
                cosine_similarity=cosineSimilarity_matrix[(businessRatedByTestUser,testBusinessId)]
            elif testBusinessId in list_business_usersList_ratingsList_ratingAvg and businessRatedByTestUser in list_business_usersList_ratingsList_ratingAvg:
                cosine_similarity=get_cosine_similarities((testBusinessId,businessRatedByTestUser), list_business_usersList_ratingsList_ratingAvg, list_userIdi_bussinessId_ratingi_avgRating)
            else:
                cosine_similarity=0
                
            
            total_cosineSimilarities.append((businessRatedByTestUser,cosine_similarity))
            if cosine_similarity > 1:
                cosineSimilarities.append((businessRatedByTestUser,cosine_similarity))
                neighborhood_count=neighborhood_count+1
                
            
            if neighborhood_count >= N:
                break
    
    
    if len(total_cosineSimilarities) < N:
        cosineSimilarities = total_cosineSimilarities
    
    num=0
    den=0
    
    for cosSimData in cosineSimilarities:
        busnId=cosSimData[0]
        cosSim=cosSimData[1]
        if (testUserId,busnId) in list_userIdi_bussinessId_ratingi_avgRating:
            #avg_rating= float(list_userIdi_bussinessId_ratingi_avgRating[(testUserId,busnId)][1])
            rating=float(list_userIdi_bussinessId_ratingi_avgRating[(testUserId,busnId)][0])
            
            num=num+float(rating*cosSim)
            den=den+abs(cosSim)
            
    if den == 0:
        return ((testUserId,testBusinessId),0)
    else:
        predRating=num/den
        return ((testUserId,testBusinessId),predRating)
    
    
   


trainingSet=sc.textFile(trainingFile).map(lambda x: x.split(','))
trainingSet= trainingSet.filter(lambda x: x[0]!= "user_id")

testSet=sc.textFile(testFile).map(lambda x: x.split(','))
testSet= testSet.filter(lambda x: x[0]!= "user_id")

rdd_user_businessList = trainingSet.map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y)

list_user_businessList = rdd_user_businessList.collectAsMap()

rdd_business_usersList = trainingSet.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y)

rdd_business_ratingsList = trainingSet.map(lambda x: (x[1],[x[2]])).reduceByKey(lambda x,y: x+y)

rdd_business_ratingsList_avgRating = rdd_business_ratingsList.map(lambda x: normalize(x))


rdd_business_usersList_join_ratingList_ratingAvg = rdd_business_usersList.join(rdd_business_ratingsList_avgRating).map(lambda x: ((x[0],x[1][0]),(x[1][1][0],x[1][1][1])))           


list_userIdi_bussinessId_ratingi_avgRating = rdd_business_usersList_join_ratingList_ratingAvg.flatMap(lambda x: convertFormat(x)).collectAsMap()


list_business_usersList_ratingsList_ratingAvg = rdd_business_usersList_join_ratingList_ratingAvg.map(lambda x : (x[0][0], (x[0][1], x[1][0], x[1][1]))).collectAsMap()


rdd_test_user_business = testSet.map(lambda x: (x[0],x[1]))

N=4000

predictedRatings=rdd_test_user_business.map(lambda x: getPredictedRatings(x,list_user_businessList,list_business_usersList_ratingsList_ratingAvg,list_userIdi_bussinessId_ratingi_avgRating))




rdd_testData_predictedData_joined=testSet.map(lambda x: ((x[0],x[1]),float(x[2]))).join(predictedRatings)
mean_squared_error= rdd_testData_predictedData_joined.map(lambda x: ((x[1][0]-x[1][1])*(x[1][0]-x[1][1]))).mean()
print("Root mean_squared_error = "+str(math.sqrt(mean_squared_error)))

w=open(outPutFile,'w')

w.write("user_id,business_id,prediction")
for predictedRating in predictedRatings.collect():
    w.write("\n")
    w.write(predictedRating[0][0]+","+predictedRating[0][1]+","+str(predictedRating[1]))
    
w.close()
    
