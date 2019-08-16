# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:08:57 2019

@author: SSB
"""

from pyspark import SparkContext, SparkConf
import sys
import math


conf=SparkConf().setAppName("User Based CollaborativeFiltering")
sc=SparkContext(conf=conf)


trainingFile=sys.argv[1]
testFile=sys.argv[2]
outPutFile=sys.argv[3]

def normalize(x):
    user_id=x[0]
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
    
    return (user_id,(ratingList,rating_avg))
        

def convertFormat(x):
    userId=x[0][0]
    bussinessId_list=x[0][1]
    ratings_list=x[1][0]
    avgRating=x[1][1]
    
    x_formatted_list=[]
    for i in range(0,len(bussinessId_list)):
        x_formatted_list.append(((userId,bussinessId_list[i]),(ratings_list[i],avgRating)))
    return x_formatted_list
        

def getPredictedRatings(x,list_userId_bussinessIdi_ratingi_avgRating,list_user_businessList,list_business_usersList,N):
    
    testUserId=x[0]
    testBusinessId=x[1]
    
    list_businessesRatedByTestUser=list_user_businessList[testUserId]
    
    avgRating_testUser = list_userId_bussinessIdi_ratingi_avgRating[(testUserId,list_businessesRatedByTestUser[0])][1]
    
    
    
    if testBusinessId not in list_business_usersList.keys():
        return ((testUserId,testBusinessId),0)
    else:
        list_corated_usersList_forTestBusiness = list_business_usersList[testBusinessId]
        cosineSimilarities = []
        
        for corated_user in list_corated_usersList_forTestBusiness:
            business_rated_by_corated_user = list_user_businessList[corated_user]
            corated_business_set=set(business_rated_by_corated_user) & set(list_businessesRatedByTestUser)
            
            neighborhood_count=0
            num=0
            den1=0
            den2=0
            cosine_similarity=0
            
            avgRating_coraterUser_for_testBusiness = list_userId_bussinessIdi_ratingi_avgRating[(corated_user,testBusinessId)][1]
            
            for corated_business in corated_business_set:
                if corated_business != testBusinessId and neighborhood_count<N :
                    centeredRating_testUser_coratedBusiness=float(list_userId_bussinessIdi_ratingi_avgRating[(testUserId,corated_business)][0])-avgRating_testUser
                    centeredRating_coratedUser_coratedBusiness=float(list_userId_bussinessIdi_ratingi_avgRating[(corated_user,corated_business)][0])-avgRating_coraterUser_for_testBusiness
                    
                    num=num+centeredRating_testUser_coratedBusiness*centeredRating_coratedUser_coratedBusiness
                    den1=den1+centeredRating_testUser_coratedBusiness*centeredRating_testUser_coratedBusiness
                    den2=den2+centeredRating_coratedUser_coratedBusiness*centeredRating_coratedUser_coratedBusiness
                    neighborhood_count=neighborhood_count+1
                else:
                    pass
                
            if num != 0:
                cosine_similarity=float(num/math.sqrt(den1*den2))
            
            cosineSimilarities.append((corated_user,cosine_similarity,avgRating_coraterUser_for_testBusiness))
            
        cosineSimilarities.sort(key= lambda x: x[1],reverse=True)
        
        num1=0
        den1=0
        top_similar_users_count=0
        
        for cosSim in cosineSimilarities:
            if top_similar_users_count < N:
                corated_user_similar=cosSim[0]
                cosine_similarity_similar=cosSim[1]
                avgRating_coraterUser_similar=cosSim[2]
                
                centered_rating = float(list_userId_bussinessIdi_ratingi_avgRating[(corated_user_similar,testBusinessId)][0])-avgRating_coraterUser_similar
                
                num1=num1+centered_rating*cosine_similarity_similar
                den1=den1+abs(cosine_similarity_similar)
                top_similar_users_count=top_similar_users_count+1
            else:
                break
            
        if den1 == 0:
            return((testUserId,testBusinessId),0)
        else:
            rating_predicted=float(avgRating_testUser) + (num1/den1)
            return((testUserId,testBusinessId),rating_predicted)
        
                    
                    
            
            
            


trainingSet=sc.textFile(trainingFile).map(lambda x: x.split(','))
trainingSet= trainingSet.filter(lambda x: x[0]!= "user_id")

testSet=sc.textFile(testFile).map(lambda x: x.split(','))
testSet= testSet.filter(lambda x: x[0]!= "user_id")


rdd_user_businessList = trainingSet.map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y)
rdd_user_ratingsList = trainingSet.map(lambda x: (x[0],[x[2]])).reduceByKey(lambda x,y: x+y)

rdd_user_ratingsList_avgRating = rdd_user_ratingsList.map(lambda x: normalize(x))


rdd_user_businessList_join_ratingList_ratingAvg = rdd_user_businessList.join(rdd_user_ratingsList_avgRating).map(lambda x: ((x[0],x[1][0]),(x[1][1][0],x[1][1][1])))           


list_user_businessList = rdd_user_businessList.collectAsMap()

list_userId_bussinessIdi_ratingi_avgRating = rdd_user_businessList_join_ratingList_ratingAvg.flatMap(lambda x: convertFormat(x)).collectAsMap()


list_business_usersList=trainingSet.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y).collectAsMap()


rdd_test_user_business = testSet.map(lambda x: (x[0],x[1]))

N=14

predictedRatings=rdd_test_user_business.map(lambda x: getPredictedRatings(x,list_userId_bussinessIdi_ratingi_avgRating,list_user_businessList,list_business_usersList,N))

rdd_testData_predictedData_joined=testSet.map(lambda x: ((x[0],x[1]),float(x[2]))).join(predictedRatings)
mean_squared_error= rdd_testData_predictedData_joined.map(lambda x: ((x[1][0]-x[1][1])*(x[1][0]-x[1][1]))).mean()
print("Root mean_squared_error = "+str(math.sqrt(mean_squared_error)))

w=open(outPutFile,'w')

w.write("user_id,business_id,prediction")
for predictedRating in predictedRatings.collect():
    w.write("\n")
    w.write(predictedRating[0][0]+","+predictedRating[0][1]+","+str(predictedRating[1]))
    
w.close()
    
