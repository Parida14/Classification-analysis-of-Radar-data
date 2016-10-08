#installation and loading of packages
install.packages("ISLR")
install.packages("class")
install.packages("randomForest")
install.packages("FSelector")
install.packages("ROCR")
install.packages("e1071")
install.packages("pls")
library(ROCR)
library(FSelector)
library(randomForest)
library(class)
library(ISLR)
library(pls)
library(MASS)
library(e1071)
library(tree)

attach(ionosphere)
Response = ifelse(ionosphere$V35 == "g",1,0)
ionosphere1 = data.frame(ionosphere[,1:34],Response)
ionosphere1 = ionosphere1[,-2] #deducting variable V2 from our analysis as it has same value in all observations

#check for missing values
ionosphere2 <- na.omit(ionosphere1)

#using Fselector package
fsel_result = cfs(Response~., data = ionosphere2)
f <- as.simple.formula(fsel_result,"Response")
print(f)

subset = consistency(Response~.,data = ionosphere2)
f=as.simple.formula(subset,"Response")
print(f)

#using PCA
pr.out = prcomp(ionosphere2,scale = T)
screeplot(ionosphere2, type = "lines")
plot(pr.out,type = "lines",lwd = 4, col = "blue")
summary(pr.out)
m = lm(Response~pr.out$x[,1]+pr.out$x[,2]+pr.out$x[,3]+pr.out$x[,4]+pr.out$x[,5]+pr.out$x[,6]+pr.out$x[,7]+pr.out$x[,8]+pr.out$x[,9], data = ionosphere2)
n = pr.out$rotation[,3]
summary(m)


#pca find important variables, eigen graph, pick best rotation

#using PLS
pls.model=plsr(Response~.,data=ionosphere2,scale=T,validation="CV")
summary(pls.model)
plot(RMSEP(pls.model), legendpos = "bottomright",lwd=5)
plot(pls.model, plottype = "scores", comps = 1:5,col="blue")

#using Random Forest
m = randomForest(Response~.,data = ionosphere2, importance = T)
varImpPlot(m)

#using SVM ranking
for(i in seq(1,5,1)){
  set.seed(i)
  d = ionosphere2[sample(nrow(ionosphere2)),]
  x= d[,-1]
  y= d[,1]
  svmFeatureRanking = function(x,y)
  {
    nc = ncol(x)
    survivingFeaturesIndexes = seq(1:nc)
    featureRankedList = vector(length=nc)
    rankedFeatureIndex = nc
    while(length(survivingFeaturesIndexes)>0)
    {
      #train the svm
      svmModel = svm(x[, survivingFeaturesIndexes], y,type="C-classification",cost=10,gamma=0.5, kernel="radial" )
      #weight vector computation
      w = t(svmModel$coefs)%*%svmModel$SV
      #nking criteria computation
      rankingCriteria = w * w
      #feature ranking
      ranking = sort(rankingCriteria, index.return = TRUE)$ix
      #update feature ranked list
      featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]]
      rankedFeatureIndex = rankedFeatureIndex - 1
      
      #eliminate the feature with smallest ranking criterion
      (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]])       
    }
    return (featureRankedList)
  }
  featureRankedList = svmFeatureRanking(x,y)
  print(featureRankedList[1:20])
}


#Classification Techniques

#Logistic Regression

glm.fit1 = glm(Response~V12+V17+V33+V1+V25+V24+V28+V20+V30+V4+V22+V19+V7+V5+V29+V14+V31+V21+V8+V27,family = "binomial",data = ionosphere2)
summary(glm.fit1)

glm.fit2 = glm(Response~V12+V17+V33+V1+V25+V24+V28+V30+V4+V22+V7+V5+V29+V14+V8+V27,family = "binomial",data = ionosphere2)
summary(glm.fit2)

glm.fit3 = glm(Response~V33+V1+V4+V22+V7+V5+V29+V8+V27,family = "binomial",data = ionosphere2)
summary(glm.fit3)

glm.fit4 = glm(Response~V1+V4+V22+V7+V5+V29+V8+V27,family = "binomial",data = ionosphere2)
summary(glm.fit4)

#cross-validation on #glmiteration3

error = rep(0,20)
for (i in 1:20){
  set.seed(i)
  train = sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  a.train = ionosphere2[train,]
  a.test = ionosphere2[-train,]
  testresponse = a.test$Response
  y=glm(Response~V33+V1+V4+V22+V7+V5+V29+V8+V27,data = a.train,family = "binomial")
  pred=predict(y,a.test)
  a=rep(0,length(pred))
  a[pred>0.5]=1
  error[i]=mean(a!=testresponse)
}
mean(error)

#cross-validation on #glmiteration4

error = rep(0,20)
for (i in 1:20){
  set.seed(i)
  train = sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  a.train = ionosphere2[train,]
  a.test = ionosphere2[-train,]
  testresponse = a.test$Response
  y=glm(Response~V1+V4+V22+V7+V5+V29+V8+V27,data = a.train,family = "binomial")
  pred=predict(y,a.test)
  a=rep(0,length(pred))
  a[pred>0.5]=1
  error[i]=mean(a!=testresponse)
}
mean(error)

#Cross validation for selection using most Influential features for LDA
#iteration3

error.20=rep(0,20)
for(i in 1:20)
{
  set.seed(i)
  train=sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  x.train=ionosphere2[train,]
  x.test=ionosphere2[-train,]
  testresponse=x.test$Response
  ldam=lda(Response~V33+V1+V4+V22+V7+V5+V29+V8+V27,data=x.train)
  pred=predict(ldam,x.test,type="response")
  predos=pred$class
  error.20[i]=mean(predos!=testresponse)
}
mean(error.20)

#LDA CV iteration4

error.20=rep(0,20)
for(i in 1:20)
{
  set.seed(i)
  train=sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  x.train=ionosphere2[train,]
  x.test=ionosphere2[-train,]
  testresponse=x.test$Response
  ldam=lda(Response~V1+V4+V22+V7+V5+V29+V8+V27,data=x.train)
  pred=predict(ldam,x.test,type="response")
  predos=pred$class
  error.20[i]=mean(predos!=testresponse)
}
mean(error.20)

#qda cross-validation for iteration 3
error.20=rep(0,20)
for(i in 1:20)
{
  set.seed(i)
  train=sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  x.train=ionosphere2[train,]
  x.test=ionosphere2[-train,]
  testresponse=x.test$Response
  qdam=qda(Response~V33+V4+V22+V7+V5+V29+V8+V27,data=x.train)
  pred=predict(qdam,x.test,type="response")
  predos=pred$class
  error.20[i]=mean(predos!=testresponse)
}
mean(error.20)

#qda cross-validation for iteration 4
error.20=rep(0,20)
for(i in 1:20)
{
  set.seed(i)
  train=sample(1:nrow(ionosphere2),nrow(ionosphere2)/1.25)
  x.train=ionosphere2[train,]
  x.test=ionosphere2[-train,]
  testresponse=x.test$Response
  qdam=qda(Response~V4+V22+V7+V5+V29+V8+V27,data=x.train)
  pred=predict(qdam,x.test,type="response")
  predos=pred$class
  error.20[i]=mean(predos!=testresponse)
}
mean(error.20)



#tuning of svm parameters
ionosphere3 = ionosphere2
attach(ionosphere3)
tune.out = tune(svm,Response~V1+V4+V22+V7+V5+V29+V8+V27, data = x.train, kernel = "radial", ranges = list(cost=c(0.1,1,10,100,1000),gamma = c(0.5,1,2,3,4)))
summary(tune.out)


#iteration 1 svm tuning
g=data.frame(matrix(nrow=100, ncol=10))
for(i in seq(101,200,1))
{
  set.seed(i)
  ionosphere3=ionosphere3[sample(nrow(ionosphere3)),]
  folds=cut(seq(nrow(ionosphere3)),breaks=10,labels=FALSE)
  for(j in 1:10)
  {
    testIndex=which(folds==j,arr.ind=TRUE)
    testdata=ionosphere3[testIndex,]
    traindata=ionosphere3[-testIndex,]
    y=svm(Response~V1+V4+V22+V7+V5+V29+V8+V27,cost=1,gamma=0.5,data=traindata,kernel="radial")
    pred=predict(y,testdata)
    a=rep(0,length(pred))
    a[pred>0.5]=1
    g[i,j]=mean(a==testdata$Response)
  }
}
k=na.omit(g)
mean(as.matrix(k))

#iteration 2 svm tuning
g=data.frame(matrix(nrow=100, ncol=10))
for(i in seq(101,200,1))
{
  set.seed(i)
  ionosphere3=ionosphere3[sample(nrow(ionosphere3)),]
  folds=cut(seq(nrow(ionosphere3)),breaks=10,labels=FALSE)
  for(j in 1:10)
  {
    testIndex=which(folds==j,arr.ind=TRUE)
    testdata=ionosphere3[testIndex,]
    traindata=ionosphere3[-testIndex,]
    y=svm(Response~V4+V22+V7+V5+V29+V8+V27+V33,cost=1,gamma=0.5,data=traindata,kernel="radial")
    pred=predict(y,testdata)
    a=rep(0,length(pred))
    a[pred>0.5]=1
    g[i,j]=mean(a==testdata$Response)
  }
}
k=na.omit(g)
mean(as.matrix(k))

#svm tuning iteration 3 
g=data.frame(matrix(nrow=100, ncol=10))
for(i in seq(101,200,1))
{
  set.seed(i)
  ionosphere3=ionosphere3[sample(nrow(ionosphere3)),]
  folds=cut(seq(nrow(ionosphere3)),breaks=10,labels=FALSE)
  for(j in 1:10)
  {
    testIndex=which(folds==j,arr.ind=TRUE)
    testdata=ionosphere3[testIndex,]
    traindata=ionosphere3[-testIndex,]
    y=svm(Response~V33+V1+V4+V22+V7+V5+V29+V8+V27,cost=1,gamma=0.5,data=traindata,kernel="radial")
    pred=predict(y,testdata)
    a=rep(0,length(pred))
    a[pred>0.5]=1
    g[i,j]=mean(a==testdata$Response)
  }
}
k=na.omit(g)
mean(as.matrix(k))

#decision tree
attach(ionosphere)
set.seed(200)
t = sample(dim(ionosphere)[1],300)
training = ionosphere[t, ]
testing = ionosphere[-t,]

#fitting a classification tree
tree.ion  = tree(V35~V33+V4+V22+V7+V5+V29+V8+V27, data = training)
plot(tree.ion,lwd=3,col="red")
text(tree.ion,pretty = 0)

cv.tree.ion = cv.tree(tree.ion,FUN = prune.misclass)
plot(cv.tree.ion$size,cv.tree.ion$dev,type="b",col="blue",lwd = 3)

#pruning
prunedmodel = prune.misclass(tree.ion,best = 3)
z = predict(prunedmodel,newdata = testing, type = "class")
testresponse = testing$V35
mean(z!=testresponse)
table(z,testresponse)

#bagging
bagmodel.tree = randomForest(V35~V33+V4+V22+V7+V5+V29+V8+V27,data = training,importance = T)
predict.bag  = predict(bagmodel.tree,newdata = testing)
mean(predict.bag!=testresponse)

#randomforest
rfmodel.tree = randomForest(V35~V33+V4+V22+V7+V5+V29+V8+V27,data = training,importance = T,mtry = 3)
predict.rf  = predict(rfmodel.tree,newdata = testing)
mean(predict.rf!=testresponse)


#roc curves
set.seed(12)
#roc glm
glm=glm(Response~V1+V4+V22+V7+V5+V29+V8+V27, family="binomial" ,data=x.train)
pred=predict(glm,x.test,type="response")
predos=
predis=ROCR::prediction(predos,testingvector)
perf=performance(predis,"tpr","fpr")
plot(perf)
perf=performance(predis,"auc")
perf

#roc lda
ldam=lda(Response~V33+V1+V4+V22+V7+V5+V29+V8+V27 ,data=x.train)
pred=predict(ldam,x.test,type="response")
predos=pred$posterior[,2]
predis=ROCR::prediction(predos,testingvector)
perf=performance(predis,"tpr","fpr")
plot(perf)
perf=performance(predis,"auc")
perf

set.seed(20)

index=sample(1:nrow(ionosphere2),nrow(ionosphere2)*0.8)

train=ionosphere2[index,]

test=ionosphere2[-index,]

y=svm(Response~V4+V22+V7+V5+V29+V8,cost=1,gamma=0.5,data=train,kernel="radial")

pred=predict(y,test)

predos=ROCR::prediction(pred,test$Response)

perf=performance(predos,"tpr","fpr")

plot(perf,lwd=3,col="blue")

performance(predos,"auc")
