import numpy as np
import numpy.matlib
import math
from sklearn import datasets
iris=datasets.load_iris()


def linearleastsquares(trainpercent,lmbda):



    a=iris.data[(iris.target==0)]
    b=iris.data[(iris.target==1)]
    c=iris.data[(iris.target==2)]
    np.random.shuffle(a)
    np.random.shuffle(b)
    np.random.shuffle(c)


    N=50
    v=int(50*(trainpercent/100))

    xtrain=a[0:v,:]
    xtest=a[v:50,:]






    ytrain=b[0:v,:]
    ytest=b[v:50,:]


    ztrain=c[0:v,:]
    ztest=c[v:50,:]


    y1=[];y2=[];y3=[]

    for i in range(0,v):
        y1.append([1,0,0]);
        y2.append([0,1,0]);
        y3.append([0,0,1]);

    train=np.concatenate([xtrain,ytrain,ztrain])
    test=np.concatenate([xtest,ytest,ztest])



    trainone=np.ones((150*(trainpercent/100),1))

    testone=np.ones((150*((100-trainpercent)/100),1))

    X=np.hstack((train,trainone))

    Xtest=np.hstack((test,testone))
    XT=X.T
    X2=np.dot(XT,X)
    id=lmbda*np.matlib.identity(5)

    pm=X2+id
    pminv=np.mat(pm).I
    inter=np.dot(pminv,XT)
    Y=np.concatenate([y1,y2,y3])
    theta=np.dot(inter,Y)

    trainx=np.full((len(xtrain),1),0,dtype=np.int)
    trainy=np.full((len(ytrain),1),1,dtype=np.int)
    trainz=np.full((len(ztrain),1),2,dtype=np.int)
    trainclass=np.concatenate([trainx,trainy,trainz])


    targetx=np.full((len(xtest),1),0,dtype=np.int)
    targety=np.full((len(ytest),1),1,dtype=np.int)
    targetz=np.full((len(ztest),1),2,dtype=np.int)
    testclass=np.concatenate([targetx,targety,targetz])

    predictiontest=np.dot(Xtest,theta)


    predictclasstest=np.argmax(predictiontest,axis=1)
    misclassificationError=0

    for i in range(0,(len(predictclasstest))):


        if predictclasstest[i]!=testclass[i]:
            misclassificationError +=1
    totalError=misclassificationError/(len(predictclasstest))
    print(round(totalError,4)*100)

    predictiontraining=np.dot(X,theta)


    predictclasstraining=np.argmax(predictiontraining,axis=1)
    misclassificationErrorTrain=0


    for i in range(0,(len(predictclasstraining))):


        if predictclasstraining[i]!=trainclass[i]:
            misclassificationErrorTrain +=1
    totalErrorTrain=misclassificationErrorTrain/(len(predictclasstraining))
    print(round(totalErrorTrain,4)*100)


linearleastsquares(10,math.pow(10,-8));
linearleastsquares(30,math.pow(10,-8));
linearleastsquares(50,math.pow(10,-8));