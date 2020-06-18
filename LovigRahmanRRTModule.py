import random
import matplotlib.pyplot as plt
import numpy as np
import math

class circleEstimator:

    def __init__(self,n,bins,COMP,COMQ,noiseM,noiseV,type):

        self.n = n
        self.COMP = COMP
        self.COMQ = COMQ
        self.noiseM = noiseM
        self.noiseV = noiseV
        self.type = type
        self.bins = bins

        distances = np.empty(n)
        distancesNoise = np.empty(n)
        counter = 0

        for person in range(0,self.n):

            if type == "N":
                if random.random() < .5:
                    randp = random.uniform(0,1)
                    randq = random.uniform(0,randp)

                else:
                    randq = random.uniform(0, 1)
                    randp = random.uniform(0, randq)

                dist = math.sqrt((randp-self.COMP)**2 + (randq-self.COMQ)**2)

            if type == "R":
                r = random.uniform(0, math.sqrt((1 - self.COMP) ** 2 + (0 - self.COMQ) ** 2))
                dist = r

            if type == "B":
                if random.random() < .5:
                    r = random.uniform(.6, .74)

                else:
                    r = random.uniform(0, .1)

                dist = r

            distances[counter] = dist
            distancesNoise[counter] = dist + random.gauss(self.noiseM,math.sqrt(self.noiseV))
            counter += 1

        distances.sort()
        distancesNoise.sort()

        self.mean0 = np.mean(distances)
        self.mean = np.mean(distancesNoise)

        self.var0 = np.var(distances)
        self.var = np.var(distancesNoise)-self.noiseV

        self.distances = distances
        self.distancesNoise = distancesNoise

    def getA(self,p,q):

        dist0 = math.sqrt((p - self.COMP) ** 2 + (q - self.COMQ) ** 2)
        found = False
        if dist0 < self.distances[0]:
            return(1)

        for i in range(0,self.n-1):
            if dist0 > self.distances[i] and dist0 < self.distances[i+1]:
                found = True
                return(1-(i+1)/self.n)

        if found == False:
            return 0

    def getNoisyA(self,p,q):

        dist0 = math.sqrt((p - self.COMP) ** 2 + (q - self.COMQ) ** 2)
        found = False
        if dist0 < self.distancesNoise[0]:
            return(1)

        for i in range(0,self.n-1):
            if dist0 > self.distancesNoise[i] and dist0 < self.distancesNoise[i+1]:
                found = True
                return(1-(i+1)/self.n)

        if found == False:
            return 0
    def getShape(self):

        counter = 0
        size = int((self.bins + 1) * self.bins / 2)
        x = np.empty(size)
        y = np.empty(size)
        z = np.empty(size)
        z0 = np.empty(size)

        for i in range(0, self.bins):
            for n in range(0, self.bins - i):
                p = i/self.bins
                q = n/self.bins
                x[counter] = p
                y[counter] = q
                z[counter] = self.getNoisyA(p, q)
                z0[counter] = self.getA(p, q)
                counter += 1

        fig = plt.figure()

        axScrambler = fig.add_subplot(111, projection='3d')

        trueCircle = axScrambler.plot_trisurf(x, y, z0)
        noisyCircle = axScrambler.plot_trisurf(x, y, z)

        axScrambler.set_xlabel("Direct Question Proportion")
        axScrambler.set_ylabel("Indirect Question Proportion")
        axScrambler.set_zlabel("Lie Proportion")

        axScrambler.set_xlim(0, 1)
        axScrambler.set_ylim(0, 1)
        axScrambler.set_zlim(0, 1)

        plt.show()
        plt.clf()
        plt.close()

    def getEstM(self):
        return self.mean

    def getTrueM(self):
        return self.mean0

    def getEstV(self):
        return self.var

    def getTrueV(self):
        return self.var0
class circleApproximator:

    def __init__(self,Estimator):

        self.n = Estimator.n
        self.meanR = Estimator.getEstM()
        self.varR = Estimator.getEstV()
        self.COMP = Estimator.COMP
        self.COMQ = Estimator.COMQ
        self.bins = Estimator.bins
        self.didMSE = False

        distances = np.empty(n)
        counter = 0

        for person in range(0,n):
            randr = random.gauss(self.meanR,math.sqrt(self.varR))
            if randr < 0:
                randr = 0

            distances[counter] = randr
            counter += 1

        distances.sort()
        self.distances = distances

    def getA(self,p,q):

        dist0 = math.sqrt((p - self.COMP) ** 2 + (q - self.COMQ) ** 2)
        found = False
        if dist0 < self.distances[0]:
            return(1)

        for i in range(0,self.n-1):
            if dist0 > self.distances[i] and dist0 < self.distances[i+1]:
                found = True
                return(1-(i+1)/self.n)

        if found == False:
            return 0

    def getMSE(self):

        counter = 0
        size = int((self.bins + 1) * self.bins / 2)
        self.x = np.empty(size)
        self.y = np.empty(size)
        self.z = np.empty(size)

        for i in range(0, self.bins):
            for n in range(0, self.bins - i):
                p = i/self.bins
                q = n/self.bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = self.getA(p, q)
                counter += 1

        self.didMSE = True

    def getShape(self):
        if self.didMSE == False:
            counter = 0
            size = int((self.bins + 1) * self.bins / 2)
            self.x = np.empty(size)
            self.y = np.empty(size)
            self.z = np.empty(size)

            for i in range(0, self.bins):
                for n in range(0, self.bins - i):
                    p = i / self.bins
                    q = n / self.bins
                    self.x[counter] = p
                    self.y[counter] = q
                    self.z[counter] = self.getA(p, q)
                    counter += 1

        fig = plt.figure()

        axApproximator = fig.add_subplot(111, projection='3d')

        ApproximatorCircle = axApproximator.plot_trisurf(self.x, self.y, self.z)

        axApproximator.set_xlabel("Direct Question Proportion")
        axApproximator.set_ylabel("Indirect Question Proportion")
        axApproximator.set_zlabel("Lie Proportion")

        axApproximator.set_xlim(0, 1)
        axApproximator.set_ylim(0, 1)
        axApproximator.set_zlim(0, 1)

        plt.show()
        plt.clf()
        plt.close()
class stdEstimator:
    def __init__(self, bins):
        self.bins = bins
        size = int(bins * (bins + 1) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins):
            for n in range(0, bins - i):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = (p * q * (1 - p - q) / ((1 / 3) ** 3))
                print(counter / size)
                counter += 1

    def getShape(self):
        fig = plt.figure()
        stdax = fig.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Lying Proportion")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0.0, 1)
        stdax.set_title("Standard Lying Distribution")
        plt.show()
class mixedModel:

    def __init__ (self,n,epochs,tPi,zPi,mode):

        self.n = n
        self.epochs = epochs
        self.tPi = tPi
        self.zPi = zPi
        self.mode = mode

    def run(self,p,q,a):

        sumOfError = 0

        for epoch in range(0,self.epochs):

            Yes, No = 0, 0
            for i in range(0,self.n):
                doesSensitive = random.random() <= self.tPi
                doesUnrelated = random.random() <= self.zPi
                tellsTruth = random.random() <= a

                qu = random.random()
                if qu<=p:
                    question = 0

                elif qu<=(p+q):
                    question = 1

                else:
                    question = 2

                if tellsTruth:
                    if question == 0:
                        if doesSensitive:
                            Yes += 1

                        else:
                            No += 1

                    elif question == 1:
                        if doesSensitive:
                            No += 1

                        else:
                            Yes += 1

                    else:
                        if doesUnrelated:
                            Yes += 1

                        else:
                            No += 1

                else:
                    if question == 0:
                        No += 1

                    elif question == 1:
                        Yes += 1

                    else:
                        if doesUnrelated:
                            Yes += 1

                        else:
                            No += 1

            if self.mode == False:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (p - q)

            if self.mode == True:
                piEst = (Yes/(No + Yes) - q*a - a*self.zPi*(1-p-q)-(1-a)*(q+self.zPi*(1-q-p)))/(a*p-a*q)

            sumOfError = sumOfError + (self.tPi - piEst)**2

        if self.mode == False:
            return (sumOfError / self.epochs)

        if self.mode == True:
            return (sumOfError / self.epochs)

    def getMinPQ(self,cieling,aFunction):

        bins = aFunction.bins
        size = int(bins * (bins + 1) / 2)
        counter = 0
        minM = 1
        minname = ""

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins):
            for n in range(0, bins - i):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a!=0:
                    p = i / bins
                    q = n / bins
                    M = self.run(p, q, a)
                    self.x[counter] = p
                    self.y[counter] = q

                    if M > cieling:
                        M = cieling

                    self.z[counter] = M

                    if M < minM:
                        minM = M
                        minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                    print(counter / size)

                counter += 1

        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)
        '''
        self.x = np.delete(x, 0)
        self.y = np.delete(y, 0)
        self.z = np.delete(z, 0)
        '''

        return minM, minname

    def getShape(self):
        fig = plt.figure()
        axMin = fig.add_subplot(111, projection='3d')
        Min = axMin.plot_trisurf(self.x, self.y, self.z, shade=True);
        axMin.set_xlabel("Direct Question Proportion")
        axMin.set_ylabel("Indirect Question Proportion")
        axMin.set_zlabel("MSE")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        axMin.set_zlim(0.0, cieling)
        axMin.set_title("Mixed Model Without Lies Accounted")

        plt.show()
        plt.clf()
        plt.close()

n = 100
#The Sample Size for Each Survey
bins = 100
#The Percision of P and Q, larger is more percise
epochs = 10
#How Many Times to Rerun a Survey for Each Individual P and Q
tPi = .3
#The True Proportion of PI_X
zPi = .7
#The Proportion of PI_Y
mode = True
#Does The Model Account for Lying?
cieling = .1
#Sets The Maximum Z Axis for The Graph, Should Be At Least 1
COMP = 1/3
COMQ = 1/3
#Defining The Center of The Distriburion of P and Q
ScramblerMean = 0
ScramblerVariance = 0

Estimator = circleEstimator(n,bins,COMP,COMQ,ScramblerMean,ScramblerVariance,"N")
Estimator.getShape()

Approximator = circleApproximator(Estimator)
Approximator.getShape()

MM = mixedModel(n,epochs,tPi,zPi,mode)
print(MM.getMinPQ(cieling,Approximator))
MM.getShape()

stdDist = stdEstimator(bins)
stdDist.getShape()

