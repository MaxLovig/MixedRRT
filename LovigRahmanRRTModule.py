import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import xlsxwriter

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

        self.further = max(math.sqrt((0 - self.COMP) ** 2 + (1 - self.COMQ) ** 2),
                           math.sqrt((1 - self.COMP) ** 2 + (0 - self.COMQ) ** 2))

        out = random.uniform(0, self.further/2)
        ins = random.uniform(0, self.further/2)
        mean = random.random()
        var = random.random()
        #mean = random.uniform(0,self.further)
        #var = random.uniform(0,min(mean,1-mean)/random.uniform(0,2))
        #var = random.uniform(0, self.further)
        for person in range(0,self.n):
            if type == "H":
                if random.random() < .5:
                    randp = random.uniform(0,1)
                    randq = random.uniform(0,randp)

                else:
                    randq = random.uniform(0, 1)
                    randp = random.uniform(0, randq)

                dist = math.sqrt((randp-self.COMP)**2 + (randq-self.COMQ)**2)
            if type == "N":
                r = random.gauss(mean, var)
                dist = r
            if type == "P":

                r = random.uniform(0, self.further)
                dist = r

            if type == "Pl":
                if random.random() < .5:
                    r = random.uniform(self.further-ins, self.further)

                else:
                    r = random.uniform(0, out)

                dist = r

            distances[counter] = dist
            distancesNoise[counter] = dist + random.gauss(self.noiseM,math.sqrt(self.noiseV))
            counter += 1

        distances.sort()
        distancesNoise.sort()

        self.mean0 = np.mean(distances)
        self.mean = np.mean(distancesNoise)- self.noiseM

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
        size = int((self.bins+1) * (self.bins + 2) / 2)
        x = np.empty(size)
        y = np.empty(size)
        z = np.empty(size)
        z0 = np.empty(size)

        for i in range(0, self.bins+1):
            for n in range(0, self.bins - i+1):
                p = i/self.bins
                q = n/self.bins
                x[counter] = p
                y[counter] = q
                z[counter] = self.getNoisyA(p, q)
                z0[counter] = self.getA(p, q)
                counter += 1

        fig1 = plt.figure()

        axScrambler = fig1.add_subplot(111, projection='3d')

        trueCircle = axScrambler.plot_trisurf(x, y, z0);

        noisyCircle = axScrambler.plot_trisurf(x, y, z);

        axScrambler.set_xlabel("Direct Question Proportion")
        axScrambler.set_ylabel("Indirect Question Proportion")
        axScrambler.set_zlabel("Lie Proportion")
        axScrambler.set_title("Actual Distribution " + self.type)

        axScrambler.set_xlim(0, 1)
        axScrambler.set_ylim(0, 1)
        axScrambler.set_zlim(0, 1)

    def getEstM(self):
        return self.mean

    def getTrueM(self):
        return self.mean0

    def getEstV(self):
        return self.var

    def getTrueV(self):
        return self.var0

class circleApproximator:

    def __init__(self,Estimator, mode, useEstN):
        if useEstN:
            self.n = Estimator.n
        else:
            self.n = 10000
        self.meanR = Estimator.getEstM()
        self.varR = Estimator.getEstV()
        self.COMP = Estimator.COMP
        self.COMQ = Estimator.COMQ
        self.bins = Estimator.bins
        self.didMSE = False
        self.mode = mode
        self.Estimator = Estimator
        self.further = Estimator.further
        self.noise = Estimator.noiseV

        distances = np.empty(self.n)
        counter = 0

        out = -0.056583143673601 -0.004751539350972*self.n/1000 -0.023889342443338*self.COMP/.5-0.01030636407667*self.noise/.3+ 1.3528038890947*self.meanR-2.1471346903107*self.varR
        for i in range(0,100):
            ins = 1.39671996598428 -0.00209757809825 * self.n / 1000 -0.45087561361082 * self.COMP / .5 + 0.005035063384458 * self.noise / .3 -2.09996626412467 * self.meanR -0.44800073325705 * self.varR + 0.504567930875165 * out
            out = -0.613594797288792 - 0.002921505986497 * self.n / 1000 + 0.164579261531773 * self.COMP / .5 - 0.010239108626763 * self.noise / .3+ 1.9298519873225 * self.meanR - 1.52367828721598 * self.varR + 0.407121690752586 * ins
        sd = math.sqrt(abs(self.varR))
        for person in range(0,self.n):

            if self.mode == "Reg":

                if random.random() < .5:
                    randr = random.uniform(self.further - ins, self.further)

                else:
                    randr = random.uniform(0, out)


            if self.mode == "N":
                randr = random.gauss(self.meanR,sd)

            if self.mode == "Exp":
                randr = random.expovariate((1/self.meanR + 1/sd)/2)
                if randr>self.further:
                    randr = randr = abs(randr-self.further)

            if self.mode == "Z":
                randr = random.gauss(self.meanR, sd)
                if randr < 0:
                    randr = 0
                if randr > self.further:
                    randr = self.further

            if self.mode == "A":
                randr = abs(random.gauss(self.meanR, sd))
                if randr > self.further:
                    randr = abs(2*self.further-randr)

            if self.mode == "E":
                redo = True
                while redo:
                    randr = random.gauss(self.meanR, sd)
                    if randr > 0 and randr < self.further:
                        redo = False


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

    def getMSE(self,sec):

        sumOfError = 0
        counter = 0
        for i in range(0, sec+1):
            for n in range(0, sec -i+1):
                p = i / sec
                q = n / sec
                sumOfError = sumOfError + (self.getA(p,q)-self.Estimator.getA(p,q))**2
                counter +=1
        return sumOfError/counter

    def getShape(self):

        counter = 0
        size = int((self.bins+1) * (self.bins + 2) / 2)
        self.x = np.empty(size)
        self.y = np.empty(size)
        self.z = np.empty(size)

        for i in range(0, self.bins+1):
            for n in range(0, self.bins - i+1):
                p = i / self.bins
                q = n / self.bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = self.getA(p, q)
                counter += 1

        fig2 = plt.figure()

        axApproximator = fig2.add_subplot(111, projection='3d')

        ApproximatorCircle = axApproximator.plot_trisurf(self.x, self.y, self.z)

        axApproximator.set_xlabel("Direct Question Proportion")
        axApproximator.set_ylabel("Indirect Question Proportion")
        axApproximator.set_zlabel("Lie Proportion")

        axApproximator.set_xlim(0, 1)
        axApproximator.set_ylim(0, 1)
        axApproximator.set_zlim(0, 1)
        axApproximator.set_title("Approximated Distribution, " + self.mode)

class stdEstimator:
    def __init__(self, bins):
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = (p * q * (1 - p - q) / ((1 / 3) ** 3))
                counter += 1

    def getA(self,p,q):
        return (p*q*(1-p-q))/((1/3)**3)
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Lying Proportion")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0.0, 1)
        stdax.set_title("Standard Lying Distribution")

class comfortGraph:
    def __init__(self, bins,tPi,zPi):
        Cq = zPi / (1 - tPi + (zPi / tPi))
        Cp = zPi/(tPi-tPi*tPi+zPi)-zPi/(1-tPi+(zPi/tPi))
        c = max(-1+Cp+2*Cq,-1+Cq+2*Cp,0)
        #c=-1+Cp+Cq
        #c=1
        a = 1-Cq-2*Cp +c
        b = 1-Cp-2*Cq +c
        d = ((a+Cp) * (b+Cq) * (c+ 1 - Cp - Cq))
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        m = min(a*b*(c+1), a*(b+1)*c,(a+1)*b*c)
        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = ((a+p) * (b+q) * (c + 1 - p - q)-m)/(d-m)
                counter += 1
        max1 = self.z.max()
        k = np.where(self.z == max1)
        self.Cp = Cp
        self.Cq = Cq
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.m = m

    def getA(self,p,q):
        return ((((self.a+p) * (self.b+q) * (self.c + 1 - p - q))-self.m)/(self.d-self.m))**2
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Privacy")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0, 1)
        stdax.set_title("")

class comfortGraph2:
    def __init__(self, bins,tPi,zPi):
        Cq = zPi / (1 - tPi + (zPi / tPi))
        Cp = zPi/(tPi-tPi*tPi+zPi)-zPi/(1-tPi+(zPi/tPi))
        print(Cp, " : ", Cq)
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = ((tPi*p) * ((1-tPi)*q) * ((1-p-q)*zPi))/((tPi*Cp) * ((1-tPi)*Cq) * ((1-Cp-Cq)*zPi))
                counter += 1
        max1 = self.z.max()
        k = np.where(self.z == max1)
        print(k)
        print(max1)
        print(self.x[k], " : ", self.y[k])
        self.Cp = Cp
        self.Cq = Cq
    def getA(self,p,q):
        return ((self.a+p) * (self.b+q) * (self.c + 1 - p - q))/self.d
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Privacy")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0, 1)
        stdax.set_title("")

class weird2Estimator:
    def __init__(self, bins):
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = math.sqrt((((1+p) * (1+q) * (2 - p - q))-2)*2.7)
                counter += 1

    def getA(self,p,q):
        return math.sqrt((((1+p) * (1+q) * (2 - p - q))-2)*2.7)
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Privacy")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0, 1)
        stdax.set_title("")

class elevatedStdEstimator:
    def __init__(self, bins):
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = (p * q * (1 - p - q) / ((1 / 3)**2))+(2/3)
                counter += 1

    def getA(self,p,q):
        return (p * q * (1 - p - q) / ((1 / 3)))+(1-1/(3*3*math.sqrt(3)))
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Lying Proportion")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0.0, 1)
        stdax.set_title("Standard Lying Distribution")

class focusedEstimator:
    def __init__(self, bins, COMP, COMQ, slope):
        self.COMP = COMP
        self.COMQ = COMQ
        self.slope = slope
        self.bins = bins
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for n in range(0, bins - i+1):
                p = i/bins
                q = n/bins
                self.x[counter] = p
                self.y[counter] = q
                self.z[counter] = 1-math.sqrt((p-COMP)**2 + (q-COMQ)**2)/self.slope
                counter += 1

    def getA(self, p, q):
        return (1-math.sqrt((p-self.COMP)**2 + (q-self.COMQ)**2)/self.slope)
    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Lying Proportion")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0.0, 1)
        stdax.set_title("Standard Lying Distribution")

class constantEstimator:
    def __init__(self,a):
        self.a = a
    def getA(self,p,q):
        return self.a

class privacyEstimator:

    def __init__(self, bins, n, tpi, zpi):
        self.bins = bins
        self.n = n
        self.tpi = tpi
        self.zpi = zpi
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)

        for i in range(0, bins+1):
            for c in range(0, bins - i+1):
                if c!=i and n!=0 and i!=0:
                    p = i/bins
                    q = c/bins
                    self.x[counter] = p
                    self.y[counter] = q
                    self.z[counter] = 1.2-getTrueMSEwithPrivacy(n,tpi,zpi,p,q,1,False)[3]
                counter += 1

        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)

    def getA(self,p,q):
        if(p!=q):
            return 1-getTrueMSEwithPrivacy(self.n,self.tpi,self.zpi,p,q,1,False)[3]
        else:
            return .001

    def getShape(self):
        fig3 = plt.figure()
        stdax = fig3.add_subplot(111, projection='3d')
        pnt3d = stdax.plot_trisurf(self.x, self.y, self.z, shade=True);
        stdax.set_xlabel("Direct Question Proportion")
        stdax.set_ylabel("Indirect Question Proportion")
        stdax.set_zlabel("Lying Proportion")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        stdax.set_zlim(0.0, 1)
        stdax.set_title("Standard Lying Distribution")

class mixedModel:

    def __init__ (self,n,epochs,tPi,zPi,mode):

        self.n = n
        self.epochs = epochs
        self.tPi = tPi
        self.zPi = zPi
        self.mode = mode

    def run(self,p,q,a):
        sumOfBias = 0
        sumOfError = 0
        totalPiEst = 0
        privacy1 = 0
        privacy2 =  0
        for epoch in range(0,self.epochs):

            Yes, No = 0, 0
            YesandSens, NoandSens = 0,0
            for i in range(0,self.n):
                doesSensitive = random.random() <= self.tPi
                doesUnrelated = random.random() <= self.zPi
                tellsTruth = random.random() <= a

                qu = random.random()
                if qu <= p:
                    question = 0

                elif qu <= (p + q):
                    question = 1

                else:
                    question = 2

                if tellsTruth:
                    if question == 0:
                        if doesSensitive:
                            Yes += 1
                            YesandSens += 1

                        else:
                            No += 1

                    elif question == 1:
                        if doesSensitive:
                            No += 1
                            NoandSens += 1

                        else:
                            Yes += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens += 1

                else:
                    if question == 0:
                        No += 1
                        if doesSensitive:
                            NoandSens += 1

                    elif question == 1:
                        Yes += 1
                        if doesSensitive:
                            YesandSens += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens += 1

            if self.mode == False:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (p - q)

            if self.mode == True:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (a*(p - q))

            privacy1 , privacy2 = privacy1 + (YesandSens / Yes) ,privacy2 + (NoandSens / No)

            totalPiEst = totalPiEst + piEst
            sumOfError = sumOfError + (self.tPi - piEst)**2
            sumOfBias = sumOfBias + piEst


        MSE = (sumOfError / self.epochs)
        bias = (sumOfBias/self.epochs - self.tPi)
        var = MSE - bias**2
        avrPiEst = totalPiEst/self.epochs
        avrPrivacy1 = privacy1/self.epochs
        avrPrivacy2 = privacy2/self.epochs
        avrPrivacy = max(avrPrivacy1,avrPrivacy2)
        comf = comfortGraph(1,self.tPi,self.zPi)
        comfort = comf.getA(p,q)
        return MSE, bias, var, avrPiEst, avrPrivacy, MSE * avrPrivacy

    def runRandom(self,p,q,a):
        sumOfBias = 0
        sumOfError = 0
        totalPiEst = 0
        privacy1 = 0
        privacy2 =  0
        for epoch in range(0,self.epochs):
            self.tPi = random.uniform(.1,.3)
            Yes, No = 0, 0
            YesandSens, NoandSens = 0,0
            for i in range(0,self.n):
                doesSensitive = random.random() <= self.tPi
                doesUnrelated = random.random() <= self.zPi
                tellsTruth = random.random() <= a

                qu = random.random()
                if qu <= p:
                    question = 0

                elif qu <= (p + q):
                    question = 1

                else:
                    question = 2

                if tellsTruth:
                    if question == 0:
                        if doesSensitive:
                            Yes += 1
                            YesandSens += 1

                        else:
                            No += 1

                    elif question == 1:
                        if doesSensitive:
                            No += 1
                            NoandSens += 1

                        else:
                            Yes += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens += 1

                else:
                    if question == 0:
                        No += 1
                        if doesSensitive:
                            NoandSens += 1

                    elif question == 1:
                        Yes += 1
                        if doesSensitive:
                            YesandSens += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens += 1

            if self.mode == False:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (p - q)

            if self.mode == True:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (a*(p - q))

            privacy1 , privacy2 = privacy1 + (YesandSens / Yes) ,privacy2 + (NoandSens / No)

            totalPiEst = totalPiEst + piEst
            sumOfError = sumOfError + (self.tPi - piEst)**2
            sumOfBias = sumOfBias + piEst


        MSE = (sumOfError / self.epochs)
        bias = (sumOfBias/self.epochs - self.tPi)
        var = MSE - bias**2
        avrPiEst = totalPiEst/self.epochs
        avrPrivacy1 = privacy1/self.epochs
        avrPrivacy2 = privacy2/self.epochs
        avrPrivacy = max(avrPrivacy1,avrPrivacy2)
        return MSE, bias, var, avrPiEst, avrPrivacy, MSE * avrPrivacy

    def runEst(self,p,q,a,aEst):
        sumOfBias = 0
        sumOfError = 0
        totalPiEst = 0
        privacy = 0
        for epoch in range(0,self.epochs):

            Yes, No = 0, 0
            YesandSens, NoandSens= 0,0
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
                            YesandSens +=1

                        else:
                            No += 1

                    elif question == 1:
                        if doesSensitive:
                            No += 1
                            NoandSens += 1

                        else:
                            Yes += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens +=1

                else:
                    if question == 0:
                        No += 1
                        if doesSensitive:
                            NoandSens += 1

                    elif question == 1:
                        Yes += 1
                        if doesSensitive:
                            YesandSens += 1

                    else:
                        if doesUnrelated:
                            Yes += 1
                            if doesSensitive:
                                YesandSens += 1

                        else:
                            No += 1
                            if doesSensitive:
                                NoandSens += 1

            if self.mode == False:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (p - q)

            if self.mode == True:
                piEst = ((Yes / (Yes + No)) - self.zPi * (1 - q - p) - q) / (aEst*(p - q))

            privacy += max(YesandSens/Yes, NoandSens/No)

            totalPiEst = totalPiEst + piEst
            sumOfError = sumOfError + (self.tPi - piEst)**2
            sumOfBias = sumOfBias + piEst


        MSE = (sumOfError / self.epochs)
        bias = (sumOfBias/self.epochs - self.tPi)
        var = MSE - bias**2
        avrPiEst = totalPiEst/self.epochs
        avrPrivacy = privacy/self.epochs
        return MSE, bias, var, avrPiEst, avrPrivacy

    def getMinMSE(self,bins,cieling,aFunction,throughSim):
        self.cieling = cieling
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        minM = 1
        minname = ""

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)
        bestP = 0
        bestQ = 0
        for i in range(0, bins+1):
            for n in range(0, bins - i + 1):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a!=0:
                    p = i / bins
                    q = n / bins
                    if throughSim:
                        M = self.run(p, q, a)[0]
                    else:
                        M = getTrueMSE(self.n,self.tPi,self.zPi,p,q,a,self.mode)[0]
                    self.x[counter] = p
                    self.y[counter] = q

                    if M > self.cieling:
                        M = self.cieling

                    self.z[counter] = M

                    if M < minM:
                        minM = M
                        bestP = p
                        bestQ = q
                        minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                counter += 1

        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)
        print(minname)
        return minM, bestP, bestQ

    def getMinPrivacy(self,bins,cieling,aFunction,throughSim):
        self.cieling = cieling
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        minM = 1
        minname = ""

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)
        bestP = 0
        bestQ = 0
        for i in range(0, bins+1):
            for n in range(0, bins - i + 1):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a!=0:
                    p = i / bins
                    q = n / bins
                    if throughSim:
                        M = self.run(p, q, a)[5]
                    else:
                        M = getTrueMSEwithPrivacy(self.n,self.tPi,self.zPi,p,q,a,self.mode)[4]
                    self.x[counter] = p
                    self.y[counter] = q

                    if M > self.cieling:
                        M = self.cieling

                    self.z[counter] = M

                    if M < minM:
                        minM = M
                        bestP = p
                        bestQ = q
                        minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                counter += 1
        '''
        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)
                '''
        print(minname)
        return minM, bestP, bestQ

    def getMinComfort(self,bins,cieling,aFunction,throughSim):
        comf = comfortGraph(1,self.tPi,self.zPi)
        self.cieling = cieling
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        minM = 1
        minname = ""

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)
        bestP = 0
        bestQ = 0
        for i in range(0, bins+1):
            for n in range(0, bins - i + 1):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a!=0:
                    p = i / bins
                    q = n / bins
                    if throughSim:
                        M = self.run(p, q, a)[5]
                    else:
                        M = getTrueMSEwithPrivacy(self.n,self.tPi,self.zPi,p,q,a,self.mode)[0]
                        M = M / (comf.getA(p,q)+.00001)
                        print(comf.getA(p,q))
                    self.x[counter] = p
                    self.y[counter] = q

                    if M > self.cieling:
                        M = self.cieling

                    self.z[counter] = M

                    if M < minM:
                        minM = M
                        bestP = p
                        bestQ = q
                        minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                counter += 1
        '''
        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)
                '''
        print(minname)
        return minM, bestP, bestQ

    def getMinPrivacyRandom(self,bins,cieling,aFunction,throughSim):
        self.cieling = cieling
        size = int((bins+1) * (bins + 2) / 2)
        counter = 0
        minM = 1
        minname = ""

        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.z = np.zeros(size)
        bestP = 0
        bestQ = 0
        for i in range(0, bins+1):
            for n in range(0, bins - i + 1):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a!=0 and p>q:
                    p = i / bins
                    q = n / bins
                    if throughSim:
                        M = self.runRandom(p, q, a)[5]
                    else:
                        M = getTrueMSEwithPrivacy(self.n,self.tPi,self.zPi,p,q,a,self.mode)[4]
                    self.x[counter] = p
                    self.y[counter] = q

                    if M > self.cieling:
                        M = self.cieling

                    self.z[counter] = M

                    if M < minM:
                        minM = M
                        bestP = p
                        bestQ = q
                        minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                counter += 1

        for i in range(1, counter+1):
            if self.z[counter - i] == 0:  # or z[counter - i ] >.1:
                self.x = np.delete(self.x, counter - i)
                self.y = np.delete(self.y, counter - i)
                self.z = np.delete(self.z, counter - i)
        print(minname)
        return minM, bestP, bestQ

    def getMinMSEWithPrivacyConstrained(self, bins, cieling, aFunction, throughSim,minPrivacy, maxPrivacy):
        self.cieling = cieling
        size = int((bins + 1) * (bins + 2) / 2)
        counter = 0
        minM = 1
        minname = ""
        bestP = 0
        bestQ = 0
        for i in range(0, bins + 1):
            for n in range(0, bins - i + 1):
                p = i / bins
                q = n / bins
                a = aFunction.getA(p, q)
                if i != n and a != 0 :
                    pri = getTrueMSEwithPrivacy(self.n, self.tPi, self.zPi, p, q, a, self.mode)[3]
                    if pri<=maxPrivacy and pri>=minPrivacy:
                        p = i / bins
                        q = n / bins
                        print(str(p) + " equals direct pct " + str(q) + " equals indirect pct")
                        if throughSim:
                            M = self.run(p, q, a)[0]
                        else:
                            M = getTrueMSEwithPrivacy(self.n, self.tPi, self.zPi, p, q, a, self.mode)[0]

                        if M > self.cieling:
                            M = self.cieling

                        if M < minM:
                            minM = M
                            bestP = p
                            bestQ = q
                            minname = str(p) + " equals direct pct " + str(q) + " equals indirect pct"

                counter += 1
        print(minname)
        return minM, bestP, bestQ

    def getShape(self,title):
        fig4 = plt.figure()
        axMin = fig4.add_subplot(111, projection='3d')
        Min = axMin.plot_trisurf(self.x, self.y, self.z, shade=True);
        axMin.set_xlabel("Direct Question Proportion")
        axMin.set_ylabel("Indirect Question Proportion")
        axMin.set_zlabel("MSE")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        axMin.set_zlim(0.0, self.cieling)
        if self.mode == False:
            axMin.set_title(title)
        else:
            axMin.set_title("Mixed Model With Lies Accounted")

    def getMinMSEDescent(self,stepSize, aFunction, startP,startQ):
        cont = True
        bestP = startP
        bestQ = startQ
        bestMSE = self.run(bestP,bestQ,aFunction.getA(bestP,bestQ))[0]
        while cont:
            cont = False
            if bestP <= 1-stepSize:
                if cont == False:
                    up = self.run(bestP+stepSize,bestQ,aFunction.getA(bestP+stepSize,bestQ))[0]
                    if up<bestMSE:
                        bestP = bestP+stepSize
                        bestQ = bestQ
                        bestMSE = up
                        cont = True

            if bestP >= stepSize:
                if cont == False:
                    down = self.run(bestP-stepSize,bestQ,aFunction.getA(bestP-stepSize,bestQ))[0]
                    if down<bestMSE:
                        bestP = bestP-stepSize
                        bestQ = bestQ
                        bestMSE = down
                        cont = True

        cont = True
        while cont:
            cont = False
            if bestQ <= 1-stepSize:
                if cont == False:
                    right = self.run(bestP,bestQ+stepSize,aFunction.getA(bestP,bestQ+stepSize))[0]
                    if right<bestMSE:
                        bestP = bestP
                        bestQ = bestQ+stepSize
                        bestMSE = right
                        cont = True

            if bestQ >= stepSize:
                if cont == False:
                    left = self.run(bestP,bestQ-stepSize,aFunction.getA(bestP,bestQ-stepSize))[0]
                    if left<bestMSE:
                        bestP = bestP
                        bestQ = bestQ-stepSize
                        bestMSE = left
                        cont = True


        while cont:
            cont = False
            if bestP <= 1-stepSize:
                if cont == False:
                    up = self.run(bestP+stepSize,bestQ,aFunction.getA(bestP+stepSize,bestQ))[0]
                    if up<bestMSE:
                        bestP = bestP+stepSize
                        bestQ = bestQ
                        bestMSE = up
                        cont = True

            if bestP >= stepSize:
                if cont == False:
                    down = self.run(bestP-stepSize,bestQ,aFunction.getA(bestP-stepSize,bestQ))[0]
                    if down<bestMSE:
                        bestP = bestP-stepSize
                        bestQ = bestQ
                        bestMSE = down
                        cont = True

            if bestQ <= 1-stepSize:
                if cont == False:
                    right = self.run(bestP,bestQ+stepSize,aFunction.getA(bestP,bestQ+stepSize))[0]
                    if right<bestMSE:
                        bestP = bestP
                        bestQ = bestQ+stepSize
                        bestMSE = right
                        cont = True

            if bestQ >= stepSize:
                if cont == False:
                    left = self.run(bestP,bestQ-stepSize,aFunction.getA(bestP,bestQ-stepSize))[0]
                    if left<bestMSE:
                        bestP = bestP
                        bestQ = bestQ-stepSize
                        bestMSE = left
                        cont = True

        print(str(bestP) + " equals direct pct " + str(bestQ) + " equals indirect pct")
        return bestMSE,bestP,bestQ

    def makeTable(self, pStart, pEnd, pBreak, qStart, qEnd, qBreak, aStart, aEnd, aBreak, name):
        pSpace = pEnd - pStart
        qSpace = qEnd - qStart
        aSpace = aEnd - aStart
        workbook = xlsxwriter.Workbook(str(name))
        worksheet = workbook.add_worksheet()
        counter = 0

        worksheet.write(counter, 0, "P")
        worksheet.write(counter, 1, "Q")
        worksheet.write(counter, 2, "A")
        worksheet.write(counter, 3, "Empirical MSE")
        worksheet.write(counter, 4, "Estimate for Pi_x")
        counter += 1

        for i in range(0, pBreak + 1):
            for n in range(0, qBreak + 1):
                for a in range(0, aBreak + 1):

                    curP = pStart + i / (pBreak)*pSpace
                    curQ = qStart + n / (qBreak)*qSpace
                    curA = aStart + a / (aBreak)*aSpace

                    if curP != curQ and curP+curQ<=1 and curP >= 0 and curQ >= 0 and curA >=0 and curA <= 1 :
                        list = self.run(curP,curQ,curA)
                        curMSE, piEst = list[0],list[3]
                        worksheet.write(counter, 0, curP)
                        worksheet.write(counter, 1, curQ)
                        worksheet.write(counter, 2, curA)
                        worksheet.write(counter, 3, curMSE)
                        worksheet.write(counter, 4, piEst)
                        counter += 1

        workbook.close()

    def makeTablePrivacy(self, pStart, pEnd, pBreak, qStart, qEnd, qBreak, aStart, aEnd, aBreak, name):
        pSpace = pEnd - pStart
        qSpace = qEnd - qStart
        aSpace = aEnd - aStart
        workbook = xlsxwriter.Workbook(str(name))
        worksheet = workbook.add_worksheet()
        counter = 0

        worksheet.write(counter, 0, "P")
        worksheet.write(counter, 1, "Q")
        worksheet.write(counter, 2, "A")
        worksheet.write(counter, 3, "Empirical Privacy")
        worksheet.write(counter, 4, "Theorhetical Privacy")
        counter += 1

        for i in range(0, pBreak + 1):
            for n in range(0, qBreak + 1):
                for a in range(0, aBreak + 1):

                    curP = pStart + i / (pBreak)*pSpace
                    curQ = qStart + n / (qBreak)*qSpace
                    curA = aStart + a / (aBreak)*aSpace

                    if curP != curQ and curP+curQ<=1 and curP >= 0 and curQ >= 0 and curA >=0 and curA <= 1 :
                        pri = self.run(curP,curQ,curA)[4]
                        worksheet.write(counter, 0, curP)
                        worksheet.write(counter, 1, curQ)
                        worksheet.write(counter, 2, curA)
                        worksheet.write(counter, 3, pri)
                        worksheet.write(counter, 4, getTrueMSEwithPrivacy(self.n,self.tPi,self.zPi,curP,curQ,curA, True)[3])
                        counter += 1
                        print(counter)
        workbook.close()

def getTrueMSE(n,tPi,zPi,p,q,a,mode):
    if mode == False:
        phat = (tPi * (p - q) + q + (1 - p - q) * zPi)
        var = (phat * (1 - phat)) / ((n - 1) * ((p - q) ** 2))
        bias = tPi * (a - 1)
        MSE = var + bias**2
    else:
        phat = (tPi * a * (p - q) + q + (1 - p - q) * zPi)
        var = (phat * (1 - phat)) / ((n - 1) * a*a*((p - q) ** 2))
        bias = 0
        MSE = var + bias**2
    return MSE,bias,var

def getTrueMSEwithPrivacy(n,tPi,zPi,p,q,a,mode):
    if mode == False:
        phat1 = (tPi * (p - q) + q + (1 - p - q) * zPi)
        var = (phat1 * (1 - phat1)) / ((n - 1) * ((p - q) ** 2))
        bias = tPi * (a - 1)
        MSE = var + bias**2
        privacy1 = p*tPi+(1-p-q)*tPi*zPi
        privacy2 = q*tPi+(1-p-q)*(1-zPi)*tPi
        privacy1 = privacy1/phat1
        privacy2 = privacy2/(1-phat1)
    else:
        phat = (tPi * a * (p - q) + q + (1 - p - q) * zPi)
        var = (phat * (1 - phat)) / ((n - 1) * a*a*((p - q) ** 2))
        bias = 0
        MSE = var + bias**2
        privacy1 = q * tPi * (1 - a) + tPi * zPi * (1 - p - q) + a * p * tPi
        privacy2 = p*tPi*(1-a)+q*tPi*a+(1-p-q)*(1-zPi)*tPi
        privacy1 = privacy1 / phat
        privacy2 = privacy2 / (1 - phat)
    privacy = max(privacy1,privacy2)
    return MSE,bias,var, privacy, privacy

def getMostScrambledPQ(tPi,zPi):
    q = zPi/(1-tPi+(zPi/tPi))
    p = (q-q*tPi)/tPi
    print("p: ", p , ", q: ", q)
    print(tPi*p, " : ", q*(1-tPi), " : ", (1-p-q)*zPi)

tPi = .3
zPi = .2
p=.2
MM = mixedModel(500,10000,.2,.2,False)
MM.getMinMSE(100,.1,constantEstimator(1),False)
MM.getShape("MSE, Pix = .2, Piy = .2")
plt.show()


'''
pa = .9
pq = 0
p = .7
q = 0
std = stdEstimator(100)
circle = circleEstimator(500,100, 1/3,1/3,0,.1,"Pl")
circleN = circleApproximator(circle,"N", True)
MM = mixedModel(500,10000,circle.getA(p,q),.1,False)
a0 = circle.getA(p,q)
a1 = MM.run(pa,pq,1)[3]
a2 = circleN.getA(p,q)
MM1 = mixedModel(500,10000,.1,.1,True)
print("Regular RRT")
print(MM1.runEst(p,q,a0,a1))
MM2 = mixedModel(500,10000,.1,.1,True)
list = MM2.getMinPQ(100,1,circle,False)
p0, q0 = list[1], list[2]
print("Special RRT")
print(MM2.runEst(p0,q0,circle.getA(p0,q0),circle.getA(p0,q0)))
MM4 = mixedModel(500,10000,.1,.1,False)
print("Regular RRT no lie")
print(MM4.runEst(p,q,a0,a1))
MM3 = mixedModel(500,10000,.1,.1,False)
list = MM3.getMinPQ(100,1,std,False)
p0,q0 = list[1],list[2]
print("Special RRT no lie")
print(MM3.runEst(p0,q0,circle.getA(p0,q0),std.getA(p0,q0)))
circle.getShape()
circleN.getShape()
plt.show()
'''
