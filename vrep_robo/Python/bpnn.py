# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import math
import random
import string
import csv
import numpy as np

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        velocidades = []
        # for p in patterns:
        velocidades = self.update(patterns)
        return (velocidades[0], velocidades[1])
        

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=120000, N=0.085, M=0.0001): #, error_thresould
        # N: learning rate 
        # M: momentum factor
        plot_error = []
        epoch = []
        last_error = 0
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if (i % 1000 == 0 or last_error < error):
                plot_error.append(error)
                epoch.append(i)
                print('error %-.10f' % error)
            last_error = error
        low = 0.00000001
        high = 0.001
        plt.ylim(0.000001, 0.01)
        plt.plot(epoch, plot_error)
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()




def wanderr():
    import vrep
    #definicoes iniciais
    serverIP = '127.0.0.1'
    serverPort = 19997
    leftMotorHandle = 0
    vLeft = 0.
    rightMotorHandle = 0
    vRight = 0.       
    sensorHandle = [0,0,0,0,0,0]
    thresould = 0.25
    noDetectionDist=0.4
    maxDetectionDist=0.3


    new_data = [
    [[0.20,1,1,1,1,1], [1,0.5]],
    [[0.20,0.20,1,1,1,1], [1,0.5]],
    [[0.10,0.10,1,1,1,1], [1,0]],
    [[1,1,1,1,1,1], [1,1]],
    [[1,1,0.10,0.10,1,1], [0,0.75]],
    [[1,1,0.20,0.20,1,1], [0.6,1]],
    [[1,1,1,1,0.20,0.20], [0.5,1]],
    [[1,1,1,1,1,0.20], [0.5,1]],
    [[1,1,1,1,0.10,0.10], [0,1]],
    [[1,1,1,0.20,1,1], [0,1]],
    [[1,1,0.20,1,1,1], [0,1]],
    [[1,1,1,0.10,0.10,1], [0,0.5]]

    ]
    
    data = csv.reader(open('/home/jgfilho/Documentos/UFRN/2017.2/Inteligência Artificial/vrep_robo/Python/data_set.csv', 'r'))
    entrada = []
    entradaFinal = []
    for rows in data:
        x = rows
        x = list(map(float, x))
        entrada.extend([x[:4], x[4:]])
        entradaFinal.append(entrada)
        entrada = []


    n = NN(6, 10, 2)
    n.train(new_data)


    clientID = vrep.simxStart(serverIP,serverPort,True,True,2000,5)
    if (clientID != -1):
        print ('Servidor conectado!')

        # inicialização dos motores
        erro, leftMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
        if erro != 0:
            print ('Handle do motor esquerdo nao encontrado!')
        else:
            print ('Conectado ao motor esquerdo!')

        erro, rightMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
        if erro != 0:
            print ('Handle do motor direito nao encontrado!')
        else:
            print ('Conectado ao motor direito!')

        #inicialização dos sensores (remoteApi)
        for i in range(6):
            erro, sensorHandle[i] = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_ultrasonicSensor%d" % (i+1),vrep.simx_opmode_oneshot_wait)
            if erro != 0:
                print ("Handle do sensor Pioneer_p3dx_ultrasonicSensor%d nao encontrado!" % (i+1))
            else:
                print ("Conectado ao sensor Pioneer_p3dx_ultrasonicSensor%d!" % (i+1))
                erro, state, coord, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, sensorHandle[i],vrep.simx_opmode_streaming)

        #desvio e velocidade do robo
        inputs = []
        outputs = []
        dados = ''
        contador = 0
        while vrep.simxGetConnectionId(clientID) != -1:
            distancias = []
            for i in range(6):
                erro, state, coord, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, sensorHandle[i],vrep.simx_opmode_buffer)
                if(state != 0):
                    distancias.append(coord[2])
                else:
                    distancias.append(1)
              
            
            vl, vr = n.test(distancias)
            vl -= 0.5
            vr -= 0.5
            vl *= 4
            vr *= 4
            print('vl é ~> %s e vr é ~> %s' % (vl, vr))
            # print('vleft -> %s vright -> %s ' % (vl, vr))
            # atualiza velocidades dos motores
            erro = vrep.simxSetJointTargetVelocity(clientID, leftMotorHandle, vl, vrep.simx_opmode_streaming)
            erro = vrep.simxSetJointTargetVelocity(clientID, rightMotorHandle, vr, vrep.simx_opmode_streaming)


if __name__ == '__main__':
    #grafico log
    wanderr()