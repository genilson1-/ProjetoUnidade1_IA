# -*- coding: utf-8 -*-
import math
import random
import numpy
import os
import csv
import vrep
import matplotlib.pyplot as plt

random.seed(0)
def criar_linha():
    print ("-"*80)

# gera numeros aleatorios obedecendo a regra:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a

# nossa funcao de ativação sigmoide - gera graficos em forma de S
def funcao_ativacao_tang_hip(x):
    return math.tanh(x)

# derivada da função de ativação
def derivada_funcao_ativacao(x): 
    # t = funcao_ativacao_tang_hip(x)
    return 1.0 - x**2

class RedeNeural:
    #
    def __init__(self, nos_entrada, nos_ocultos, nos_saida):
        # camada de entrada
        self.nos_entrada = nos_entrada + 1 # +1 por causa do no do bias
        # camada oculta
        self.nos_ocultos = nos_ocultos
        # camada de saida
        self.nos_saida = nos_saida
        # quantidade maxima de interacoes
        self.max_interacoes = 40000
        # taxa de aprendizado
        self.taxa_aprendizado = 0.085
        self.momentum = 0.0001
        self.erro_minumum = 0.00001

        # activations for nodes 
        # cria uma matriz, preenchida com uns, de uma linha pela quantidade de nos
        self.ativacao_entrada = numpy.ones(self.nos_entrada)
        self.ativacao_ocultos = numpy.ones(self.nos_ocultos)
        self.ativacao_saida = numpy.ones(self.nos_saida)
        
        # contém os resultados das ativações de saída
        self.resultados_ativacao_saida = numpy.ones(self.nos_saida)
 
        # criar a matriz de pesos, preenchidas com zeros
        self.wi = numpy.zeros((self.nos_entrada, self.nos_ocultos))
        self.wo = numpy.zeros((self.nos_ocultos, self.nos_saida))
		
        # adicionar os valores dos pesos
        # vetor de pesos da camada de entrada - intermediaria
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                self.wi[i][j] = rand(-0.2, 0.2)

        # vetor de pesos da camada intermediaria - saida
        for j in range(self.nos_ocultos):
            for k in range(self.nos_saida):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = numpy.zeros((self.nos_entrada, self.nos_ocultos))
        self.co = numpy.zeros((self.nos_ocultos, self.nos_saida))
        
    def fase_forward(self, entradas):
        # input activations: -1 por causa do bias
        for i in range(self.nos_entrada - 1):           
            self.ativacao_entrada[i] = entradas[i]

        # calcula as ativacoes dos neuronios da camada escondida
        for j in range(self.nos_ocultos):
            soma = 0
            for i in range(self.nos_entrada):
                soma = soma + self.ativacao_entrada[i] * self.wi[i][j]
            self.ativacao_ocultos[j] = funcao_ativacao_tang_hip(soma)

        # calcula as ativacoes dos neuronios da camada de saida
        # Note que as saidas dos neuronios da camada oculta fazem o papel de entrada 
        # para os neuronios da camada de saida.
        for j in range(self.nos_saida):
            soma = 0
            for i in range(self.nos_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.wo[i][j]
            self.ativacao_saida[j] = funcao_ativacao_tang_hip(soma)
   
        return (self.ativacao_saida, self.wi, self.wo)    

    def fase_backward(self, saidas_desejadas):
        # calcular os gradientes locais dos neuronios da camada de saida
        output_deltas = numpy.zeros(self.nos_saida)
        erro = 0
        for i in range(self.nos_saida):
            erro = saidas_desejadas[i] - self.ativacao_saida[i]
            output_deltas[i] = derivada_funcao_ativacao(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neuronios da camada escondida
        hidden_deltas = numpy.zeros(self.nos_ocultos)
        for i in range(self.nos_ocultos):
            erro = 0
            for j in range(self.nos_saida):
                erro = erro + output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = derivada_funcao_ativacao(self.ativacao_ocultos[i]) * erro

        # a partir da ultima camada ate a camada de entrada
        # os nos da camada atual ajustam seus pesos de forma a reduzir seus erros
        for i in range(self.nos_ocultos):
            for j in range(self.nos_saida):
                change = output_deltas[j] * self.ativacao_ocultos[i]
                self.wo[i][j] = self.wo[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.co[i][j])
                self.co[i][j] = change

        # atualizar os pesos da primeira camada
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                change = hidden_deltas[j] * self.ativacao_entrada[i]
                self.wi[i][j] = self.wi[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.ci[i][j])
                self.ci[i][j] = change

        # calcula erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro = erro + 0.5 * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):
        velocidades = []
        velocidades, wi, wo = self.fase_forward(entradas_saidas)
        return (velocidades[0], velocidades[1])

    def treinar(self, entradas_saidas):
        plot_error = []
        epoch = []
        last_error = 0
        aux = []
        for i in range(self.max_interacoes):
            erro = 0
            for p in entradas_saidas:
                entradas = p[0]
                saidas_desejadas = p[1]
                self.fase_forward(entradas)[0]
                erro = erro + self.fase_backward(saidas_desejadas)
            if (i % 300 == 0 or last_error < erro):
                plot_error.append(erro)
                epoch.append(i)
                print ("Erro = %2.7f epoch = %s" % (erro, i))        
            last_error = erro
            aux = [erro, i]
        print ("Erro = %2.7f epoch = %s" % (aux[0], aux[1]+1))
        
        low = 0.00000001
        high = 0.001
        # plt.ylim(low, high)
        plt.semilogy(epoch, plot_error)
        plt.ylabel('error')
        plt.xlabel('epoch')
        self.test([0.20,0.20,1,1,1,1])
        plt.show()
    
def wanderr():
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
    
    # cria rede neural com duas entradas, duas ocultas e um no de saida    
    n = RedeNeural(6, 10, 2)
    # treinar com os padrões
    n.treinar(new_data)
    # testar
    # criar_linha()
    vl, vr = n.test([0.20,0.20,1,1,1,1]) 
    
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
    wanderr()
