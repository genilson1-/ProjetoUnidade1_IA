# -*- coding: utf-8 -*-
##    Client of V-REP simulation server (remoteApi)
##    Copyright (C) 2015  Rafael Alceste Berri rafaelberri@gmail.com
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
##
##Habilite o server antes na simulação V-REP com o comando lua:
##simExtRemoteApiStart(portNumber) -- inicia servidor remoteAPI do V-REP


import vrep
import time

#definicoes iniciais
serverIP = '127.0.0.1'
serverPort = 19997
leftMotorHandle = 0
vLeft = 0.
rightMotorHandle = 0
vRight = 0.
sensorHandle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
thresould = 0.30


# variaveis de cena e movimentação do pioneer
noDetectionDist=0.4
maxDetectionDist=0.3
detect=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
braitenbergR=[-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
v0=2

clientID = vrep.simxStart(serverIP,serverPort,True,True,2000,5)
if clientID <> -1:
    print ('Servidor conectado!')

    # inicialização dos motores
    erro, leftMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
    if erro <> 0:
        print 'Handle do motor esquerdo nao encontrado!'
    else:
        print 'Conectado ao motor esquerdo!'

    erro, rightMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
    if erro <> 0:
        print 'Handle do motor direito nao encontrado!'
    else:
        print 'Conectado ao motor direito!'

    #inicialização dos sensores (remoteApi)
    for i in range(16):
        erro, sensorHandle[i] = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_ultrasonicSensor%d" % (i+1),vrep.simx_opmode_oneshot_wait)
        if erro <> 0:
            print "Handle do sensor Pioneer_p3dx_ultrasonicSensor%d nao encontrado!" % (i+1)
        else:
            print "Conectado ao sensor Pioneer_p3dx_ultrasonicSensor%d!" % (i+1)
            erro, state, coord, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, sensorHandle[i],vrep.simx_opmode_streaming)

    #desvio e velocidade do robo
    inputs = []
    outputs = []
    inicio = time.time()
    dados = ''
    contador = 0
    while vrep.simxGetConnectionId(clientID) != -1:
        distancias = []
            
        for i in range(16):
            erro, state, coord, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, sensorHandle[i],vrep.simx_opmode_buffer)

            if erro == 0:
                dist = coord[2]
                if state > 0 and dist < noDetectionDist:
                        
                    if dist < maxDetectionDist:
                        dist = maxDetectionDist

                    detect[i] = 1-((dist-maxDetectionDist) / (noDetectionDist-maxDetectionDist))
                else:
                    detect[i] = 0
            else:
                detect[i] = 0


            if ((i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 6) and state != 0):
                if(coord[2] <= 0.2 and coord[2] >= 0):
                    distancias.append(coord[2])
                elif (coord[2] > maxDetectionDist):
                    distancias.append(1)

        vLeft = v0
        vRight = v0

        for i in range(16):
            vLeft  = vLeft  + braitenbergL[i] * detect[i]
            vRight = vRight + braitenbergR[i] * detect[i]

        
        if( (time.time() - inicio) >= 0.2 ):
            inicio = time.time()
            inputs.append(distancias)
            # outputs.append([vLeft, vRight])
            dados += str(inputs[contador]).strip('[]').replace(',', '\t')
            dados += '\t'
            dados += str(vLeft/2)
            dados += '\t'
            dados += str(vRight/2)
            dados += '\n'
            # if(vLeft < 0):
            #     dados += str(0)
            #     dados += '\t'
            #     dados += '\t'
            # else:
            #     dados += str(vLeft/2)
            #     dados += '\t'
            #     dados += '\t'
                
            # if(vRight < 0):
            #     dados += str(0)
            #     dados += '\t'
            #     dados += '\t'
            # else:
            #     dados += str(vRight/2)                      
            contador += 1
            # print(dados)
        # atualiza velocidades dos motores
        erro = vrep.simxSetJointTargetVelocity(clientID, leftMotorHandle, vLeft/1.4, vrep.simx_opmode_streaming)
        erro = vrep.simxSetJointTargetVelocity(clientID, rightMotorHandle, vRight/1.4, vrep.simx_opmode_streaming)
            
    vrep.simxFinish(clientID) # fechando conexao com o servidor
    # print 'Conexao fechada!'
    arquivo = open('/home/jgfilho/Documentos/UFRN/2017.2/Inteligência Artificial/vrep_robo/Python/data_set.txt', 'w')
    arquivo.write(dados)
    arquivo.close()

else:
    print 'Problemas para conectar o servidor!'
