"""
CES - 30 Lab 2 - Árvore de decisão e Redes baysianas

author: Dylan Nakandakari Sugimoto & Gabriel Adriano de Melo
Data: 24/09/2018
"""
import numpy as np


class BayesIngenuo:
    
    def __init__(self, entradaTreinamento, saidaTreinamento):
        """
        Recebe uma matriz com cada linha sendo um caso de treinamento e um vetor com as respostas.
        Os valores devem ser numéricos crescentes a partir de zero, pois os uso para indexar a matrix
        """
        if (len(entradaTreinamento) != len(saidaTreinamento)):
            raise Exception("Quantidade de casos de entrada devem ser iguais ao de saída")
        self.entradaTreinamento = entradaTreinamento
        self.saidaTreinamento = saidaTreinamento
        self.classesSaida = np.unique(saidaTreinamento)
        self.classesAtributos = np.unique(entradaTreinamento)
        self.quantAtributos = len(entradaTreinamento[0])
        self.treinar(entradaTreinamento, saidaTreinamento)
    
    def treinar(self, entrada, saida):
        quantCasos = len(saida)
        probs = np.zeros((self.quantAtributos, len(self.classesAtributos), len(self.classesSaida)))
        
        for atributo in range(self.quantAtributos):
            i=0
            for caso in entrada:
                probs[atributo][caso[atributo]][saida[i]] += 1
                i+=1
            probs[atributo] = probs[atributo] / probs[atributo].sum(axis=0)
        
        self.probs = probs
        
        estados = np.zeros((len(self.classesSaida)))
        for s in self.classesSaida:
            estados[s] = (saida == s).sum()
            estados[s] /= quantCasos
            
        self.estados = estados
        
    def predizer(self, entrada):
        quantCasos = len(entrada)
        predito = np.zeros((quantCasos))
        probsPred = np.zeros((len(self.classesSaida)))
        for c in range(quantCasos):
            for s in range(len(self.classesSaida)):
                probsPred[s] = self.estados[s]
                for a in range(self.quantAtributos):
                    probsPred[s] *= self.probs[a][entrada[c][a]][s]
            predito[c] = probsPred.argmax()
        return predito


import csv
import random

print('Conjunto de Treinamento, Taxa de Acerto, Erro Quadradático Médio, Kappa, A priori')


random.seed(19)

for tam_treino in [k*250 for k in range(21, 31)]:
    
    # Carregar arquivo
    with open('connect-4.data', 'r') as arquivo:
        jogos = list(csv.reader(arquivo, delimiter=','))
    tam_validacao = len(jogos) - tam_treino
    #tam_validacao = 10000
    random.shuffle(jogos)
    
    valor = {'win':2, 'x':2, 'draw':1, 'b':1, 'o':0, 'loss':0}
    entrada = np.array([[valor[j] for j in jogo[:-1]] for jogo in jogos])
    esperado = np.array([valor[jogo[-1]] for jogo in jogos])
    
    # Fazer predição
    preditor = BayesIngenuo(entrada[:-tam_validacao], esperado[:-tam_validacao])
    predicoes = preditor.predizer(entrada[-tam_validacao:])
    confusao = np.zeros((4, 4), dtype='uint')
    for i in range(tam_validacao):
        confusao[esperado[-i]][int(predicoes[-i])] += 1
    confusao[3] = np.sum(confusao, axis=0)
    confusao[:, -1] = np.sum(confusao, axis=1)
    #print(confusao)
    
    #confusao[:-1, [3]].dot(confusao[[3], :-1])/confusao[3][3]
    
    pe = 0
    for i in range(3):
        pe += confusao[i][3]*confusao[3][i]/confusao[3][3]/confusao[3][3]
    p0 = 0
    for i in range(3):
        p0 += confusao[i][i]/confusao[3][3]
    kappa = (p0-pe)/(1-pe)
    
    custos = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    custos = np.multiply(custos, custos)
    erros = np.multiply(custos, confusao[:-1, :-1])
    erroQuadraticoMedio = np.sum(erros)/confusao[3][3]
    
    print(tam_treino, p0, erroQuadraticoMedio, kappa, confusao[2][3]/confusao[3][3], sep=', ')
    
    #print('Kappa:', kappa, '\nTaxa de Acerto:', p0, '\nA priori:', confusao[2][3]/confusao[3][3], pe)
    
    #print("Procentagem de erros %f" % (100/(entrada.shape[0]-tam_validacao)*(esperado[-tam_validacao:] != predicoes).sum()))
