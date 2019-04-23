import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tqdm
import math
from scipy.stats import gamma

def gera_meshgrid(lim_l,lim_h, p):
    ''' Gera um meshgrid a partir dos limites recebidos e 
    da resolução (p).
    '''
    X = np.arange(-lim_l, lim_h, p)
    Y = np.arange(-lim_l, lim_h, p)
    X, Y = np.meshgrid(X, Y) # Cria grid para aplicarmos a função
    return X, Y


def calcula_function(x, y, function):
    '''Recebe duas arrays grid e aplica a função 
    nas coordenadas correspondentes.
    '''
    z = np.zeros([len(x),len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = function(x[i][j],y[i][j])
    return z

def plot_function(lim_l, lim_h, p, function):
    '''Recebe os limites de domínio, o passo de calculo desejado 
    e uma função. Plota a superfície calculada em 3 dimensões.
    '''
    X, Y = gera_meshgrid(lim_l, lim_h, p) # Cria grid para aplicarmos a função
    
    Z = calcula_function(X,Y, function)

    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection='3d')
    ax.mouse_init(rotate_btn=0, zoom_btn=3)
    ax.contour3D(X,Y,Z,200) #200
    
    #ax.scatter(419.9923236, 420.94817857, 1500, c='r', marker='o')
    
    return X, Y, Z

def schwefel(x, y):
    '''Implementa a função de schwefel em duas
    dimensões.
    '''
    t = lambda delta : delta*math.sin(math.sqrt(abs(delta)))
    
    return 418.9829*2-sum([t(x), t(y)]) # Para n dimensões pode ser usado 418.9829*len(x)-sum([t(i) for i in x])

#X, Y, Z = plot_function(500, 500, 1, schwefel)

# Definição da função Rastrigin
rastrigin = lambda x, y : 20 + x**2 + y**2 -10*(math.cos(math.pi*2*x) + math.cos(math.pi*2*y))
#X, Y, Z = plot_function(5, 5, 0.1, rastrigin)

pike = lambda x,y : x*math.exp(-(x**2+y**2))
#X, Y, Z = plot_function(2, 2, 0.1, pike)

# Geração da população e cálculo do fitness

def individuo(tam_individuo, minimo, maximo):
    return np.array(np.random.rand(tam_individuo)*np.random.choice([minimo, maximo]))

def gerar_pop(tam_populacao=10, n=2, lim_l=-500, lim_h=500):
    '''Gera uma população com indivíduos de n dimensões com os valores de lim_l à lim_h gerados aleatoriamente.'''
    return np.array([individuo(n, lim_l, lim_h) for i in range(tam_populacao)])

def cal_fitness(ido, funcao):
    # Retorna o cálculo da função para um vetor de duas dimensões.
    return funcao(ido[0],ido[1])

def seleciona_torneio(pop_fit, maxi = 1):
    '''Recebe uma array com o fitness da população que se deseja selecionar e retorna os índices dos intens escolhidos
    de acordo com o metodo de seleção torneio.'''
    
    tamanho = int(pop_fit.shape[0]) # Guarda o número de indivíduos
    
    vencedores = list()
    
    for i in range(tamanho):
        # Sorteia número de competidores.
        k = 2 #np.random.randint(2, tamanho)
        
        # Escolhe os indivíduos aleatoriamente para o torneio.
        competidores = np.random.choice(range(tamanho), k, replace=False)
        
        # Verifica vencedor
        if maxi:
            vencedores.append(competidores[pop_fit[competidores].argmax()])
        else:
            vencedores.append(competidores[pop_fit[competidores].argmin()])
    return vencedores

def seleciona_roleta(pop_fit, maxi = 0):
    '''Recebe uma array com o fitness da população que se deseja selecionar e retorna os índices dos intens escolhidos
    de acordo com o metodo de seleção roleta.'''
    
    tamanho = int(pop_fit.shape[0]) # Verifica o número de indivíduos

    # Verifica se o menor valor é negativo
    if pop_fit.min() < 0:
        pop_fit = pop_fit + abs(pop_fit.min())
        
    soma_fitness = sum(pop_fit)
    
    if(maxi == 1):
        fitness_relativo = pop_fit/soma_fitness
    else:
        fitness_relativo = (pop_fit - soma_fitness)/sum(pop_fit - soma_fitness)
    
    if True in np.isnan(fitness_relativo):
        return range(tamanho)
        
    return np.random.choice(range(tamanho), tamanho, p=fitness_relativo)

def crossover_ponto(individuos, pc):
    '''Recebe uma array com os indivíduos que devem ser 
    permutados e uma probabilidade de crossover (pc) e os permuta aleatóriamente de acordo com a probabilidade 
    passada como argumento.'''
    
    tamanho = individuos.shape[0]
    
    # Gera uma lista com a permutação dos índices dos indivíduos
    permuta = np.random.choice(range(tamanho), tamanho, replace=False)
        
    for i in range(0,(tamanho-1),2):
        if np.random.rand() < pc:
            intermed = individuos[permuta[i]][0] 
            individuos[permuta[i]][0] = individuos[permuta[i+1]][0]
            individuos[permuta[i+1]][0] = intermed

    return individuos

def crossover_aritimetico(individuos, pc):
    '''Recebe uma array com os indivíduos que devem ser permutados e 
    uma probabilidade de crossover (pc) e aplica o crossover aritimético
    aleatóriamente de acordo com a probabilidade passada como argumento.'''
    
    tamanho = individuos.shape[0]
    
    # Gera uma lista com a permutação dos índices dos indivíduos
    permuta = np.random.choice(range(tamanho), tamanho, replace=False)
        
    for i in range(0,(tamanho-1),2):
        if np.random.rand() < pc:
            alfa = np.random.rand()
            intermed = individuos[permuta[i]][0] 
            individuos[permuta[i]][0] = (1-alfa)*intermed 
            individuos[permuta[i]][1] = alfa*individuos[permuta[i+1]][1]
            individuos[permuta[i+1]][0] = alfa*intermed
            individuos[permuta[i+1]][1] = (1-alfa)*individuos[permuta[i+1]][1]
            
    return individuos    
    
# Seleção e cruzamento
def seleciona_e_cruza(populacao, funcao, selecao, crossover, maxi=0):
    
    tamanho = populacao.shape[0] # Identifica o tamanho da população.
    
    # Calcula o fitness da populacao.
    pop_fit = np.array([cal_fitness(populacao[i], funcao) for i in range(tamanho)])
    
    # Encontra o melhor fitness
    if maxi:
        best_fit = pop_fit.max()
        best_individuo = populacao[pop_fit.argmax()]
    else:
        best_fit = pop_fit.min()
        best_individuo = populacao[pop_fit.argmin()]
        
    # Seleção.
    selecionados = selecao(pop_fit, maxi)
    
    individuos_crosados = crossover(populacao[selecionados], 0.7) # Faz o crossover entre os indivíduos selecionados.
    
    #print(np.unique(individuos_crosados).shape)
    
    return individuos_crosados, best_fit, best_individuo

def muta_pop(pop, lim_l=-500, lim_h=500, pc = 0.1):
    # Mutação - Segundo DEJONG (1975), a probabilidade de ocorrência de mutação
    # deve ser igual ao inverso do tamanho da população.
    
    tamanho, n = pop.shape
    
    for i in range(int(pop.shape[0])):
        if np.random.rand() < pc:#float(1.0/tamanho):
            pop[i][np.random.choice([0,1])] = individuo(n, lim_l, lim_h)[0]
            
    return pop

def otimiza_funcao(pop, funcao, selecao, crossover, loops, maxi = 0):
    tamanho = pop.shape[0]
    melhor_individuo = np.array([9999,9999])
    melhor_valor = 9999 if maxi == 0 else 0
    for i in range(loops):
        pop, valor, individuo_atual  = seleciona_e_cruza(pop, funcao, selecao, crossover, maxi)
        if(maxi == 0):
            if(valor < melhor_valor):
                melhor_valor = valor
                melhor_individuo = individuo_atual
                print("Melhor valor encontrado: {0}\nMelhor Indivíduo: {1} Looping: {2}".format(melhor_valor, melhor_individuo, i))
        else:
            if(valor > melhor_valor):
                melhor_valor = valor
                melhor_individuo = individuo_atual
                print(melhor_valor, melhor_individuo, i)
        #pop = muta_pop(pop)
        if(np.unique(pop).shape[0] <= 2):
            print("A população convergiu para {}".format(pop[0]))
            break
    return pop

print("Otimizando a função Schwefel...")
populacao = gerar_pop(tam_populacao=1000, n=2, lim_l=-500, lim_h=500)
populacao = otimiza_funcao(populacao, schwefel, seleciona_torneio, crossover_ponto, 200, 1)

print("Otimizando a função Rastrigin...")
populacao = gerar_pop(tam_populacao=100, n=2, lim_l=-5, lim_h=5)
populacao = otimiza_funcao(populacao, rastrigin, seleciona_torneio, crossover_aritimetico, 200, 0)

print("Otimizando a outra função...")
populacao = gerar_pop(tam_populacao=100, n=2, lim_l=-2, lim_h=2)
populacao = otimiza_funcao(populacao, pike, seleciona_torneio, crossover_aritimetico, 200, 0)