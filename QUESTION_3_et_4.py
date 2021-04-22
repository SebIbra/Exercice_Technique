# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from scipy.stats import norm
from numpy import sqrt, array
import warnings
from multiprocessing import Pool, cpu_count
from time import process_time
from numpy.random import normal, lognormal


def time(func):
    '''
    Décorateur qui calcule le temps d'une fonction
    '''
    def decorateur(*args, **kwargs):
        t1 = process_time()
        f = func(*args, **kwargs)
        t2 = process_time()
        print(f'La fonction {func.__name__} a pris {t2-t1} secondes à tourner')
        return f
    return decorateur


class Loss_model:
    '''
    Crée une classe de modèle de perte
    
    Paramètres
    ----------
    rho : réel entre 0 et 1
    matrixM : matrice de probabilités, np.array((size,size))
    matrixC : matrice de probabilités cumulées, np.array((size,size))
    matrixM ou matrixC doit être renseigné, si les 2 sont renseignées on prend par défaut matrixM
    '''
    def __init__(self, rho = 0.5, matrixM = None, matrixC = None):
        self.rho = rho ### Paramètre rho

        ### Construction de matrixC et de matrixM
        if matrixM is None and matrixC is None :
            raise ValueError('Il faut renseigner matrixM ou matrixC')
        elif matrixM is not None :
            if matrixC is not None :
                warnings.warn('Attention matrixM et matrixC sont renseignées, par défaut seul matrixM sera compté',
                              Warning)
            self.matrixM = matrixM
            self.matrixC = np.cumsum(matrixM,axis = 1)  ### Somme cumulée sur les colonnes de matrixM
        elif matrixC is not None :
            self.matrixC = matrixC
            self.matrixM = matrixC.copy()
            self.matrixM[:,1:] -= self.matrixM[:,:-1].copy() ### On fait la différence entre 2 colonnes successives de matrixC

    ######### Question 3
    def simulate_note(self, n0=0, X=1):
        '''
        Simule la note d'une entreprise à t = 0 dans 1 an

        Paramètres
        ----------
        n0 : entiers note à t = 0
        X : réel, état de l'économie
        '''
        rho = self.rho
        matrixC = self.matrixC
        Z = sqrt(rho)*X + sqrt(1-rho)*normal() ## Simulation de Z
        phi_Z = 1 - norm.cdf(Z) ##Simulation de 1 - phi(Z)
        return np.where(phi_Z<=matrixC[n0])[0][0]  ## On récupère le premier indice de la ligne n0 qui vérifie phi_Z <= matrixC[n0]

    def simulate_notes(self,notes,X=1) :
        '''
        Simule la note de n entreprises sachant leurs notes (n = taille du vecteur notes)
        Fonction vectorisée, pratique pour accélérer les calculs

        Paramètres
        ----------
        notes : liste d'entiers de notes à t = 0
        X : réel, état de l'économie
        '''
        rho = self.rho
        matrixC = self.matrixC
        Zs = sqrt(rho)*X + sqrt(1-rho)*normal(size = len(notes))  ## Simulation des Z
        phi_Zs = 1 - norm.cdf(Zs) ## Simulation des phi_Z
        phi_Zs = phi_Zs.reshape((len(notes),1))
        return np.argmax(phi_Zs<=matrixC[notes], axis = 1) ###Retourne les notes à 1 an ssociées

    @time
    def check_question2(self, n0=0, n1an=1, X=0, N=10000):
        '''
        Calcule la probabilité de passer d'une note n0 à n1an à un an, conditionnellement à X

        Paramètres
        ----------
        n0, n1an : entiers note à t = 0 et à t = 1
        X : réel, état de l'économie
        N : entier, nombre de simulations pour calculer la valeur empirique de la probabilité
        '''
        matrixCi = self.matrixC[n0]
        rho = self.rho
        ### Calcul théorique de la probabilité via les formules de la qustion 2
        theorique_k = norm.cdf(1/sqrt(1-rho)*(norm.ppf(matrixCi[n1an]) + sqrt(rho)*X))
        theorique_k1 = 0 if n1an == 0 else norm.cdf(1/sqrt(1-rho)*(norm.ppf(matrixCi[n1an-1]) + sqrt(rho)*X))
        theorique = theorique_k - theorique_k1

        ### Calcul empirique de la probabilité de la question 2 via des simulations
        empirique = np.mean([self.simulate_notes([n0]*N,X)==n1an])

        return {'n0':n0, 'n1an':n1an,
                'valeur théorique':theorique, 'valeur empirique':empirique}

    ######### Question 4

    ## Calcul de une perte
    def one_perte(self,nb_companies=1000,notes=None,E=None,X=None,mu=0,sigma=1):
        '''
        Calcule la perte à 1 an sachant le nombre de compagnies,
        leurs notes et leurs expositions à t=0

        Paramètres
        ----------
        nb_companies : entier, nombre de compagnies
        notes : vecteur de taille nb_companies de notes (entiers). S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation

        E :     vecteur de taille nb_companies d'expositions (réels). S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation (loi lognormale LN(mu,sigma))

        X :     réel, valeur de l'état économique. S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation (loi normale N(mu,sigma))


        mu,sigma : réels, paramètres de la LN(mu,sigma)
        '''
        if notes is not None :
            assert len(notes)==nb_companies, "Le nombre de notes n'est pas égal au nombre d'entreprises"
        else :
            notes = np.random.randint(0,3,nb_companies) # Si les notes ne sont pas renseignées : random

        if E is not None :
            assert len(E)==nb_companies, "Le nombre d'expositions n'est pas égal au nombre d'entreprises"
        else :
            E = lognormal(mu,sigma,nb_companies) # Si les expositions ne sont pas renseignées : random

        if X is None :
            X = normal() # Variable d'état économique

        defaults = array(self.simulate_notes(notes,X)) == 2  ### On prend en compte uniquement les entreprises en défaut
        perte = np.dot(defaults,E) ### On fait la somme des pertes correspondant aux entreprises en défaut
        return perte

    ## Calcul des pertes totales
    @time
    def calcul_pertes(self,nb_simulations=10000,nb_companies=1000,notes=None,E=None,X=None,
                     isParallel=False,nbCores=4,
                     mu=0,sigma=1):
        '''
        Calcule la perte à 1 an sachant le nombre de compagnies,
        leurs notes et leurs expositions à t=0

        Paramètres
        ----------
        nb_simulations : entier, nombre de simulations
        nb_companies : entier, nombre de compagnies

        notes : vecteur de taille nb_companies de notes (entiers). S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation

        E :     vecteur de taille nb_companies d'expositions (réels). S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation (loi lognormale LN(mu,sigma))

        X :     réel, valeur de l'état économique. S'il n'est pas renseigné,
                il sera considéré comme aléatoire pour chaque simulation (loi normale N(mu,sigma))

        isParallel : bool, calcul en parallèle?

        nbCores : entier, nombre de coeurs utilisés pour le calcul en parallèle

        mu,sigma : réels, paramètres de la LN(mu,sigma)

        '''
        params=nb_companies,notes,E,X,mu,sigma
        if not isParallel :
            pertes = [self.one_perte(*params) for _ in range(nb_simulations)]

        if isParallel :
            cpuCount = cpu_count()
            if nbCores > cpuCount :
                warnings.warn(f'Attention, nbCores > nombre de coeurs disponibles ({cpuCount})',
                         Warning)
                nbCores = cpuCount
            pool = Pool(nbCores) ## On ouvre des pools

            numberOfSimulationsByCore = nb_simulations//nbCores+1 ## Nombre de simulations par coeur
            new_params = [numberOfSimulationsByCore,self.one_perte] + list(params) ## Paramètres pour fonction externe de parallélisation
            pertes = pool.map(calcul_parallele,[new_params for _ in range(nbCores)]) ## On partage le calcul sur les coeurs en appelant la fonction calcul parallèle


        return pertes


def calcul_parallele(new_params) :
    '''
    Paramètres
    ----------
    Fonction qui renvoie n pertes (utile pour faire de la parallélisation)
    new_params : liste de paramètres
    new_params[0] : nombre de simulations pour le coeur
    new_params[1] : nom de la fonction à appliquer
    new_params[2:] : paramètres de la fonction
    '''
    numberOfSimulationsByCore = new_params[0]
    function = new_params[1]
    other_params = new_params[2:]
    return [function(*other_params) for _ in range(numberOfSimulationsByCore)]


def statistics(liste,alpha):
    '''
    Fonction qui renvoie quelques statistiques sur une liste L

    Paramètres
    ----------
    L : liste de pertes
    alpha : réel entre 0 et 1, valeur pour laquelle on veut calculer le quantile
    '''
    return {'min' : np.min(liste),
            'max' : np.max(liste),
            'moyenne' : np.mean(liste),
            'std' : np.std(liste),
            'médiane' : np.percentile(liste,50),
            f'quantile a {alpha}' : np.percentile(liste, alpha*100)
            }





if __name__ == '__main__' :
    matrixM = array([[0.8,0.1,0.1],[0.2,0.75,0.05],[0,0,1]])
    matrixC = np.cumsum(matrixM, axis = 1)
    rho = 0.5
    print({'M' : matrixM})
    print({'C' : matrixC})
    print({'rho' : rho})

    ### Construction du modèle
    model = Loss_model(rho=rho, matrixM=matrixM)

    ### Vérification
    n0 = 0
    n1an = 1
    X = normal()
    print('###################')
    print('On vérifie que le calcul de la question 2 fonctionne pour la proba P(n1an = 1 |n0 = 0)')
    print(model.check_question2(n0,n1an,N=1000000))
    print('###################\n')


    print('###################')
    print('Pour le calcul des pertes, on suppose que X, les notes à t = 0 et les expositions sont aléatoires\npour chaque simulation\n')
    nb_simulations = 10_000
    nb_companies=1000
    alpha = 0.65

    print('Sans parallélisation')
    pertes = model.calcul_pertes(nb_simulations = nb_simulations, nb_companies = nb_companies, isParallel = False)
    print(statistics(pertes,alpha),end = '\n')
    print('Avec parallélisation')
    pertes = model.calcul_pertes(nb_simulations = nb_simulations, nb_companies = nb_companies, isParallel = True, nbCores=4)
    print(statistics(pertes,alpha),end = '\n')
    print('###################\n')


    print('###################')
    print('Pour le calcul des pertes, on suppose que X, les notes à t = 0 et les expositions sont fixées une bonne fois pour toute')
    nb_simulations = 10_000
    nb_companies=1000
    alpha = 0.65
    X = normal()
    notes = np.random.randint(3, size = nb_companies)
    E = lognormal(size = nb_companies)

    print(f'Le X utilisé est {X}')


    print('Sans parallélisation')
    pertes = model.calcul_pertes(nb_simulations=nb_simulations,nb_companies=nb_companies,notes=notes,
                                 E=E,X=X,isParallel=False,nbCores=4)
    print(statistics(pertes,alpha),end = '\n')
    print('Avec parallélisation')
    pertes = model.calcul_pertes(nb_simulations=nb_simulations,nb_companies=nb_companies,notes=notes,
                                 E=E,X=X,isParallel=True,nbCores=4)
    print(statistics(pertes,alpha),end = '\n')
    print('###################\n')

