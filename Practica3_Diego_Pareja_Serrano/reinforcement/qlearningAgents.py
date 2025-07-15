# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Métodos que debes implementar:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

    Variables:
        - self.epsilon (probabilidad de exploración)
        - self.alpha (tasa de aprendizaje)
        - self.discount (factor de descuento)
    
    Métodos útiles:
        - self.getLegalActions(state): devuelve acciones legales para un estado
    """

    def __init__(self, **args):
        # Inicializamos el agente Q-learning heredando de ReinforcementAgent
        ReinforcementAgent.__init__(self, **args)

        # Diccionario para almacenar Q-values: Q(s,a)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
        Devuelve Q(s,a). Si nunca se ha visto antes, devuelve 0.0 por defecto.
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Devuelve V(s) = max_a Q(s,a).
        Si no hay acciones legales (estado terminal), devuelve 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0

        # Calcula el máximo Q-value entre las acciones legales
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
        Devuelve la acción óptima desde un estado (argmax_a Q(s,a)).
        Si hay empate entre acciones, elige una aleatoriamente entre las mejores.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Calculamos el máximo Q-value del estado
        maxQ = self.computeValueFromQValues(state)

        # Filtramos todas las acciones que tengan ese Q-value máximo
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQ]

        # Elegimos aleatoriamente entre las mejores (desempate)
        return random.choice(bestActions)

    def getAction(self, state):
        """
        Devuelve la acción a tomar desde el estado actual usando la política ε-greedy.
        Con probabilidad epsilon, se elige una acción aleatoria (exploración).
        Con probabilidad 1 - epsilon, se elige la mejor acción según Q (explotación).
        """

        # Obtenemos todas las acciones legales desde el estado actual
        legalActions = self.getLegalActions(state)

        # Si no hay acciones legales (estado terminal), no se puede hacer nada
        if not legalActions:
            return None

        # Decidimos si exploramos o explotamos usando una moneda sesgada
        if util.flipCoin(self.epsilon):
            # Explorar: elegir una acción aleatoria entre las legales
            return random.choice(legalActions)
        else:
            # Explotar: elegir la mejor acción basada en los valores Q aprendidos
            return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
        Realiza el aprendizaje Q: actualiza Q(s,a) usando la muestra observada.
        Fórmula: Q(s,a) ← (1 - α) * Q(s,a) + α * (r + γ * max_a' Q(s', a'))
        """
        # Estimamos el valor futuro: r + γ * V(s')
        sample = reward + self.discount * self.computeValueFromQValues(nextState)

        # Mezclamos con el valor anterior según el learning rate α
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        # Devuelve la mejor acción aprendida (para visualización o test)
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        # Devuelve el valor actual del estado V(s) = max_a Q(s,a)
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Agente Q-Learning con parámetros por defecto adaptados a Pacman"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        Parámetros:
        - epsilon: tasa de exploración durante el entrenamiento
        - gamma: factor de descuento
        - alpha: tasa de aprendizaje
        - numTraining: número de partidas de entrenamiento
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # Pacman siempre tiene índice 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Usa getAction del QLearningAgent y notifica a Pacman de su acción.
        Este método no debe modificarse.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action





class ApproximateQAgent(PacmanQAgent):
    """
    Agente de Q-Learning Aproximado.
    Solo necesitas sobrescribir getQValue y update.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        # Inicializa el extractor de características, por defecto usa IdentityExtractor
        self.featExtractor = util.lookup(extractor, globals())()

        # Llama al constructor del agente PacmanQAgent
        PacmanQAgent.__init__(self, **args)

        # Inicializa los pesos como un Counter (diccionario con valor por defecto 0)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
        Devuelve Q(s,a) ≈ w · f(s,a)
        donde f(s,a) es el vector de características y w es el vector de pesos.
        """

        # Obtenemos el vector de características para el par (estado, acción)
        features = self.featExtractor.getFeatures(state, action)

        # Calculamos el producto escalar entre características y pesos
        q_value = sum(self.weights[feature] * value for feature, value in features.items())

        # Devolvemos el Q-valor aproximado
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Actualiza los pesos usando la diferencia entre la predicción y la realidad:
        wi ← wi + α * [r + γ * max_a' Q(s',a') - Q(s,a)] * fi(s,a)
        """

        # Extraemos las características del par (estado, acción)
        features = self.featExtractor.getFeatures(state, action)

        # Obtenemos las acciones legales desde el siguiente estado
        nextLegalActions = self.getLegalActions(nextState)

        # Calculamos el mejor Q(s', a') en el siguiente estado
        nextQValue = 0.0
        if nextLegalActions:
            nextQValue = max(self.getQValue(nextState, nextAction) for nextAction in nextLegalActions)

        # Calculamos la diferencia: (target - predicción actual)
        difference = (reward + self.discount * nextQValue) - self.getQValue(state, action)

        # Actualizamos cada peso en función de su característica correspondiente
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """
        Se llama al final de cada episodio.
        Útil para depurar pesos al final del entrenamiento.
        """
        PacmanQAgent.final(self, state)

        # Puedes activar esta línea para imprimir pesos al final del entrenamiento
        # if self.episodesSoFar == self.numTraining:
        #     print("Final weights:", self.weights)

    def getWeights(self):
        # Devuelve el diccionario de pesos actuales
        return self.weights

