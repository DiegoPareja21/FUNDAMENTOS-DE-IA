# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    Un agente reflejo que elige una acción basándose en la evaluación de los posibles estados futuros.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción basada en la evaluación de los posibles movimientos.
        """
        # 1. Obtener todas las acciones legales que Pac-Man puede realizar en el estado actual.
        legalMoves = gameState.getLegalActions()

        # 2. Evaluar cada acción usando la función de evaluación personalizada.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        # 3. Identificar la mejor puntuación obtenida entre todas las acciones evaluadas.
        bestScore = max(scores)

        # 4. Seleccionar todas las acciones que tienen la mejor puntuación.
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        # 5. Elegir una de las mejores acciones de manera aleatoria para evitar patrones repetitivos.
        chosenIndex = random.choice(bestIndices)

        # 6. Devolver la acción elegida.
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evalúa qué tan buena es una acción en función del nuevo estado del juego.
        """
        # 1. Generar el estado sucesor después de que Pac-Man realice la acción.
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # 2. Obtener la nueva posición de Pac-Man después de moverse.
        newPos = successorGameState.getPacmanPosition()

        # 3. Obtener la lista de posiciones de comida en el nuevo estado.
        newFood = successorGameState.getFood().asList()

        # 4. Obtener los estados de los fantasmas después del movimiento.
        newGhostStates = successorGameState.getGhostStates()

        # 5. Obtener la lista de tiempos de asustado de los fantasmas (cuánto tiempo seguirán asustados).
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # 6. Obtener la lista de cápsulas disponibles en el nuevo estado.
        newCapsules = successorGameState.getCapsules()

        # 7. Evaluar la cercanía de los fantasmas y aplicar penalizaciones o recompensas según su estado.
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            # Si un fantasma no está asustado y está muy cerca, penalizar severamente.
            if ghostDist < 2 and ghostState.scaredTimer == 0:
                return -float('inf')  # Movimiento peligroso, Pac-Man podría perder.

            # Si un fantasma está asustado y cerca, dar una alta recompensa por la oportunidad de comerlo.
            elif ghostState.scaredTimer > 0 and ghostDist < 2:
                return 1000  # Priorizar moverse hacia fantasmas asustados.

        # 8. Evaluar la distancia a la comida más cercana y asignar una puntuación en base a eso.
        minFoodDist = min([manhattanDistance(newPos, food) for food in newFood], default=1)
        foodScore = 10.0 / minFoodDist  # Más prioridad a comida más cercana.

        # 9. Evaluar la proximidad de cápsulas y darles prioridad si hay fantasmas cerca.
        minCapsuleDist = min([manhattanDistance(newPos, cap) for cap in newCapsules], default=1)
        if minCapsuleDist < 5:  # Si hay una cápsula cercana, darle una prioridad moderada.
            capsuleScore = 50.0 / minCapsuleDist
        else:
            capsuleScore = 0

        # 10. Obtener la puntuación general del estado sucesor.
        gameScore = successorGameState.getScore()

        # 11. Combinar todas las puntuaciones para obtener el valor final de evaluación del movimiento.
        return gameScore + foodScore + capsuleScore
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    
"""
    def evaluationFunction(self, currentGameState, action):
        """
"""
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
"""
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()
"""
def scoreEvaluationFunction(currentGameState):

    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """
   

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Agente que usa el algoritmo Minimax para decidir el mejor movimiento.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción para Pac-Man usando Minimax.
        """

        def minimax(state, depth, agentIndex):
            """
            Función recursiva de Minimax.
            
            Parámetros:
            - state: estado actual del juego.
            - depth: profundidad actual en el árbol de búsqueda.
            - agentIndex: índice del agente que juega (Pac-Man es 0, los fantasmas son 1, 2, ...).
            """
            # 1. Caso base: Si hemos alcanzado la profundidad máxima o el juego terminó (ganado o perdido),
            #    devolvemos la evaluación del estado actual.
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # 2. Turno de Pac-Man (MAX): intentamos maximizar la puntuación.
            if agentIndex == 0:
                return max(
                    (minimax(state.generateSuccessor(agentIndex, action), depth, 1), action)  
                    for action in state.getLegalActions(agentIndex)  # Evaluamos todas las acciones posibles.
                )[0]  # Retornamos solo el valor máximo, ignorando la acción en esta parte.

            else:
                # 3. Turno de los fantasmas (MIN): intentan minimizar la puntuación de Pac-Man.
                nextAgent = (agentIndex + 1) % state.getNumAgents()  # Avanzamos al siguiente agente.
                nextDepth = depth + 1 if nextAgent == 0 else depth  # Aumentamos la profundidad solo cuando vuelve a ser Pac-Man.

                return min(
                    minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)  
                    for action in state.getLegalActions(agentIndex)  # Evaluamos todas las acciones posibles.
                )  # Retornamos el valor mínimo ya que los fantasmas intentan minimizar la puntuación de Pac-Man.

        # 4. Ejecutamos Minimax desde Pac-Man (agente 0) para decidir la mejor acción.
        #    Se exploran todas las acciones posibles de Pac-Man y se elige la que tenga el mejor valor Minimax.
        bestAction = max(
            (minimax(gameState.generateSuccessor(0, action), 0, 1), action)  
            for action in gameState.getLegalActions(0)  # Evaluamos todas las acciones legales de Pac-Man.
        )[1]  # Retornamos la mejor acción en lugar del valor Minimax.

        return bestAction  # Devolvemos la acción óptima para Pac-Man.

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Agente que usa el algoritmo Minimax con poda alfa-beta para decidir el mejor movimiento.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción para Pac-Man usando Minimax con poda alfa-beta.
        """

        # 1. Inicializamos los valores de α (mínimo garantizado para MAX) y β (máximo garantizado para MIN).
        alpha = float("-inf")
        beta = float("inf")
        best_action = None  # Acción óptima a seleccionar.
        max_value = float("-inf")  # Inicializamos el valor máximo.

        # 2. Exploramos todas las acciones legales de Pac-Man (agente 0).
        for action in gameState.getLegalActions(0):
            # 3. Llamamos a la función recursiva alphaBeta para obtener el valor de la acción.
            value = self.alphaBeta(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            
            # 4. Si encontramos un mejor valor, actualizamos la mejor acción.
            if value > max_value:
                max_value = value
                best_action = action

            # 5. Actualizamos el valor de α (el mejor valor encontrado hasta ahora para MAX).
            alpha = max(alpha, value)

        return best_action  # 6. Retornamos la acción óptima para Pac-Man.

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        """
        Implementación recursiva de Minimax con poda alfa-beta.
        
        Parámetros:
        - gameState: estado actual del juego.
        - agentIndex: índice del agente que juega (Pac-Man es 0, los fantasmas son 1, 2, ...).
        - depth: profundidad actual en el árbol de búsqueda.
        - alpha: mejor valor encontrado para MAX.
        - beta: mejor valor encontrado para MIN.
        """

        # 🚨 Condición de parada: si alcanzamos la profundidad máxima o el juego termina (ganado o perdido).
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # 🟡 Turno de Pac-Man (MAX).
        if agentIndex == 0:
            return self.maxValue(gameState, depth, alpha, beta)
        
        # 👻 Turno de los fantasmas (MIN).
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, depth, alpha, beta):
        """
        Función para maximizar la ganancia de Pac-Man.
        """

        max_value = float("-inf")  # Inicializamos el valor máximo.

        # 1. Recorremos todas las acciones legales de Pac-Man.
        for action in gameState.getLegalActions(0):
            # 2. Calculamos el valor minimax con poda alfa-beta para el siguiente turno (fantasmas).
            max_value = max(max_value, self.alphaBeta(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
            
            # 3. Si encontramos un valor mayor que β, no es necesario seguir evaluando (poda β).
            if max_value > beta:
                return max_value  
            
            # 4. Actualizamos α con el mejor valor encontrado hasta ahora.
            alpha = max(alpha, max_value)

        return max_value  # 5. Retornamos el mejor valor posible para Pac-Man.

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        """
        Función para minimizar la ganancia de Pac-Man (jugadas de los fantasmas).
        """

        min_value = float("inf")  # Inicializamos el valor mínimo.
        next_agent = agentIndex + 1  # Pasamos al siguiente fantasma.
        num_agents = gameState.getNumAgents()  # Número total de agentes en el juego.

        # 1. Si es el último fantasma, el siguiente turno es de Pac-Man con profundidad +1.
        if next_agent == num_agents:
            next_agent = 0  # Volvemos a Pac-Man.
            depth += 1  # Aumentamos la profundidad de búsqueda.

        # 2. Recorremos todas las acciones legales del fantasma actual.
        for action in gameState.getLegalActions(agentIndex):
            # 3. Calculamos el valor minimax con poda alfa-beta para el siguiente turno.
            min_value = min(min_value, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), next_agent, depth, alpha, beta))
            
            # 4. Si encontramos un valor menor que α, no es necesario seguir evaluando (poda α).
            if min_value < alpha:
                return min_value  
            
            # 5. Actualizamos β con el mejor valor encontrado hasta ahora.
            beta = min(beta, min_value)
        
        return min_value  # 6. Retornamos el peor valor posible para Pac-Man (mejor para los fantasmas).


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Agente que usa el algoritmo Expectimax para decidir el mejor movimiento.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción para Pac-Man usando Expectimax.
        """

        best_action = None  # Acción óptima a seleccionar.
        max_value = float("-inf")  # Inicializamos el valor máximo.

        # 1. Exploramos todas las acciones legales de Pac-Man (agente 0).
        for action in gameState.getLegalActions(0):
            # 2. Llamamos a la función recursiva expectimax para obtener el valor de la acción.
            value = self.expectimax(gameState.generateSuccessor(0, action), 1, 0)
            
            # 3. Si encontramos un mejor valor, actualizamos la mejor acción.
            if value > max_value:
                max_value = value
                best_action = action

        return best_action  # 4. Retornamos la acción óptima para Pac-Man.

    def expectimax(self, gameState, agentIndex, depth):
        """
        Implementación recursiva de Expectimax.
        
        Parámetros:
        - gameState: estado actual del juego.
        - agentIndex: índice del agente que juega (Pac-Man es 0, los fantasmas son 1, 2, ...).
        - depth: profundidad actual en el árbol de búsqueda.
        """

        # 🚨 Condición de parada: si alcanzamos la profundidad máxima o el juego termina (ganado o perdido).
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # 🟡 Turno de Pac-Man (MAX).
        if agentIndex == 0:
            return self.maxValue(gameState, depth)
        
        # 👻 Turno de los fantasmas (EXPECTIMAX).
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, depth):
        """
        Función para maximizar la ganancia de Pac-Man.
        """

        max_value = float("-inf")  # Inicializamos el valor máximo.

        # 1. Recorremos todas las acciones legales de Pac-Man.
        for action in gameState.getLegalActions(0):
            # 2. Calculamos el valor expectimax para el siguiente turno (fantasmas).
            max_value = max(max_value, self.expectimax(gameState.generateSuccessor(0, action), 1, depth))
        
        return max_value  # 3. Retornamos el mejor valor posible para Pac-Man.

    def expValue(self, gameState, agentIndex, depth):
        """
        Función para calcular el valor esperado de las acciones de los fantasmas.
        """

        next_agent = agentIndex + 1  # Pasamos al siguiente fantasma.
        num_agents = gameState.getNumAgents()  # Número total de agentes en el juego.

        # 1. Si es el último fantasma, el siguiente turno es de Pac-Man con profundidad +1.
        if next_agent == num_agents:
            next_agent = 0  # Volvemos a Pac-Man.
            depth += 1  # Aumentamos la profundidad de búsqueda.

        actions = gameState.getLegalActions(agentIndex)  # Obtiene todas las acciones legales del fantasma.

        # 2. Si el fantasma no tiene movimientos válidos, devolvemos la evaluación del estado actual.
        if not actions:
            return self.evaluationFunction(gameState)

        # 3. Calculamos el valor esperado como el promedio de todas las acciones posibles.
        total_value = sum(
            self.expectimax(gameState.generateSuccessor(agentIndex, action), next_agent, depth)
            for action in actions
        )

        return total_value / len(actions)  # 4. Promediamos para obtener la expectativa.




def betterEvaluationFunction(currentGameState):
    """
    Una función de evaluación mejorada para Pac-Man.
    """

    # 📌 1. Obtener información relevante del estado actual
    pacmanPos = currentGameState.getPacmanPosition()  # Posición actual de Pac-Man.
    foodList = currentGameState.getFood().asList()  # Lista de posiciones de comida en el mapa.
    ghostStates = currentGameState.getGhostStates()  # Estados actuales de los fantasmas.
    capsules = currentGameState.getCapsules()  # Posiciones de las cápsulas de poder.
    score = currentGameState.getScore()  # Puntuación actual del juego.

    # 🍏 2. Calcular la distancia a la comida más cercana.
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
    else:
        minFoodDist = 0  # No hay comida restante, por lo que no se considera penalización.

    # 👻 3. Calcular la distancia a los fantasmas y su estado (asustados o no).
    ghostDistances = []  # Lista para almacenar las distancias a los fantasmas.
    scaredGhosts = 0  # Contador de fantasmas asustados.

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()  # Obtener la posición del fantasma.
        distance = manhattanDistance(pacmanPos, ghostPos)  # Calcular distancia entre Pac-Man y el fantasma.
        ghostDistances.append(distance)

        # Si el fantasma está asustado, se considera una oportunidad para comerlo.
        if ghost.scaredTimer > 0:
            scaredGhosts += 1

    # Si hay fantasmas, obtener la distancia mínima a uno de ellos.
    minGhostDist = min(ghostDistances) if ghostDistances else float("inf")

    # 🟣 4. Calcular la distancia a la cápsula más cercana.
    minCapsuleDist = min(manhattanDistance(pacmanPos, cap) for cap in capsules) if capsules else 0

    # 🎯 5. Calcular la función de evaluación combinando múltiples factores.
    evaluation = (
        score  # ✅ Prioriza el estado con mayor puntuación general.
        + (1.0 / (1 + minFoodDist)) * 10  # Favorece la comida más cercana.
        - (1.0 / (1 + minGhostDist)) * 15  # Penaliza estar cerca de fantasmas.
        + (1.0 / (1 + minCapsuleDist)) * 5  # Prioriza la recolección de cápsulas.
        + (scaredGhosts * 20)  # Recompensa la presencia de fantasmas asustados (pueden ser comidos).
    )

    return evaluation

# Abreviatura para el evaluador
better = betterEvaluationFunction

