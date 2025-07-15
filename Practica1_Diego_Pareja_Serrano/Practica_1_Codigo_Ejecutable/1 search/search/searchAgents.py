# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################



def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

# Código propio 

class CornersProblem(search.SearchProblem):
    """
    Definición del problema de búsqueda en las esquinas (CornersProblem).
    
    En este problema, Pacman debe recorrer el laberinto y visitar todas las esquinas al menos una vez.
    """

    def __init__(self, startingGameState):
        """
        Inicializa el problema con la información del estado inicial del juego.

        Parámetros:
        - startingGameState: Estado inicial del juego que contiene la posición de Pacman y el mapa del laberinto.
        """
        self.startingPosition = startingGameState.getPacmanPosition()  # Posición inicial de Pacman
        self.walls = startingGameState.getWalls()  # Mapa de las paredes del laberinto

        # Definimos las coordenadas de las cuatro esquinas del laberinto
        self.corners = (
            (1, 1),
            (1, self.walls.height - 2),
            (self.walls.width - 2, 1),
            (self.walls.width - 2, self.walls.height - 2),
        )

        self._expanded = 0  # Contador de nodos expandidos para análisis de rendimiento

    def getStartState(self):
        """
        Retorna el estado inicial del problema.

        El estado está representado como una tupla:
        - Posición de Pacman en el laberinto.
        - Lista de esquinas visitadas hasta el momento (vacía al inicio).
        """
        return (self.startingPosition, ())

    def isGoalState(self, state):
        """
        Determina si un estado dado es un estado objetivo.

        Parámetros:
        - state: Estado actual representado como (posición de Pacman, esquinas visitadas).

        Retorna:
        - True si se han visitado todas las esquinas, False en caso contrario.
        """
        _, visitedCorners = state  # Extraemos las esquinas visitadas en el estado actual
        return len(visitedCorners) == 4  # Si se visitaron todas las esquinas, se alcanzó el objetivo

    def getSuccessors(self, state):
        """
        Genera los sucesores de un estado dado.

        Parámetros:
        - state: Estado actual representado como (posición de Pacman, esquinas visitadas).

        Retorna:
        - Una lista de sucesores, donde cada sucesor es una tupla:
          ((nuevaPosición, nuevasEsquinasVisitadas), direcciónTomada, costo).
        """
        successors = []  # Lista para almacenar los sucesores generados
        position, visitedCorners = state  # Extraemos la posición actual y las esquinas visitadas

        # Iteramos sobre las direcciones posibles (Norte, Sur, Este, Oeste)
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position  # Coordenadas actuales de Pacman
            dx, dy = Actions.directionToVector(direction)  # Convertimos la dirección en desplazamiento (dx, dy)
            nextPosition = (int(x + dx), int(y + dy))  # Calculamos la nueva posición

            # Verificamos que la nueva posición no esté bloqueada por una pared
            if not self.walls[nextPosition[0]][nextPosition[1]]:  
                newVisitedCorners = visitedCorners  # Mantenemos la lista de esquinas visitadas

                # Si la nueva posición es una esquina y aún no ha sido visitada, la agregamos a la lista
                if nextPosition in self.corners and nextPosition not in visitedCorners:
                    newVisitedCorners = visitedCorners + (nextPosition,)

                # Agregamos el sucesor a la lista con su respectiva dirección y costo (siempre 1)
                successors.append(((nextPosition, newVisitedCorners), direction, 1))

        self._expanded += 1  # Aumentamos el contador de nodos expandidos
        return successors  # Retornamos la lista de sucesores generados

    def getCostOfActions(self, actions):
        """
        Calcula el costo total de una secuencia de acciones.

        Parámetros:
        - actions: Lista de acciones tomadas.

        Retorna:
        - El número total de movimientos realizados (cada movimiento cuesta 1).
        - Si actions es None, retorna infinito, indicando que la secuencia no es válida.
        """
        if actions is None:
            return float("inf")  # Si no hay acciones disponibles, el costo es infinito
        return len(actions)  # El costo es simplemente la cantidad de movimientos realizados



def cornersHeuristic(state, problem):
    """
    Heurística para el problema de búsqueda en las esquinas (CornersProblem).

    Parámetros:
    - state: Estado actual de búsqueda, representado como una tupla:
      (posición de Pacman, esquinas visitadas).
    - problem: Instancia del problema CornersProblem, que contiene información sobre 
      las esquinas y las paredes del laberinto.

    Retorna:
    - Un valor heurístico que representa una estimación del costo restante para 
      alcanzar el estado objetivo.
    """

    corners = problem.corners  # Lista de coordenadas de las esquinas del laberinto
    walls = problem.walls  # Mapa de las paredes del laberinto (matriz de celdas bloqueadas)

    # Extraemos la posición actual de Pacman y las esquinas que ya ha visitado
    position, visitedCorners = state  

    # Generamos una lista de esquinas que aún no han sido visitadas
    unvisitedCorners = [corner for corner in corners if corner not in visitedCorners]

    # Si ya se han visitado todas las esquinas, la heurística es 0 (estado objetivo alcanzado)
    if not unvisitedCorners:
        return 0

    # Calculamos la distancia de Manhattan desde la posición actual a cada esquina no visitada
    distances = [util.manhattanDistance(position, corner) for corner in unvisitedCorners]

    # Calculamos la distancia máxima entre cualquier par de esquinas no visitadas
    max_distance = 0

    # Recorremos todas las combinaciones posibles de esquinas no visitadas
    for i in range(len(unvisitedCorners)):
        for j in range(i + 1, len(unvisitedCorners)):
            distance = util.manhattanDistance(unvisitedCorners[i], unvisitedCorners[j])
            if distance > max_distance:
                max_distance = distance  # Actualizamos si encontramos una distancia mayor

    # La heurística devuelve la suma de:
    # 1. La distancia mínima desde la posición actual a una esquina no visitada.
    # 2. La distancia máxima entre cualquier par de esquinas no visitadas.
    return min(distances) + max_distance



class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

#INTENTO DE PROBLEMA CORRECTO
class FoodSearchProblem(PositionSearchProblem):
    """
    Representa el problema de búsqueda de comida en Pacman.  
    El objetivo es recoger toda la comida del tablero, optimizando el camino recorrido.
    """

    def __init__(self, startingGameState):
        """
        Inicializa el problema con el estado inicial del juego.

        Parámetros:
        - startingGameState: Estado inicial del juego, que contiene la posición de Pacman,
          la distribución de la comida y las paredes.
        """
        self.start = startingGameState.getPacmanPosition()  # Posición inicial de Pacman
        self.food = startingGameState.getFood()  # Distribución de la comida en el tablero
        self.walls = startingGameState.getWalls()  # Distribución de las paredes
        self.startingGameState = startingGameState  # Estado completo del juego
        self._expanded = 0  # Contador de nodos expandidos durante la búsqueda

    def getStartState(self):
        """
        Devuelve el estado inicial del problema.

        Retorna:
        - Una tupla (posición de Pacman, estado de la comida).
        """
        return (self.start, self.food)

    def isGoalState(self, state):
        """
        Comprueba si se ha alcanzado el estado objetivo.

        Parámetros:
        - state: Estado actual del juego, que incluye la posición de Pacman y la comida restante.

        Retorna:
        - True si no queda comida en el tablero, False en caso contrario.
        """
        pacmanPosition, foodGrid = state
        return foodGrid.count() == 0  # Si no queda comida, se ha alcanzado el objetivo

    def getSuccessors(self, state):
        """
        Genera los sucesores de un estado dado, explorando posibles movimientos.

        Parámetros:
        - state: Estado actual, representado por (posición de Pacman, estado de la comida).

        Retorna:
        - Una lista de sucesores, donde cada sucesor es una tupla:
          ((nueva posición de Pacman, nuevo estado de la comida), acción, costo)
        """
        successors = []
        pacmanPosition, foodGrid = state

        # Iteramos sobre todas las direcciones posibles (Norte, Sur, Este, Oeste)
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = pacmanPosition  # Extraemos coordenadas actuales de Pacman
            dx, dy = Actions.directionToVector(action)  # Convertimos acción en movimiento
            nextPosition = (int(x + dx), int(y + dy))  # Calculamos la nueva posición

            # Verificamos que la nueva posición no sea una pared
            if not self.walls[nextPosition[0]][nextPosition[1]]:
                nextFood = foodGrid.copy()  # Copiamos el estado de la comida

                # Si hay comida en la nueva posición, la "comemos" (la eliminamos del grid)
                if nextFood[nextPosition[0]][nextPosition[1]]:
                    nextFood[nextPosition[0]][nextPosition[1]] = False

                # Añadimos el sucesor con un costo de 1 por cada movimiento
                successors.append(((nextPosition, nextFood), action, 1))

        self._expanded += 1  # Aumentamos el contador de nodos expandidos
        return successors

    def getCostOfActions(self, actions):
        """
        Calcula el costo de un conjunto de acciones.

        Parámetros:
        - actions: Lista de acciones tomadas.

        Retorna:
        - El costo total, que en este caso es simplemente el número de pasos dados.
        """
        return len(actions)


# IMPLEMENTACIÓN DE UNA HEURÍSTICA PROPIA
def foodHeuristic(state, problem):
    """
    Heurística para el problema de búsqueda de comida en Pacman.
    Se basa en la distancia de Manhattan a la comida más lejana.

    Parámetros:
    - state: Estado actual del juego, que incluye la posición de Pacman y la comida restante.
    - problem: Instancia del problema de búsqueda de comida.

    Retorna:
    - Un valor heurístico basado en la distancia a la comida más lejana.
    """
    pacmanPosition, foodGrid = state
    foodList = foodGrid.asList()  # Convertimos la cuadrícula de comida en una lista de posiciones

    if not foodList:
        return 0  # Si no queda comida, la heurística es 0 (ya estamos en el objetivo)

    # Calculamos la distancia de Manhattan a la comida más lejana
    return max([util.manhattanDistance(pacmanPosition, food) for food in foodList])









class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem



#CODIGO PROPIO
class ClosestDotSearchAgent(SearchAgent):
    """
    Un agente que busca la comida más cercana en cada paso.  
    Esta estrategia es subóptima porque no planea una ruta eficiente para recoger toda la comida,  
    sino que simplemente va al punto más cercano en cada momento.
    """

    def findPathToClosestDot(self, gameState):
        """
        Encuentra y devuelve el camino hasta el punto de comida más cercano.
        
        Parámetros:
        - gameState: Estado actual del juego, que incluye la posición del agente y la distribución de la comida.
        
        Retorna:
        - Una lista de acciones (movimientos) que llevan al agente al punto de comida más cercano.
        """

        # Se crea un problema de búsqueda donde el objetivo es alcanzar cualquier punto de comida.
        problem = AnyFoodSearchProblem(gameState)

        # Se importa el algoritmo de búsqueda en anchura (BFS), que es óptimo en este caso.
        from search import breadthFirstSearch

        # Se ejecuta BFS sobre el problema y se devuelve el camino encontrado.
        return breadthFirstSearch(problem)




def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
#CODIGO PROPIO 



class AnyFoodSearchProblem(PositionSearchProblem):
    """
    Un problema de búsqueda en el que Pacman debe encontrar cualquier punto de comida.
    Se diferencia del problema general de búsqueda de comida en que no necesita recoger toda la comida,
    solo llegar al punto de comida más cercano.
    """

    def __init__(self, gameState):
        """
        Inicializa el problema con la información del estado del juego.

        Parámetros:
        - gameState: Estado actual del juego, que contiene la posición de Pacman y la distribución de la comida.
        """
        self.food = gameState.getFood()  # Obtenemos la matriz de comida del estado del juego
        self.start = gameState.getPacmanPosition()  # Posición inicial de Pacman

        # Llamamos al constructor de PositionSearchProblem para heredar sus funcionalidades
        super().__init__(gameState)

    def isGoalState(self, state):
        """
        Determina si un estado dado es un estado objetivo.

        Parámetros:
        - state: Estado actual representado por la posición de Pacman (x, y).

        Retorna:
        - True si en la posición (x, y) hay comida, False en caso contrario.
        """
        x, y = state
        return self.food[x][y]  # Verifica si hay comida en la posición actual
