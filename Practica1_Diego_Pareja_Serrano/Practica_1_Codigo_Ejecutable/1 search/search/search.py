# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Implementación de la búsqueda en profundidad (DFS).
    Explora el nodo más profundo posible antes de retroceder.
    """

    #  1. Inicialización de estructuras de datos
    stack = util.Stack()  # Usamos una pila (Stack) para la exploración en profundidad.
    visited = set()  # Conjunto para almacenar los estados visitados y evitar ciclos.

    #  2. Obtener el estado inicial del problema.
    start_state = problem.getStartState()
    
    #  3. Insertar el estado inicial en la pila con:
    # - Su estado actual
    # - La lista de acciones tomadas hasta ese estado
    # - Su costo acumulado (aunque DFS no usa el costo, se mantiene por consistencia)
    stack.push((start_state, [], 0))  # (estado, lista de acciones, costo acumulado)

    #  4. Mientras haya nodos en la pila, seguimos explorando.
    while not stack.isEmpty():
        #  5. Extraer el nodo actual de la pila (último en entrar, primero en salir - LIFO).
        state, actions, cost = stack.pop()

        # 6. Si alcanzamos el objetivo, retornamos la lista de acciones que nos llevó hasta aquí.
        if problem.isGoalState(state):
            return actions

        #  7. Si el estado no ha sido visitado antes, lo marcamos como visitado.
        if state not in visited:
            visited.add(state)

            #  8. Expandimos los sucesores del estado actual.
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:  # Evitamos volver a visitar nodos ya explorados.
                    stack.push((successor, actions + [action], cost + step_cost))

    return []  # 9. Si no encontramos una solución, devolvemos una lista vacía.


def breadthFirstSearch(problem):
    """
    Implementación de la búsqueda en anchura (BFS).
    Explora los nodos más cercanos primero antes de avanzar a niveles más profundos.
    """

    # 1. Inicialización de estructuras de datos
    cola = util.Queue()  # Usamos una cola (FIFO) porque BFS expande los nodos en orden de llegada.
    visitado = set()  # Conjunto para almacenar los estados visitados y evitar ciclos.

    # 2. Obtener el estado inicial del problema.
    nodo_inicial = problem.getStartState()
    
    # 3. Insertar el estado inicial en la cola con:
    # - Su estado actual
    # - La lista de acciones tomadas hasta ese estado
    # - Su costo acumulado (BFS no usa el costo, pero se mantiene por consistencia)
    cola.push((nodo_inicial, [], 0))  # (estado, lista de acciones, costo acumulado)

    # 4. Mientras haya nodos en la cola, seguimos explorando.
    while not cola.isEmpty():
        # 5. Extraer el nodo actual de la cola (primero en entrar, primero en salir - FIFO).
        estado, accion, coste = cola.pop()

        # 6. Si alcanzamos el objetivo, retornamos la lista de acciones que nos llevó hasta aquí.
        if problem.isGoalState(estado):
            return accion
        
        # 7. Si el estado no ha sido visitado antes, lo marcamos como visitado.
        if estado not in visitado:
            visitado.add(estado)

            #8. Expandimos los sucesores del estado actual.
            for sucesor, accion_sucesor, coste_sucesor in problem.getSuccessors(estado):
                if sucesor not in visitado:  # Evitamos volver a visitar nodos ya explorados.
                    cola.push((sucesor, accion + [accion_sucesor], coste + coste_sucesor))

    return []  #9. Si no encontramos una solución, devolvemos una lista vacía.



def uniformCostSearch(problem):
    """
    Implementación de la búsqueda de costo uniforme (UCS).
    Explora siempre el nodo con el menor costo acumulado primero.
    """

    #  1. Inicialización de estructuras de datos
    cola_prioridad = util.PriorityQueue()  # Cola de prioridad para expandir nodos con menor costo.
    visitados = {}  # Diccionario para registrar el costo mínimo alcanzado por cada estado.

    #  2. Obtener el estado inicial del problema.
    nodo_inicial = problem.getStartState()

    #  3. Insertar el estado inicial en la cola de prioridad con:
    # - Su estado actual.
    # - La lista de acciones tomadas hasta ese estado.
    # - Su costo acumulado (inicialmente 0).
    # - La prioridad basada en el costo (0 para el inicio).
    cola_prioridad.push((nodo_inicial, [], 0), 0)  # (estado, lista de acciones, costo acumulado), prioridad = costo.

    #  4. Mientras haya nodos en la cola, seguimos explorando.
    while not cola_prioridad.isEmpty():
        #  5. Extraer el nodo con el menor costo acumulado.
        estado, acciones, coste = cola_prioridad.pop()

        #  6. Si encontramos el estado objetivo, retornamos la lista de acciones que nos llevó hasta aquí.
        if problem.isGoalState(estado):
            return acciones

        #  7. Si el estado no ha sido visitado antes, o encontramos un camino más barato, lo procesamos.
        if estado not in visitados or coste < visitados[estado]:
            visitados[estado] = coste  # Registramos el menor costo encontrado para este estado.

            #  8. Expandimos los sucesores del estado actual.
            for sucesor, accion_sucesor, coste_sucesor in problem.getSuccessors(estado):
                nuevo_coste = coste + coste_sucesor  # Calculamos el costo total hasta el sucesor.
                cola_prioridad.push((sucesor, acciones + [accion_sucesor], nuevo_coste), nuevo_coste)

    return []  #  9 Si no encontramos una solución, devolvemos una lista vacía.



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):


    cola_prioridad = util.PriorityQueue()  # Creamos una cola de prioridad
    visitados = {}  # Diccionario para registrar el menor costo encontrado por cada estado

    nodo_inicial = problem.getStartState()  # Obtenemos el estado inicial
    cola_prioridad.push((nodo_inicial, [], 0), 0)  # Añadimos el nodo inicial a la cola de prioridad con el costo 0

    while not cola_prioridad.isEmpty():  # Mientras la cola no esté vacía
        estado, acciones, coste = cola_prioridad.pop()  # Extraemos el nodo con el menor costo total (g(n) + h(n))

        if problem.isGoalState(estado):  # Si hemos alcanzado el estado objetivo
            return acciones  # Retornamos el camino de acciones que llevamos hasta el objetivo

        if estado not in visitados or coste < visitados[estado]:  # Si no hemos visitado el estado o encontramos un camino más barato
            visitados[estado] = coste  # Marcamos el estado como visitado y guardamos el costo

            for sucesor, accion_sucesor, coste_sucesor in problem.getSuccessors(estado):  # Expandimos el estado
                nuevo_coste = coste + coste_sucesor  # Calculamos el nuevo costo (g(n) del sucesor)
                prioridad = nuevo_coste + heuristic(sucesor, problem)  # Calculamos la prioridad: f(n) = g(n) + h(n)
                cola_prioridad.push((sucesor, acciones + [accion_sucesor], nuevo_coste), prioridad)  # Añadimos el sucesor a la cola con su prioridad

    return []  # Si no encontramos solución, retornamos una lista vacía


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
