# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    
    def runValueIteration(self):
        # Bucle principal que se repite tantas veces como se haya especificado (por defecto 100)
        for i in range(self.iterations):

            # Creamos una copia de los valores actuales (self.values) para hacer una actualización por lotes (batch update)
            # Esto evita usar valores ya actualizados en la misma iteración.
            new_values = self.values.copy()

            # Recorremos todos los estados del MDP
            for state in self.mdp.getStates():

                # Si el estado es terminal (no se puede hacer nada desde aquí), lo saltamos
                if self.mdp.isTerminal(state):
                    continue

                # Variable para guardar el mejor valor Q que encontremos (inicializamos con menos infinito)
                max_value = float('-inf')

                # Recorremos todas las acciones posibles que se pueden hacer desde este estado
                for action in self.mdp.getPossibleActions(state):

                    # Calculamos el valor Q de hacer esta acción desde este estado
                    # Esto tiene en cuenta: recompensa inmediata + valor futuro de los estados a los que puedo llegar
                    q_value = self.computeQValueFromValues(state, action)

                    # Nos quedamos con el mayor valor Q posible
                    max_value = max(max_value, q_value)

                # Una vez encontré la mejor acción, actualizo el valor de este estado en el nuevo diccionario
                # ¡Pero todavía no actualizo self.values! Eso se hace fuera del for
                new_values[state] = max_value

            # Al final de cada iteración, actualizo todos los valores de golpe
            # Esto asegura que todos los estados fueron actualizados con los valores de la iteración anterior
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        # Inicializamos el valor Q en 0
        q_value = 0

        # Obtenemos todas las transiciones posibles desde 'state' con la acción 'action'
        # Cada transición tiene una probabilidad y un estado destino (s')
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):

            # Obtenemos la recompensa inmediata por ir de 'state' a 'next_state' haciendo 'action'
            reward = self.mdp.getReward(state, action, next_state)

            # Sumamos al valor Q: probabilidad * (recompensa + valor futuro descontado)
            # self.discount es el factor γ (gamma), que da menos peso a recompensas futuras
            q_value += prob * (reward + self.discount * self.values[next_state])

        # Retornamos el valor Q total para esta acción
        return q_value


    def computeActionFromValues(self, state):
        # Si el estado es terminal, no hay ninguna acción posible → devolvemos None
        if self.mdp.isTerminal(state):
            return None

        # Inicializamos las variables para encontrar la mejor acción (la de mayor Q)
        best_action = None
        max_q = float('-inf')

        # Recorremos todas las acciones legales en este estado
        for action in self.mdp.getPossibleActions(state):

            # Calculamos el valor Q de hacer esta acción
            q = self.computeQValueFromValues(state, action)

            # Si este valor Q es el mayor que hemos visto hasta ahora, lo guardamos
            if q > max_q:
                max_q = q
                best_action = action

        # Retornamos la acción con el mayor valor Q
        return best_action



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Obtenemos todos los estados del MDP (entorno de decisiones de Markov)
        states = self.mdp.getStates()

        # Repetimos el proceso durante el número total de iteraciones especificadas
        for i in range(self.iterations):
            # Seleccionamos un estado de forma cíclica con módulo
            state = states[i % len(states)]

            # Si el estado es terminal, lo saltamos (no se actualiza su valor)
            if self.mdp.isTerminal(state):
                continue

            # Inicializamos la mejor utilidad esperada (Q-Value) como negativo infinito
            best_value = float('-inf')

            # Calculamos el Q-Value de cada acción posible desde este estado
            for action in self.mdp.getPossibleActions(state):
                q_value = self.computeQValueFromValues(state, action)

                # Guardamos el valor más alto (máximo Q-Value)
                if q_value > best_value:
                    best_value = q_value

            # Asignamos el mejor Q-Value calculado como el nuevo valor del estado
            self.values[state] = best_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    def runValueIteration(self):
        # Obtenemos todos los estados del MDP
        states = self.mdp.getStates()

        # Diccionario para guardar el conjunto de predecesores de cada estado
        predecessors = {}

        # Calculamos los predecesores de todos los estados
        for state in states:
            predecessors[state] = set()  # Usamos set para evitar duplicados

        # Para cada estado, revisamos a dónde puede llegar con cada acción
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        # El estado actual es un predecesor del siguiente estado
                        predecessors[nextState].add(state)

        # Inicializamos la cola de prioridad
        priorityQueue = util.PriorityQueue()

        # Para cada estado no terminal, calculamos su prioridad inicial
        for state in states:
            if self.mdp.isTerminal(state):
                continue

            # Calculamos el mejor Q-value actual para el estado
            max_q_value = max(
                [self.computeQValueFromValues(state, action)
                for action in self.mdp.getPossibleActions(state)]
            )

            # Diferencia entre el valor actual y el nuevo valor estimado
            diff = abs(self.values[state] - max_q_value)

            # Insertamos el estado en la cola con prioridad negativa (mayor prioridad = mayor diferencia)
            priorityQueue.push(state, -diff)

        # Realizamos la iteración de valor durante self.iterations pasos
        for iteration in range(self.iterations):
            # Si la cola está vacía, terminamos antes de agotar las iteraciones
            if priorityQueue.isEmpty():
                break

            # Sacamos el estado con mayor prioridad (mayor diferencia)
            state = priorityQueue.pop()

            # Si el estado no es terminal, actualizamos su valor
            if not self.mdp.isTerminal(state):
                max_q_value = max(
                    [self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)]
                )
                self.values[state] = max_q_value

            # Para cada predecesor del estado actualizado
            for pred in predecessors[state]:
                # Calculamos el mejor Q-value posible del predecesor
                max_q_value_pred = max(
                    [self.computeQValueFromValues(pred, action)
                    for action in self.mdp.getPossibleActions(pred)]
                )

                # Calculamos la diferencia entre el valor actual y el estimado
                diff = abs(self.values[pred] - max_q_value_pred)

                # Si supera el umbral theta, actualizamos su prioridad en la cola
                if diff > self.theta:
                    priorityQueue.update(pred, -diff)  # Actualizamos con prioridad negativa

        



            




