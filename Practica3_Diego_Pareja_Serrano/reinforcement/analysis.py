# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # Factor de descuento (γ): se mantiene en 0.9 para valorar recompensas futuras.
    answerDiscount = 0.9

    # Ruido: se elimina (0.0) para que el agente no falle al cruzar el puente.
    answerNoise = 0.0

    # Devuelve los parámetros que permiten al agente cruzar el puente.
    return answerDiscount, answerNoise


def question3a():
    # Prefiere la salida cercana, arriesgando el acantilado
    answerDiscount = 0.1        # Descuento bajo: no valora recompensas futuras, prefiere salir pronto.
    answerNoise = 0.0           # Sin ruido: no hay riesgo al caminar cerca del acantilado.
    answerLivingReward = -1.0   # Vivir penaliza, lo que incentiva terminar rápido por la salida cercana.
    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    # Prefiere la salida cercana, pero evitando el acantilado
    answerDiscount = 0.3        # Valor medio: sigue prefiriendo salir pronto.
    answerNoise = 0.2           # Con ruido: acercarse al acantilado implica riesgo, así que lo evita.
    answerLivingReward = -1.0   # Vivir sigue siendo costoso, por lo que saldrá, pero con más precaución.
    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    # Prefiere la salida lejana, arriesgando el acantilado 
    answerDiscount = 0.6        # Valor medio-alto: el +10 de la salida lejana empieza a importar más.
    answerNoise = 0.0           # Sin ruido: se puede arriesgar el camino corto junto al acantilado.
    answerLivingReward = 0      # Vivir no penaliza ni recompensa, por lo que el +10 se vuelve más atractivo.
    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    # Prefiere la salida lejana , evitando el acantilado 
    answerDiscount = 0.6        # Valora más el futuro, así que el +10 es deseable.
    answerNoise = 0.2           # Con ruido: el agente evita rutas arriesgadas como el borde inferior.
    answerLivingReward = 0      # Vivir no castiga ni premia, así que el agente toma un camino largo pero seguro.
    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    # Evita tanto las salidas como el acantilado (no termina nunca)
    answerDiscount = 0.9        # Descuento alto: le importa el largo plazo, pero no hay recompensa futura.
    answerNoise = 0.2           # Ruido normal: se ignora porque el agente no quiere moverse mucho.
    answerLivingReward = 1.0    # Vivir es una recompensa, así que el agente prefiere seguir vivo eternamente.
    return answerDiscount, answerNoise, answerLivingReward



def question8():
    """
    Devuelve una tupla (epsilon, learningRate) si existe una combinación que permite aprender
    la política óptima en BridgeGrid tras 50 episodios.
    Si no existe, devuelve 'NOT POSSIBLE'.
    """

    # Establecemos una tasa de exploración moderada:
    # suficiente para explorar caminos, pero no tan alta como para actuar al azar.
    epsilon = 0.1

    # Establecemos una tasa de aprendizaje suficientemente alta para que el agente
    # incorpore rápidamente lo aprendido en los pocos episodios disponibles.
    learningRate = 0.5

    return epsilon, learningRate

    # Si no existiera una combinación confiable, usaríamos esto:
    # return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
