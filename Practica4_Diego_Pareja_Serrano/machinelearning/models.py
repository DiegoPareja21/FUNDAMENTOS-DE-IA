import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Inicializa una nueva instancia de Perceptrón.

        Crea un vector de pesos w entrenable de tamaño 1 x dimensions.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Devuelve el parámetro actual de pesos.
        """
        return self.w

    def run(self, x):
        """
        Calcula el producto escalar entre los pesos y la entrada x.

        Entrada:
            x: nodo Constant con forma (1 x dimensions)
        Salida:
            Nodo DotProduct con el valor w · x
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Devuelve la clase predicha para x:
        - 1 si el producto escalar w · x es >= 0
        - -1 si es negativo
        """
        score = self.run(x)                 # Nodo escalar: w · x
        value = nn.as_scalar(score)        # Convertimos el nodo a número real
        return 1 if value >= 0 else -1     # Clasificación binaria

    def train(self, dataset):
        """
        Entrena el perceptrón hasta que clasifique todos los ejemplos correctamente.

        Recorre el dataset y ajusta los pesos si comete un error:
        w ← w + y * x
        """
        converged = False

        while not converged:
            converged = True  # Asumimos que no hay errores al principio

            # Recorremos cada ejemplo uno por uno
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)     # Clasificación actual
                true_label = nn.as_scalar(y)            # Etiqueta real

                # Si la predicción es incorrecta, actualizamos pesos
                if prediction != true_label:
                    # y * x es la dirección de actualización
                    direction = nn.Constant(true_label * x.data)
                    self.w.update(direction, 1.0)       # w ← w + direction * 1.0
                    converged = False                   # Hubo un error, seguimos entrenando


class RegressionModel(object):
    def __init__(self):
        """
        Red neuronal con UNA sola capa oculta.
        Arquitectura: Entrada escalar (1 dimensión) → 256 neuronas ocultas → salida escalar (1).
        Esta estructura es suficientemente expresiva para aproximar funciones no lineales como sin(x).
        """

        self.input_size = 1         # Cada x es un número real (1 dimensión)
        self.hidden_size = 256      # Capa oculta con 256 neuronas (ancha para compensar tener solo una capa)
        self.output_size = 1        # La salida también es un escalar real (predicción de y = sin(x))

        # Definimos los parámetros entrenables (pesos y bias) para cada capa
        # w1: matriz de pesos (1 x 256) que conecta la entrada con la capa oculta
        self.w1 = nn.Parameter(self.input_size, self.hidden_size)

        # b1: vector de sesgo (1 x 256), uno por cada neurona oculta
        self.b1 = nn.Parameter(1, self.hidden_size)

        # w2: matriz de pesos (256 x 1) que conecta la capa oculta con la salida
        self.w2 = nn.Parameter(self.hidden_size, self.output_size)

        # b2: sesgo (1 x 1) para la única neurona de salida
        self.b2 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Ejecuta una pasada hacia adelante (forward pass) del modelo.
        Entrada:
            x: nodo de entrada (batch_size x 1)
        Salida:
            nodo con predicción (batch_size x 1)
        """

        # Primera capa: aplicamos transformación lineal + ReLU
        # Linear(x, w1): multiplicación de matriz para proyectar x en espacio de 256 dimensiones
        # AddBias(..., b1): sumamos un sesgo a cada una de esas 256 activaciones
        # ReLU: activación no lineal que hace que la red pueda aproximar funciones no lineales
        h = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))

        # Capa de salida: solo lineal (sin ReLU)
        # Queremos que la salida pueda ser positiva o negativa, por eso NO usamos ReLU aquí
        # Linear(h, w2) reduce de (batch_size x 256) a (batch_size x 1)
        return nn.AddBias(nn.Linear(h, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Calcula la pérdida cuadrática media (MSE) entre la predicción y el valor real.
        Inputs:
            x: entradas del batch (batch_size x 1)
            y: etiquetas reales (batch_size x 1)
        Output:
            nodo de pérdida escalar
        """
        prediction = self.run(x)                  # Generamos la predicción del modelo
        return nn.SquareLoss(prediction, y)       # Calculamos la MSE entre predicción y realidad

    def train(self, dataset):
        """
        Entrena el modelo usando descenso por gradiente por batches.
        Recorre el conjunto de datos infinitamente hasta que la pérdida cae por debajo de 0.02.
        """

        learning_rate = 0.01    # Tasa de aprendizaje (ajuste pequeño en cada paso para evitar inestabilidad)
        batch_size = 5          # Mini-batch pequeño para capturar bien los detalles de la función sin(x)

        while True:
            # Iteramos sobre los datos en batches de 5 ejemplos
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Calculamos la pérdida para el batch actual
                loss = self.get_loss(x_batch, y_batch)

                # Calculamos los gradientes de la pérdida respecto a todos los parámetros del modelo
                grads = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                # Actualizamos los pesos y sesgos con descenso por gradiente
                # Regla: nuevo_valor ← actual - learning_rate * gradiente
                self.w1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.w2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)

            # Verificamos si la pérdida del último batch ya es suficientemente baja
            # nn.as_scalar convierte el nodo escalar en un float de Python
            if nn.as_scalar(loss) < 0.02:
                break  # Fin del entrenamiento



class DigitClassificationModel(object):
    def __init__(self):
        """
        Red neuronal para clasificación de dígitos (MNIST).
        Arquitectura: 784 → 256 → 128 → 10
        Usamos ReLU en capas ocultas y softmax en la pérdida.
        """

        self.input_size = 784       # 28x28 píxeles planos → 784
        self.hidden1 = 256          # Primera capa oculta: 256 neuronas
        self.hidden2 = 128          # Segunda capa oculta: 128 neuronas
        self.output_size = 10       # 10 clases posibles (dígitos del 0 al 9)

        # Parámetros de la capa 1 (784 → 256)
        self.w1 = nn.Parameter(self.input_size, self.hidden1)
        self.b1 = nn.Parameter(1, self.hidden1)

        # Parámetros de la capa 2 (256 → 128)
        self.w2 = nn.Parameter(self.hidden1, self.hidden2)
        self.b2 = nn.Parameter(1, self.hidden2)

        # Parámetros de la capa de salida (128 → 10)
        self.w3 = nn.Parameter(self.hidden2, self.output_size)
        self.b3 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Realiza una pasada hacia adelante (forward pass).
        Entrada:
            x: nodo con datos de entrada (batch_size x 784)
        Salida:
            nodo con puntuaciones (batch_size x 10), sin aplicar softmax aún
        """
        # Capa 1: Linear + Bias + ReLU
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))

        # Capa 2: Linear + Bias + ReLU
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))

        # Capa de salida: solo Linear + Bias (sin ReLU porque usamos softmax en la pérdida)
        output = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Calcula la pérdida de clasificación con softmax.
        Inputs:
            x: entradas (batch_size x 784)
            y: etiquetas reales en one-hot (batch_size x 10)
        """
        scores = self.run(x)                # Puntuaciones sin normalizar
        return nn.SoftmaxLoss(scores, y)   # Pérdida softmax + cross-entropy

    def train(self, dataset):
        """
        Entrena la red hasta que la precisión de validación supere el 97.5%
        (el autograder requiere al menos 97% en el conjunto de test).
        """

        learning_rate = 0.1    # Tasa de aprendizaje rápida pero estable
        batch_size = 100       # Batch grande para estabilizar la clasificación

        while True:
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Calculamos la pérdida actual para el batch
                loss = self.get_loss(x_batch, y_batch)

                # Obtenemos los gradientes de la pérdida con respecto a todos los parámetros
                grads = nn.gradients(loss, [self.w1, self.b1,
                                            self.w2, self.b2,
                                            self.w3, self.b3])

                # Aplicamos descenso por gradiente (update = -learning_rate * gradiente)
                self.w1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.w2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)
                self.w3.update(grads[4], -learning_rate)
                self.b3.update(grads[5], -learning_rate)

            # Paramos cuando la precisión de validación es suficientemente alta
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= 0.975:  # Nos damos margen extra sobre el 97%
                break



class LanguageIDModel(object):
    def __init__(self):
        # --- Tamaños de capas ---
        self.hidden_size = 128              # Tamaño del estado oculto (dimensión del vector que "recuerda" letras anteriores)
        self.char_embedding_size = 47       # Tamaño de la entrada por letra (vectores one-hot de tamaño 47)
        self.num_languages = 5              # Número de idiomas posibles a predecir

        # --- Parámetros compartidos para la RNN ---
        self.W_input = nn.Parameter(self.char_embedding_size, self.hidden_size)  # Pesa cada letra individual
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)         # Pesa el estado oculto anterior
        self.b_hidden = nn.Parameter(1, self.hidden_size)                        # Bias compartido en cada paso recurrente

        # --- Parámetros de la capa de salida (categorización por idioma) ---
        self.W_output = nn.Parameter(self.hidden_size, self.num_languages)       # Pesa el último estado oculto para generar logits
        self.b_output = nn.Parameter(1, self.num_languages)                      # Bias de salida

        # --- Hiperparámetro de entrenamiento ---
        self.learning_rate = 0.2  # Tasa de aprendizaje para el descenso por gradiente

    def f_initial(self, x):
        """
        Calcula el primer estado oculto a partir de la primera letra.
        Usa una capa lineal seguida de una ReLU.
        """
        return nn.ReLU(nn.AddBias(nn.Linear(x, self.W_input), self.b_hidden))

    def f_recurrent(self, h_prev, x):
        """
        Calcula el nuevo estado oculto combinando:
        - El estado oculto anterior h_prev
        - La nueva letra x
        Ambos se proyectan con sus respectivas matrices de pesos y se suman.
        Se aplica ReLU a la suma final.
        """
        x_part = nn.Linear(x, self.W_input)           # Transformación de la nueva letra
        h_part = nn.Linear(h_prev, self.W_hidden)     # Transformación del estado oculto anterior
        return nn.ReLU(nn.AddBias(nn.Add(x_part, h_part), self.b_hidden))  # ReLU(x @ W_in + h @ W_hidden + b)

    def run(self, xs):
        """
        Ejecuta el modelo sobre una lista de letras (cada letra es un vector one-hot).
        xs: lista de nodos (uno por letra), cada uno con shape (batch_size x char_embedding_size)
        Devuelve: logits de clasificación (batch_size x num_languages)
        """
        h = self.f_initial(xs[0])  # Primer estado oculto usando la primera letra

        for x in xs[1:]:           # Procesar el resto de letras con recurrencia
            h = self.f_recurrent(h, x)

        # Capa final: lineal para obtener logits (sin ReLU ni softmax aquí)
        return nn.AddBias(nn.Linear(h, self.W_output), self.b_output)

    def get_loss(self, xs, y):
        """
        Calcula la pérdida de clasificación comparando las predicciones con las etiquetas reales.
        """
        logits = self.run(xs)                      # Obtener las predicciones
        return nn.SoftmaxLoss(logits, y)           # Comparar con etiquetas reales

    def train(self, dataset):
        """
        Entrena el modelo usando descenso por gradiente por batches.
        Itera sobre las épocas y actualiza parámetros hasta lograr buena precisión.
        """
        for epoch in range(30):  # Máximo 30 épocas (suele converger antes)
            for xs, y in dataset.iterate_once(32):  # Usamos batches de tamaño 32
                loss = self.get_loss(xs, y)         # Calculamos la pérdida para el batch

                # Calculamos los gradientes respecto a todos los parámetros
                gradients = nn.gradients(loss, [self.W_input, self.W_hidden, self.b_hidden,
                                                self.W_output, self.b_output])

                # Descenso por gradiente: actualizamos cada parámetro
                for param, grad in zip([self.W_input, self.W_hidden, self.b_hidden,
                                        self.W_output, self.b_output], gradients):
                    param.update(grad, -self.learning_rate)

            # Verificamos la precisión de validación después de cada época
            acc = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1} - Validation accuracy: {acc:.4f}")

            # Parada temprana si se alcanza al menos un 89% de precisión
            if acc >= 0.89:
                break
