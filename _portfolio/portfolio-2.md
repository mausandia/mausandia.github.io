---
title: "Análisis de los Optimizadores de Redes Neuronales con TensorFlow"
excerpt: "Este es un proyecto para analisar los optimizadores más utilizado al entrenar redes neuronales con TensorFlow<br/><img src='/images/tf.png'>"
collection: portfolio
---
<h1 style="text-align:center">Optimización$^{[1]}$</h1>
<p style="text-align:justify">
La optimización es parte de la vida. En nuestra vida cotidiana, tomamos decisiones que creemos que pueden maximizar o minimizar nuestro conjunto de objetivos. Incluso el proceso de evolución en la naturaleza revela que sigue un proceso de optimización. 
</p>
<p style="text-align:justify">
El origen de una primera teoría de optimización nació el 4 de junio de 1694, cuando John Bernoulli planteó el problema Brachistochrone (en griego, “tiempo más corto”) y desafió públicamente al mundo matemático a resolverlo. El problema planteado fue: "¿Cuál es una trayectoria de deslizamiento por la que un objeto sin fricción se deslizaría en el menor tiempo posible?"
</p>
<h5 style="text-align:center"> Cálculo de variaciones. Ecuación de Euler-Lagrange</h5>
$$J[u] = \int_a^b L(x, u, u')dx$$
<p style="text-align:justify">
"Since the fabric of the universe is most perfect, and is the work of a
most wise Creator, nothing whatsoever takes place in the universe in
which some form of maximum and minimum does not appear."
</p>
<p style="text-align:right">
–Leonard Euler
</p>

<h1 style="text-align:center">Optimización en Inteligencia Artificial $^{[2]}$</h1>
<p style="text-align:justify">
En la inteligencia artificial,  la optimización de funciones es una área fundamental y sus técnicas de métodos númericos son utilizadas ampliamente.
</p>
<p style="text-align:justify">
    En específico, para el aprendizaje de máquina se utiliza una base de datos como entrentamiento. El APRENDIZAJE es una forma de optimización que se utiliza para predecir cualquier tipo de evento o variable.
</p>
<p style="text-align:justify">
    La optimización de funciones involucra tres elementos: 
</p>
<ol>
    <li> Input $x$. Valores de entrada de la función a evaluar </li>
    <li> Función $f(x)$. Función objetivo que evalúa el dominio, modelo. </li>
    <li> Error o Costo. Resultado de la evaluación de un elemento del dominio con la función. 
</ol>
<p style="text-align:justify">
    El espacio de búsqueda es todo el conjunto parámetros de la función $f(x)$ que minimizan la función de costo $y$.
</p>

<h1 style="text-align:center">Aprendizaje con cálculos analíticos, e.g. Regresión Lineal $^{[3]}$</h1>
<p style="text-align:justify">
Sea $X^T$ un conjunto de datos $i.i.d. \backsim \mathcal{N} \left(\mu, \sigma^2\right) $ , $\hat{y}$ la respueta del modelo con el parámetro $\beta$.
</p>
$$\hat{y} =\beta_0 + \sum_{j=1}^pX_j\hat{\beta_j}= X^T\hat{\beta}$$
Para entrenar el modelo se pueden utilizar distintos métodos. 
<h4 style="text-align:center">Mínimos Cuadrados</h4>
Se selecionan los parámetros del modelo minimizando la función de error definida por la suma de los residuos al cuadrado (RSS, residual sum of squares)
$$RSS(\beta) = \sum_{i=1}^N\left(y_i-x_i^T\beta\right)^2=\left(y-X\beta\right)^T\left(y-X\beta\right)$$
<br>
Es una función cuadrática de los parámetros; entonces, siempre existe un mínimo. Diferenciando respecto a $\beta$ se obtienen las ecuaciones normales
$$X^T(y-X\beta) = 0$$
Si $X$ es no-singular, la única solución, está dada por
$$\hat{\beta} = \left(X^TX\right)^{-1}X^Ty$$
<br>
Recordemos que por el teorema de Gauss-Markov, los parámetros del modelo de regresión lineal es el de menor varianza, si cumple algunos principios: no colonialidad, independencia, homocedasticidad, no covarianza, los errores siguen una distribución normal con un media cero y varianza constante.

<h4 style="text-align:center">Estimación de Máxima Verosimilitud con error Gaussiano $^{[3]}$</h4>
<p style="text-align:justify">
Entrenar el modelo con distribuciones de probabilidad condicional que permitan calcular los parámetros $w$ con mayor probabilidad describan los datos del proceso que siguen una ecuación lineal:
</p>
$$y = x^Tw + \epsilon$$
<br>
donde $\epsilon \in \Re$ es una variable aleatoria que describe el ruido. Se puede inferir que la variable sigue una distribución Gaussiana con varianza $\sigma^2$ con media$\mu = 0$.
Entonces, la probabilidad de condicional es:
$$y_n|\mu, \sigma^2 \backsim \mathcal{N} \left(y_n | 0, \sigma^2\right) $$
<br>
La función de la densidad de probabilidad asociada con esta distribución condicional es de la familia de las Gaussianas univariadas:
$$Pr\left(y_n|\mathbf{x}_n, \mathbf{w}, \sigma^2\right) = \frac{1}{\sigma\sqrt{2\pi}}exp\left[-\frac{1}{2\sigma^2}(y_n-\mathbf{x}_n^t\mathbf{w})^2\right]$$

Cada observación tiene un ruido asociado $\epsilon_n$, y estas variables son $i.i.d$, entonces la función de versosimilitud es un producto de las distribuciones de cada término:
$$Pr\left([y_n]_{n=1}^N|[\mathbf{x}]_{n=1}^N, \mathbf{w}, \sigma^2\right) = \prod_{n=1}^N \frac{1}{\sigma\sqrt{2\pi}}exp\left[-\frac{1}{2\sigma^2}(y_n-\mathbf{x}_n^t\mathbf{w})^2\right]$$
Es posible expresar la versosimilitud como una disribución Gaussiana de dimensión-D, $Pr\left(z|\mathbf{\mu}, \mathbf{\Sigma}\right)$. La matriz de covarianza $\mathbf{\Sigma}$ debe ser cuadrada, simétrica y definida positivamente.
<br>
Entonces
$$Pr\left(y\vert\mathbf{X}, \mathbf{w}, \sigma^2\right) = \mathcal{N} \left(y | \mathbf{Xw}, \sigma^2\mathbf{I}\right) =$$
$$(2\sigma^2\pi)^{-N/2}exp\left[-\frac{1}{2\sigma^2}(\mathbf{Xw}-y)^T(\mathbf{Xw}-y)\right]$$
Para encontrar el máxima verosimilitud, se extrae el algoritmo de la expresión anterior
$$log Pr\left(y|\mathbf{X}, \mathbf{w}, \sigma^2\right) = -\frac{N}{2}log(2\sigma^2\pi) - \frac{1}{2\sigma^2}(\mathbf{Xw}-y)^T(\mathbf{Xw}-y)$$
Entonces, el problema es encontrar el argumento máximo de la expresión anterior con $\mathbf{w}$
$$\mathbf{w}^{MLE} = arg max_{\mathbf{w}} \left[- \frac{1}{2\sigma^2}(\mathbf{Xw}-y)^T(\mathbf{Xw}-y)\right]$$
Aquí el valor de constante $\frac{1}{2\sigma^2}$ no cambia la solución, cambiando el signo se trata de un problema de optimización, idéntico a mínimos cuadrados.
$$\mathbf{w}^{MLE} = arg min_{\mathbf{w}}(\mathbf{Xw}-y)^T(\mathbf{Xw}-y)$$


## EJEMPLO --- Regresión lineal multiple
## import libriries
import numpy as np
import pandas as pd
import tensorflow as tf
import sympy as sp
import matplotlib.pyplot as plt
import os
import shutil
import seaborn as sns
import pathlib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow import keras
from keras.applications import ResNet50V2
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from keras import layers, optimizers, regularizers
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
import warnings
from PIL import Image
import cv2
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from numpy import arange
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [12, 9] # ancho, alto de figuras
plt.rcParams["font.size"] = 20
## load data set
## https://www.kaggle.com/datasets/aungpyaeap/fish-market/download?datasetVersionNumber=2
## https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download
fishData = pd.read_csv("Fish.csv")
print(f"Las columnas de la tabla son: {list(fishData.columns)}")
print(f"El tamaño de la tabla es de {fishData.shape[1]} columnas y de {fishData.shape[0]} filas")
x = fishData[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = fishData['Weight']
# sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
#statsmodels
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

<h1 style="text-align:center">Redes Neuronales $^{[3]}$</h1>
<h3 style="text-align:center">Modelo de aprendizaje</h3>
<p style="text-align:justify">
Modelo que procesa la información de forma semejante a la respuesta química-eléctrica de una neurona.
</p>
<p style="text-align:justify">
Modelo estadístico semejante a Projection pursuit regression.
Se generan $Z_M$ elementos creados por combinaciones lineales de entrada, y a su vez, la función objetivo $f_k(X)$ es modelada con una función de combinaciones lineales de $Z_M$
</p>
$$Z_m = \sigma\left( \alpha_{0m}+ \alpha_m^TX \right), m = 1, \dots, M,$$
$$T_k = \beta_{0k} + \beta_k^TZ, k = 1, \dots, K,$$
$$f_k(X) = g_k(T), k = 1, \dots, K,$$
<br>
La función de activación $\sigma(v)$ se escoge con base en el problema 
<br>
<ol>
    <li> $\mathcal{Sigmoide}$  $\sigma(v) = 1/(1+e^{-v})$ </li>
    <li> $\mathcal{ReLu}$  $\sigma(v) = max{0,z}$ </li>
    <li> $\mathcal{Softmax}$ $\sigma(v) = \frac{e^{v_k}}{\sum_{l=1}^K e^{vl}}$ </li>
</ol>

Las unidades en medio de la red calculadas con las características $Z_m$ son llamadas capas ocultas porque no son directamente observables. Mientras que en la capa externa son la de ingreso de datos y salida del modelo.

<h3 style="text-align:center">Modelo estadístico - Projection pursuit regression $^{[3]}$</h3>
Modelo estadístico no-paramétrico que consiste en combinaciones lineales de funciones de Ridge en un espacio $\Re^p$, que genera un conjunto grande de modelos.
$$f\left(X\right) = \sum_{m=1}^M g_m\left(\omega_m^TX\right)$$
<br>
donde $\omega_m$, $m = 1, 2,  \dots, M,$ es el vector-p unitario de parámetros desconocidos. Además, $V_m=\omega_m^TX$ es la variable escalar de la proyección de X en el vector unitario $\omega_m$ 
Si el número de parámetros en el modelo aumenta lo suficiente, los modelos son aproximadores universales. Sin embargo, el modelo es útil solo para predicción porque no produce un modelo interpretable y entendible que describan los datos.
<br>
Lo que se quiere lograr es aproximar los mínimos de la función de error
$$\sum_{i=1}^N\left[y_i-\sum_{m=1}^Mg_m(\omega_m^Tx_i)\right]^2$$
<br>
sobre las funciones agregadas $g_m$ y la dirección de vectores $\omega_m$. 

<h1 style="text-align:center">Entrenar redes neuronales $^{[3]}$</h1>
<p style="text-align:justify">
El modelo de redes neuronales tiene parámetros desconocidos, llamados $\mathcal{pesos}$, durante el entrenamiento con los datos, se buscan estos valores. 
</p>
$${\alpha_{0m}, \alpha_m; m = 1, 2, \dots, M}   M(p+1) \mathcal{pesos}$$
$${\beta_{0k}, \beta_k; m = 1, 2, \dots, M}  K(M+1) \mathcal{pesos}$$
<p style="text-align:justify">
Por otra parte, las funciones típicas de error o pérdida son: 
</p>
$$R(\theta)=\sum_{i=1}^N \sum_{k=1}^K(y_{ik}-f_k(x_i))^2$$ 
los errores al cuadrado para regresión.
$$R(\theta)=-\sum_{i=1}^N \sum_{k=1}^Ky_{ik}logf_k(x_i)$$ 
la entropía cruzada para clasificación.

<h1 style="text-align:center"> Algoritmos de optimización $^{[2]}$</h1>
<p style="text-align:justify">
    Como hemos visto en el curso, existen diversos métodos númericos para minimizar la función de pérdida $R(\theta)$, en un contexto de generalidad no se busca el mínimo global para evitar sobre entrenamiento (Bias-Variance Treatoff). 
    Por otra parte, en ocasiones es posible que el problema de optimización se complique porque la función de pérdida es no convexa y/o inestable, o simplemente es muy compleja para calcular su derivada. Una forma de clasificar los algoritmos de optimización es con base en si la funciones objetivo son diferenciables o no.
</p>
    
<ul>
    <li>Diferenciable</li>
        <ul>
            <li> Algoritmo Bracketing</li>
                 <ol>
                    <li> Algoritmo de Fibonacci</li>
                    <li> Algoritmo de busqueda por sección de oro (Golden Section Search)</li>
                    <li> Algoritmo de bisección</li>
                </ol>
            <li> Algoritmo descendente local</li>
                 <ol>
                    <li> Algoritmo de búsqueda lineal</li>
                </ol>
        </ul>  
</ul>
<br>


<h1 style="text-align:center"> Algoritmos de optimización $^{[2]}$</h1>
<ul>
    <li>Diferenciable</li>
        <ul>
            <li> Algoritmo de primer orden</li>
                 <ol>
                    <li> Algoritmo de gradiente descendente</li>
                    <li> Algoritmo de Momento</li>
                    <li> Algoritmo de Adagrad</li>
                    <li> Algoritmo de RMSProp</li>
                    <li> Algoritmo de Adam</li>
                </ol>
            <li> Algoritmo de segundo orden</li>
                 <ol>
                    <li> Algoritmo del método de Newton</li>
                    <li> Algoritmo del método de secante</li>
                    <li> Algoritmos Quasi-Newton</li>
                         <ol>
                            <li> Algoritmo de Davidson-Fletcher-Powell</li>
                            <li> Algoritmo de BFGS (Broyden, Fletcher, Goldfarb, and Shanno Algorithm)</li>
                            <li> Algoritmo de Limited-memory BFGS (L-BFGS)</li>
                        </ol>
                </ol>
        </ul>  
</ul>
<br>
<p style="text-align:justify">
Los algoritmos de gradiente descendente son muy populares, ya que son algoritmos muy sencillos y efectivos; además, proveen las bases para extender o modificar el algoritmo para mayor desempeño. Existen diversas variantes que se implementan en paqueterías de python sobre deep learning como TensorFlow o Pytorch.
</p><h1 style="text-align:center"> Algoritmos de optimización $^{[2]}$</h1>
<ul>
    <li>Diferenciable</li>
        <ul>
            <li> Algoritmo de primer orden</li>
                 <ol>
                    <li> Algoritmo de gradiente descendente</li>
                    <li> Algoritmo de Momento</li>
                    <li> Algoritmo de Adagrad</li>
                    <li> Algoritmo de RMSProp</li>
                    <li> Algoritmo de Adam</li>
                </ol>
            <li> Algoritmo de segundo orden</li>
                 <ol>
                    <li> Algoritmo del método de Newton</li>
                    <li> Algoritmo del método de secante</li>
                    <li> Algoritmos Quasi-Newton</li>
                         <ol>
                            <li> Algoritmo de Davidson-Fletcher-Powell</li>
                            <li> Algoritmo de BFGS (Broyden, Fletcher, Goldfarb, and Shanno Algorithm)</li>
                            <li> Algoritmo de Limited-memory BFGS (L-BFGS)</li>
                        </ol>
                </ol>
        </ul>  
</ul>
<br>
<p style="text-align:justify">
Los algoritmos de gradiente descendente son muy populares, ya que son algoritmos muy sencillos y efectivos; además, proveen las bases para extender o modificar el algoritmo para mayor desempeño. Existen diversas variantes que se implementan en paqueterías de python sobre deep learning como TensorFlow o Pytorch.
</p>

<h1 style="text-align:center"> Algoritmos de optimización $^{[2]}$</h1>
<p style="text-align:justify">
    Algoritmos de optimización no tan númericos.
    </p>
<ul>
    <li>No Diferenciable</li>
        <ul>
         <li> Algoritmos directos</li>
             <ol>
                    <li> Algoritmo Nelder-Mead</li>
                    <li> Algoritmo de coordenadas cíclicas (Cyclic Coordinate Search)</li>
                    <li> Algoritmo del método de Powell</li>
                    <li> Algoritmo del método Hooke-Jeeves</li>
             </ol>
    <li> Algoritmos estocásticos</li>
         <ol>
                    <li> Algoritmo de recocido simulado (Simulated Annealing algorithm)</li>
                    <li> Algoritmo de estrategias de evolución (Evolution strategies algorithm)</li>
                    <li> Algoritmo del método de entropía cruzada</li>
                    <li> Algoritmo escalada de colinas estocásticas (Stochastic Hill climbing algorithm)</li>
                    <li> Algoritmo Búsqueda local iterada (Iterated Local Search algorithm)</li>
             </ol>
    <li> Algoritmos poblacionales</li> 
         <ol>
                    <li> Algoritmo genéticos (genetic algorithm)</li>
                    <li> Algoritmo de evolución diferencial (Differential evolution algorithm)</li>
                    <li> Algoritmo de optimización de enjambre de partícula (Particle Swarm Optimization)</li>
             </ol>
    </ul>
    
    
</ul>

The Nelder-Mead optimization algorithm is a widely used approach for non-differentiable objective functions. As such, it is generally referred to as a pattern search algorithm and is used as a local or global search procedure, challenging nonlinear and potentially noisy and multimodal function optimization problems.

The Broyden, Fletcher, Goldfarb, and Shanno, or BFGS Algorithm, is a local search optimization algorithm. It is a type of second-order optimization algorithm, meaning that it makes use of the second-order derivative of an objective function and belongs to a class of algorithms referred to as Quasi-Newton methods that approximate the second derivative (called the Hessian) for optimization problems where the second derivative cannot be calculated. The BFGS algorithm is perhaps one of the most widely used second-order algorithms for numerical optimization and is commonly used to Ąt machine learning algorithms such as the logistic regression algorithm.


Stochastic Hill climbing is an optimization algorithm. It makes use of randomness as part of the search process. This makes the algorithm appropriate for nonlinear objective functions where other local search algorithms do not operate well. It is also a local search algorithm, meaning that it modiĄes a single solution and searches the relatively local area of the search space
until the local optima is located. This means that it is appropriate on unimodal optimization problems or for use after the application of a global optimization algorithm

Iterated Local Search is a stochastic global optimization algorithm. It involves the repeated application of a local search algorithm to modiĄed versions of a good solution found previously. 
In this way, it is like a clever version of the stochastic hill climbing with random restarts algorithm. The intuition behind the algorithm is that random restarts can help to locate many local optima in a problem and that better local optima are often close to other local optima. Therefore modest perturbations to existing local optima may locate better or even best solutions
to an optimization problem.

The genetic algorithm is a stochastic global optimization algorithm. It may be one of the most popular and widely known biologically inspired algorithms, along with artiĄcial neural networks. The algorithm is a type of evolutionary algorithm and performs an optimization procedure inspired by the biological theory of evolution by means of natural selection with a binary
representation and simple operators based on genetic recombination and genetic mutations.

Evolution strategies is a stochastic global optimization algorithm. It is an evolutionary algorithm related to others, such as the genetic algorithm, although it is designed speciĄcally for continuous function optimization.

Differential evolution is a heuristic approach for the global optimization of nonlinear and non-differentiable continuous space functions. The differential evolution algorithm belongs to a broader family of evolutionary computing algorithms. Similar to other popular direct search approaches, such as genetic algorithms and evolution strategies, the differential evolution  algorithm starts with an initial population of candidate solutions. These candidate solutions are iteratively improved by introducing mutations into the population, and retaining the Ąttest candidate solutions that yield a lower objective function value. The differential evolution algorithm is advantageous over the aforementioned popular approaches because it can handle nonlinear and non-differentiable multi-dimensional objective functions, while requiring very few control parameters to steer the minimisation. These characteristics make the algorithm easier and more practical to use.


Simulated Annealing is a stochastic global search optimization algorithm. This means that it makes use of randomness as part of the search process. This makes the algorithm appropriate for nonlinear objective functions where other local search algorithms do not operate well. Like the stochastic hill climbing local search algorithm, it modiĄes a single solution and searches the relatively local area of the search space until the local optima is located. Unlike the hill climbing algorithm, it may accept worse solutions as the current working solution. The likelihood of accepting worse solutions starts high at the beginning of the search and decreases with the progress of the search, giving the algorithm the opportunity to Ąrst locate the region for the global optima, escaping local optima, then hill climb to the optima itself.

<h1 style="text-align:center">Algoritmo de gradiente descendente $^{[2]}$</h1>
<p style="text-align:justify">
Es un algoritmo de optimización de primer orden porque calcula explícitamente la primera derivada de la función de pérdida sobre todo el vector de parámetros. Sea la función de pérdida la suma de los errores al cuadrado.
</p>
$$R(\theta)=\sum_{i=1}^N \sum_{k=1}^K(y_{ik}-f_k(x_i))^2$$
donde 
$f_k(X) = g_k(T),$ $T_k = \beta_{0k} + \beta_k^TZ,$  y a su vez $Z_m = \sigma\left( \alpha_{0m}+ \alpha_m^TX \right),$ 
con derivadas
$$\frac{\partial R_i}{\partial\beta_{km}}= -2\left(y_{ik}-f_k(x_i)\right )g_k'\left(\beta_k^Tz_i\right )z_{mi} = \sigma_{ki}z_{mi}$$
$$\frac{\partial R_i}{\partial\alpha_{ml}}= -\sum_{k=1}^K 2 \left(y_{ik}-f_k(x_i)\right )g_k'\left(\beta_k^Tz_i\right )\beta_{km}\sigma'\left(\alpha_m^Tx_i\right)x_{il} = s_{mi}x_{il} $$
Las cantidades $\sigma_{ki}$ y $s_{mi}$ son proporcionales al error del modelo de redes neuronales, el primero en la salida de capas externas y el segundo en capas escondidas.
<br>
Dado estas derivadas, el valor de los paramétros en la iteración $(r+1)$ tiene una actualización de la forma:
$$\beta_{km}^{(r+1)}=\beta_{km}^{(r)}-\gamma_r\sum_{i=1}^N\frac{\partial R_i}{\partial\beta_{km}^{(r)}}$$
$$\alpha_{ml}^{(r+1)}=\alpha_{ml}^{(r)}-\gamma_r\sum_{i=1}^N\frac{\partial R_i}{\partial\alpha_{ml}^{(r)}}$$
donde $\gamma_r$ es la tasa de aprendizaje. Hiperparámetro que controla qué tanto nos movemos en el espacio de búsqueda en cada iteración.
<br>
Se reescribe el valor de $s_{mi}$ como
$$s_{mi}= \sigma'\left(\alpha_m^Tx_i\right )\sum_{k=1}^K\beta_{km}\sigma_{ki}, $$
conocido como ecuación de propagación hacia atrás o regla-delta.

# objective function
def objective(x):
    return x**2.0
# derivative of objective function
def derivative(x):
    return x * 2.0
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial random point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    solution = solution[0]
    res = [[solution, objective(solution), np.nan]]
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # take a step
        solution = solution - step_size * gradient
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        res.append([solution, solution_eval, solution+step_size * gradient])
        # report progress
        #print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solutions, scores], res
    #return res
# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# perform the gradient descent search
[solutions, scores], res = gradient_descent(objective, derivative, bounds, n_iter, step_size)
#res = gradient_descent(objective, derivative, bounds, n_iter, step_size)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)

###Plot optimización de función x^2
plt.plot(inputs, results, label="$f(x)$")
plt.plot(solutions, scores, '.-', color='red', label="Trayectoria gradiente descendente")
plt.annotate(r"$arg$ $min$ $x$", xy=[0.0, 0], xytext=[0.12, 0.25],
                 arrowprops=dict(arrowstyle='->'),)
plt.legend(loc=(0.160, 0.80));
plt.show()

df = pd.DataFrame(res, index=range(len(res)), columns=["$x_n$", "$f(x_n)$", "$x_n - x_{n-1}$"])
df.style.format(dict(zip(df.columns, ["{:.6g}", "{:+.3e}", "{:+.3e}"])))