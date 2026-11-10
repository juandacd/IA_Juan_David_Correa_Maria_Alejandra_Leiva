"""Modelos de la Parte 1 escritos desde cero (solo numpy): Naive Bayes categórico y HMM gaussiano."""
import numpy as np


class NaiveBayesCategorico:
    """Naive Bayes escrito desde cero. Solo usa numpy (sin sklearn para el modelo)."""

    def __init__(self, n_clases=3, alpha=1.0):
        self.n_clases = n_clases   # 3 categorías: Buena, Moderada, Alerta
        self.alpha = alpha         # suavizado de Laplace: suma 1 a cada conteo para que
                                   # ninguna probabilidad sea exactamente 0

    def fit(self, X, y):
        """APRENDE: cuenta cuántas veces pasa cada cosa en los datos de entrenamiento."""
        X, y = np.asarray(X, int), np.asarray(y, int)
        n_muestras, n_pistas = X.shape

        # Cuántos valores distintos puede tomar cada pista (ej. día de la semana: 7)
        self.n_valores = [X[:, j].max() + 1 for j in range(n_pistas)]

        # PROBABILIDAD INICIAL de cada clase: qué tan frecuente es cada categoría de mañana
        conteo_clases = np.bincount(y, minlength=self.n_clases) + self.alpha
        self.log_prior = np.log(conteo_clases / conteo_clases.sum())
        # (usamos logaritmos porque multiplicar muchas probabilidades pequeñas se vuelve
        #  casi 0; en logaritmos las multiplicaciones se convierten en sumas)

        # PROBABILIDAD DE CADA PISTA dada la clase: una tabla por pista
        self.log_lik = []
        for j in range(n_pistas):
            # tabla[c, v] = conteo de veces que la pista j valió v cuando mañana fue la clase c
            tabla = np.full((self.n_clases, self.n_valores[j]), self.alpha)
            for c in range(self.n_clases):
                valores_de_la_clase = X[y == c, j]    # valores de la pista en días de clase c
                tabla[c] += np.bincount(valores_de_la_clase, minlength=self.n_valores[j])
            # convertir conteos en probabilidades (cada fila suma 1) y pasar a logaritmo
            tabla = tabla / tabla.sum(axis=1, keepdims=True)
            self.log_lik.append(np.log(tabla))
        return self

    def predict_log_proba(self, X):
        """Puntaje (en logaritmos) de cada clase para cada día."""
        X = np.asarray(X, int)
        # Empezar con la probabilidad inicial de cada clase
        puntaje = np.tile(self.log_prior, (len(X), 1))
        # Sumar el aporte de cada pista (suma de logs = producto de probabilidades)
        for j in range(X.shape[1]):
            puntaje += self.log_lik[j][:, X[:, j]].T
        return puntaje

    def predict_proba(self, X):
        """Probabilidades reales (suman 1) de Buena, Moderada y Alerta para cada día."""
        lp = self.predict_log_proba(X)
        lp = lp - lp.max(axis=1, keepdims=True)    # truco numérico para evitar errores
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predicción final: la clase con mayor puntaje."""
        return self.predict_log_proba(X).argmax(axis=1)


#HMM desde cero
class HMMGaussiano:
    """HMM gaussiano escrito desde cero (numpy).
    Idea: el aire tiene un ESTADO OCULTO (ej. limpio / medio / contaminado) que no vemos.
    Lo único que vemos es el PM2.5 de cada día. El modelo aprende los estados solo."""

    def __init__(self, n_estados=3, n_iter=200, tol=1e-6):
        self.K = n_estados        # cuántos estados ocultos (3, como las 3 categorías)
        self.n_iter = n_iter      # máximo de vueltas de aprendizaje
        self.tol = tol            # si la mejora es menor a esto, se detiene

    def _emision(self, obs):
        """Para cada día y estado: qué tan probable es ver ese PM2.5 si el aire estuviera en ese estado."""
        B = np.ones((len(obs), self.K))        # los días sin dato valen 1 = "no aportan información"
        ok = ~np.isnan(obs)
        for k in range(self.K):
            # campana de Gauss con media mu[k] y varianza var[k]
            B[ok, k] = (np.exp(-0.5 * (obs[ok] - self.mu[k]) ** 2 / self.var[k])
                        / np.sqrt(2 * np.pi * self.var[k]))
        return np.maximum(B, 1e-300)           # evita ceros exactos

    def _forward(self, B):
        """Hacia adelante: P(estado de hoy | todo lo observado hasta hoy). Solo mira el pasado."""
        T = len(B)
        alpha = np.zeros((T, self.K))          # probabilidad de cada estado cada día
        c = np.zeros(T)                        # factores de escala (evitan números minúsculos)
        a = self.pi * B[0]; c[0] = a.sum(); alpha[0] = a / c[0]
        for t in range(1, T):
            # (estado de ayer) x (probabilidad de cambiar de estado) x (qué tan bien explica el dato de hoy)
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum(); alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B, c):
        """Hacia atrás: usa el futuro de los datos de entrenamiento para aprender mejor."""
        T = len(B)
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
        return beta

    def fit(self, obs):
        """APRENDE los parámetros con el algoritmo Baum-Welch (repite: estimar estados -> ajustar parámetros)."""
        obs = np.asarray(obs, float)
        ok = ~np.isnan(obs)

        # Punto de partida: medias repartidas por cuantiles, y estados "pegajosos" (90 % de quedarse igual)
        self.mu = np.quantile(obs[ok], np.linspace(0.15, 0.85, self.K))
        self.var = np.full(self.K, obs[ok].var())
        self.A = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(self.A, 0.9)          # A[i, j] = probabilidad de pasar del estado i al j
        self.pi = np.full(self.K, 1 / self.K)  # probabilidad del estado del primer día

        ll_prev = -np.inf
        for _ in range(self.n_iter):
            # --- Paso E: estimar qué tan probable es cada estado cada día ---
            B = self._emision(obs)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            ll = np.log(c).sum()                              # qué tan bien explican los parámetros los datos
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)         # P(estado k el día t | todos los datos)
            xi = (alpha[:-1, :, None] * self.A[None] * (B[1:] * beta[1:])[:, None, :]
                  / c[1:, None, None]).sum(axis=0)            # cuántas veces se espera pasar de i a j

            # --- Paso M: actualizar los parámetros con esas probabilidades ---
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi / xi.sum(axis=1, keepdims=True)
            w = gamma[ok]
            self.mu = (w * obs[ok, None]).sum(0) / w.sum(0)
            self.var = np.maximum((w * (obs[ok, None] - self.mu) ** 2).sum(0) / w.sum(0), 1e-3)

            if abs(ll - ll_prev) < self.tol:                  # ya no mejora: parar
                break
            ll_prev = ll

        # Ordenar los estados de menor a mayor contaminación (0 = el más limpio)
        o = np.argsort(self.mu)
        self.mu, self.var, self.pi = self.mu[o], self.var[o], self.pi[o]
        self.A = self.A[np.ix_(o, o)]
        self.loglik_ = ll
        return self

    def filtrar(self, obs):
        """P(estado de hoy) para cada día, usando SOLO datos hasta ese día (sin mirar el futuro)."""
        alpha, _ = self._forward(self._emision(np.asarray(obs, float)))
        return alpha

    def viterbi(self, obs):
        """Secuencia de estados más probable de todo el período (sirve para graficar)."""
        B = np.log(self._emision(np.asarray(obs, float)))
        T = len(B)
        logA = np.log(self.A + 1e-300)
        delta = np.zeros((T, self.K)); psi = np.zeros((T, self.K), int)
        delta[0] = np.log(self.pi + 1e-300) + B[0]
        for t in range(1, T):
            m = delta[t - 1][:, None] + logA      # mejor camino que llega a cada estado
            psi[t] = m.argmax(0)                  # de qué estado venía
            delta[t] = m.max(0) + B[t]
        camino = np.zeros(T, int)
        camino[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):            # reconstruir el camino hacia atrás
            camino[t] = psi[t + 1, camino[t + 1]]
        return camino
