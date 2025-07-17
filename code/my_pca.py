from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class MyPCA(BaseEstimator, TransformerMixin):
    """
    Implémentation personnalisée de l'Analyse en Composantes Principales (PCA).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Calcule les composantes principales à partir des données.

        Paramètres:
        X : array-like, shape (n_samples, n_features)
            Les données d'entrée, où n_samples est le nombre d'échantillons
            et n_features est le nombre de caractéristiques.
        y : ignoré, présent pour la compatibilité avec l'API sklearn

        Retourne:
        self : objet
            Retourne l'instance elle-même.
        """
        # S'assurer que X est de type numpy array
        X = np.array(X)

        # 1. Centrer les données
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Calculer la matrice de covariance
        n_samples = X.shape[0]
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 3. Calculer les vecteurs propres et valeurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 4. Trier les vecteurs propres par valeurs propres décroissantes
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.components_ = eigenvectors[:, idx]

        # 5. Sélectionner les n_components premiers vecteurs propres
        self.components_ = self.components_[:, :self.n_components]

        return self

    def transform(self, X):
        """
        Applique la réduction de dimensionnalité aux données X.

        Paramètres:
        X : array-like, shape (n_samples, n_features)
            Les données à transformer.

        Retourne:
        X_transformed : array-like, shape (n_samples, n_components)
            Les données transformées.
        """
        # S'assurer que X est de type numpy array
        X = np.array(X)

        # Centrer les données avec la moyenne apprise
        X_centered = X - self.mean_

        # Projeter les données sur les composantes principales
        X_transformed = np.dot(X_centered, self.components_)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Ajuste le modèle avec X
        et applique la réduction de dimensionnalité à X.

        Paramètres:
        X : array-like, shape (n_samples, n_features)
            Les données d'entrée.
        y : ignoré

        Retourne:
        X_transformed : array, shape (n_samples, n_components)
            Les données transformées.
        """
        return self.fit(X).transform(X)
