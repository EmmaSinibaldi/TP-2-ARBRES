#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from sklearn import tree, datasets
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114  # XXX : change
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data4 = rand_checkers(n1, n2, n3, n4, sigma)

#%%
############################################################################
# Displaying labeled data
############################################################################

plt.close("all")
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(142)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(143)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(data4[:, :2], data4[:, 2], w=None)

#%%
############################################
# ARBRES
############################################


# Q2. Créer deux objets 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.

# dt_entropy = TODO
# dt_gini = TODO

# Effectuer la classification d'un jeu de données simulées avec rand_checkers des échantillons de
# taille n = 456 (attention à bien équilibrer les classes)

# data = TODO
n_samples = len(data)
# X = TODO
# Y = TODO and careful with the type (cast to int)

dt_gini.fit(X, Y)
dt_entropy.fit(X, Y)

print("Gini criterion")
print(dt_gini.get_params())
print(dt_gini.score(X, Y))

print("Entropy criterion")
print(dt_entropy.get_params())
print(dt_entropy.score(X, Y))

#%%
# Afficher les scores en fonction du paramètre max_depth

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

plt.figure(figsize=(15, 10))
for i in range(dmax):
    # dt_entropy = ... TODO
    # ...
    # scores_entropy[i] = dt_entropy.score(X, Y)

    # dt_gini = ... TODO
    # ...
    # scores_gini[i] = TODO

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X, Y, step=50, samples=False)
plt.draw()


plt.figure()
# plt.plot(...)  # TODO
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.draw()
print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

#%%
# Q3 Afficher la classification obtenue en utilisant la profondeur qui minimise le pourcentage d’erreurs
# obtenues avec l’entropie

# dt_entropy.max_depth = ... TODO

plt.figure()
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=100)
plt.title("Best frontier with entropy criterion")
plt.draw()
print("Best scores with entropy criterion: ", dt_entropy.score(X, Y))

#%%
# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed
# Voir https://scikit-learn.org/stable/modules/tree.html#classification

# TODO

#%%
# Q5 :  Génération d'une base de test
# data_test = rand_checkers(... TODO
# X_test = ...
# Y_test = ...

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(15, 10))

for i in range(dmax):
    # TODO

#%%
plt.figure()
# plt.plot(...)  # TODO
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.title("Testing error")
print(scores_entropy)

#%%
# Q6. même question avec les données de reconnaissances de texte 'digits'

# Import the digits dataset
digits = datasets.load_digits()

n_samples = len(digits.data)
# use test_train_split rather.

# X = digits.data[:n_samples // 2]  # digits.images.reshape((n_samples, -1))
# Y = digits.target[:n_samples // 2]
# X_test = digits.data[n_samples // 2:]
# Y_test = digits.target[n_samples // 2:]

# TODO

# Q7. estimer la meilleur profondeur avec un cross_val_score

# TODO


