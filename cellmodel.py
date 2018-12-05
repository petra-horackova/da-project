import math

import numpy as np
import pandas as pd
from sklearn import svm
import constants as cons

ALL_INDICES = [[1, 0, 1.],
               [-1, 0, 1.],
               [0, 1, 1.],
               [0, -1, 1.],
               [1, 1, cons.corner_neighbor_weight],
               [1, -1, cons.corner_neighbor_weight],
               [-1, 1, cons.corner_neighbor_weight],
               [-1, -1, cons.corner_neighbor_weight]]


def get_neighbor_indices(i, j, world_shape):
    # funkce která spočítá indexy všech sousedů daný buňky 
    # např. buňka 0, 0 nemá souseda i - 1, j - 1 protože jsou out of bounds
    neighbor_indices = []
    neighbor_weights = []
    for idx in ALL_INDICES:
        neighbor_i = i + idx[0]
        neighbor_j = j + idx[1]
        weight = idx[2]
        # sousedi nesměj ležet mimo hranice tenzoru
        if 0 <= neighbor_i < world_shape[0] and 0 <= neighbor_j < world_shape[1]:
            neighbor_indices.append([neighbor_i, neighbor_j])
            neighbor_weights.append(weight)

    return neighbor_indices, neighbor_weights


def get_features_target(data):
    # opět předpokládáme, že data jsou 24 x M x N numpy.array
    # do listu features budeme ukládát špočítaný ... features :)
    features = []
    # stějne tak target
    target = []
    hour = []

    # pro každou hodinu projdeme všechny buňky
    for h in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                features.append(_get_square_features(h, i, j, data))
                target.append(data[(h + 1) % 24, i, j])  # do targetu přidáme stav z příští hodiny
                hour.append(h)

    # features transforumejeme do pandas dataframu, target do numpy array
    return pd.DataFrame(features), np.array(target), np.array(hour)


def _get_square_features(h, i, j, data):
    """
    nove extrahovana funkce pro dostani features pro jeden ctverec, ktera pocita s weights
    stejna funkce se pouziva pro jednotlivy ctverec pri simulaci
    """
    neighbor_indices, neighbor_weights = get_neighbor_indices(i, j, data.shape[1:])
    indexes = np.array([[h] + n for n in neighbor_indices])
    # neighbor_data = np.take(data[h, :, :], neighbor_indices)
    # print(tuple(indexes.T))
    neighbor_data = data[tuple(indexes.T)]
    neighbor_data = np.multiply(neighbor_data, neighbor_weights)
    # print(neighbor_data)
    return {'current_value': data[h, i, j],  # současná hodnota buňky
            'mean_diff': np.mean(data[h, i, j] - neighbor_data),
            # průměrnej rozdíl buňky a sousedů
            'mean': np.mean(neighbor_data),  # průměr sousedů
            'std': np.std(neighbor_data),  # std sousedů
            'sin_h': math.sin(2 * h * math.pi / 24),  # sin, cos 24-hour cycle  transforamtion
            'cos_h': math.cos(2 * h * math.pi / 24),
            'lat_y': i,
            'lon_x': j
            }


# tady se vezmou features a target a natrénuje se model
# tohle jen nejzákladnější verze, měl by se tam ještě přidat scaling a PCA (principal component analysis)
# a možná řešit hodinu jako kategorickou proměnou
# před trénováním by se měla vyhodit začáteční hodina + validační set
# model by se měl validovat např. pomocí MSE (mean squared error)
class Model:
    def __init__(self) -> object:
        # self._m = LinearRegression()
        self._m = svm.SVR(gamma='auto')

    def fit(self, features, target):
        self._m.fit(features, target)

    def predict(self, features):
        return self._m.predict(features)


# Před použitím třídy Simulation je potřeba vyvtořit instanci třídy Model a natrénovat ji
# na features a target z funkce get_features_target, 
# natrénovanou instanci potom předat instanci týhle třídy přes argument model
class Simulation:

    def __init__(self, initial_state, model, starting_hour, n_steps=23):
        # data modelu by měly být uložený v 3D tenzoru (ideální je použít numpy.array)
        # pokud je "svět" velkej např. 20x20 buněk a chceme modelovat 24 kroků
        # tak tenzor bude mít tvar 24x20x20 - je to vlastně 24 matic naskládanejch za sebou
        # hodnota jedný buňky je počet alertů (časem se to může nějak normalizovat)
        self._n_steps = n_steps
        self._data = np.empty(shape=(n_steps + 1,) + initial_state.shape)
        self._sim_values = []
        self._last_slice = []
        self._starting_hour = starting_hour
        # jako první matici se uloží výchozí stav, ze kterýho se modelují další kroky
        # ostatní matice budou zatím prázdný
        self._data[starting_hour, :, :] = initial_state

        # tady se uloží natrénovanej model (předchozí třída model)
        self._model = model

    def _update_cell(self, step, i, j):
        # tohle je funkce která bude obsahovat pravidla
        # tady musí zopakovat postup z get_features_target
        # pomocí funkce get_neigbor_indices a np.take se spočítaj hodnoty sousedů
        # z nich se vytvoří našich 5 features a ze kterých model vypočítá hodnotu buňky v dalším kroku
        # pozor - když budeme simulovat víc než 23 kroků je potřeba přes modulo přepočítat krok na hodinu v rozsahu 0-23
        # self._sim_values.append(self._data[step, i, j])
        # podminka zajistujici doplneni hodnot pro posledni hodinu v pripade,
        # ze jsme v posledni iteraci (=predposledni patro)
        relative_index = self._starting_hour + step
        write_pos = relative_index + 1
        if relative_index == 23:  # podminka pro spravny zapsani vysledku z h=23 zpatky na zacatek kostky do h=0
            write_pos = 0
        elif relative_index >= 24:
            relative_index = abs(24 - relative_index)
            write_pos = relative_index + 1
        current_features = pd.DataFrame([_get_square_features(relative_index, i, j, self._data)])
        current_features.loc[:, 'h'] = relative_index
        new_target = self._model.predict(current_features)
        self._data[write_pos, i, j] = new_target

    def _move_one_step(self):
        # wrapper kterej spočítá novou hodnotu pro každou buňku i, j
        for i in range(self._data[0].shape[0]):
            for j in range(self._data[0].shape[1]):
                self._update_cell(self._current_step, i, j)

    def run(self):
        # hlavní metoda, která proběhne celou simulaci
        for h in range(self._n_steps):
            self._current_step = h
            self._move_one_step()
