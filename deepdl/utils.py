import tensorflow as tf
import numpy as np
import pandas as pd

# mostly Renés functions, some small changes here and there but nothing big

def model(gammas, Z, betas, X, penalty_mat, y):
    """
    Penalized Least Squares
    """
    # really important to use tensorflow operations instead of numpy:
    # Using gammas.T @ penalty @ gammas instead of tf.matmul
    # caused the trained values not to converge to the pls estimators (when smoothing was applied)
    prediction = X @ betas + Z @ gammas
    penalty = tf.matmul(tf.matmul(tf.transpose(gammas), penalty_mat), gammas)
    sq_diff = tf.reduce_sum(input_tensor=(y - prediction) ** 2)
    loss = tf.add(sq_diff, penalty)
    return loss



def split_formular(formula):
    y_names, x_names = [], []
    # Auslesen der Elemente des dicts (y, X)
    _, temp_x = next(iter(formula.items()))
    # nur den Namen behalten
    temp_x = temp_x.replace(" ", "")
    # Bearbeiten von X, entfernen aller argumente und intercept call
    temp_x = temp_x.split("+")
    return temp_x

def get_model_variables_names(formula):
    """
    Ließt die Model Variablen aus.
    Erstellt Listen der abhängigen und unabhängigen Variablen.

    Parameters
    ----------
    formula: dict
        The dictionary that contains the model equation.
        E.g. {'y': '1 + x2 + spline(x2, type="b", degree="2", nr_knots="20")'}
        is unaffected by spline options! Neat!

    Returns
    -------
    y_names : list of strings
        A list containing all variables used as dependend variable

    x_names : list of strings
        A list containing all variables used as independend variable

    """
    # erweitern um unstrukturierten Teil
    # ISSUE #28
    # Da der Model Input ein dict ist müssen wir erst die Elemente auslesen
    y_names, x_names = [], []
    # Auslesen der Elemente des dicts (y, X)
    temp_y, temp_x = next(iter(formula.items()))
    y_names.append(temp_y)
    # nur den Namen behalten
    temp_x = temp_x.replace(" ", "")
    # Bearbeiten von X, entfernen aller argumente und intercept call
    temp_x = temp_x.replace("(", "")
    temp_x = temp_x.replace(")", "")
    temp_x = temp_x.replace("spline", "")
    temp_x = temp_x.split("+")
    for j in temp_x:
        var_name = j.split(",")[0]
        x_names.append(var_name)
    return y_names, x_names


def get_data_names(data):
    """
    Get the names of all the input variables of the dataset
    """
    temp = list(data.columns)
    return temp


def get_variables_values(data, var_name):
    """
    Auslesen der Spalten-Werte des Inputs zum Correspondierenden Variablen Namen
    Aufpassen, da wir hier den Intercept als 1 notieren, muss eine extra
    Routine implementiert sein, welche hardcoded den Intercept bestimmt.
    """
    # nur die Variablen Namen auslesen, welche KEIN Intercept sind
    temp_name = var_name.copy()
    if temp_name.count("1") > 0:
        temp_name.remove("1")
    temp = data.loc[:, temp_name].values
    return temp




def get_design_matrix(data, formula):
    """
    Berechnung der benötigten Matrizen für den strukturieten Teil der Gleichung

    Parameters
    ----------
    data : numpy-array
        Input Datensatz

    formula : str
        Modellgleichung

    Returns
    -------
    design_m : pandas DataFrame
        Returns the Design Matrix of the underlying (struktured-part of the) Model

    """
    # 1. Schritt: Welche Variablen sind in dem model vorhanden?
    # Bis jetzt nur strukturierte Modelparts unterstützt
    y_name, x_name = get_model_variables_names(formula=formula)
    # 2. Schritt: Welche Variablen sind im Datensatz
    data_var_names = get_data_names(data=data)
    # 3. Schritt: Datenwerte aus dem Datensatz auslesen, welche wir brauchen
    try:
        variable_values = get_variables_values(data=data, var_name=x_name)
    except:
        # hier sollte man noch einfügen, dass genau gezeigt wird welche Variable falsch ist
        print("Variablen in Modellgleichung sind nicht im Datensatz zu finden!")
    # 4. Schritt: Wurde ein intercept spezifiziert?
    if x_name.count("1") > 0:
        intercept = np.ones(shape=(len(variable_values), 1))
        design_m = np.concatenate([intercept, variable_values], axis=1)
        # change into pandas dataframe weil alte Funktionen damit sonst nicht funktionieren.
        # Bennenung der Spalten nach den Variablen, zwecks späterer Referenz.
        design_m = pd.DataFrame(design_m, columns=x_name)
    else:
        design_m = pd.DataFrame(variable_values, columns=x_name)
    return design_m


def find_spl_name(foo):
    spl_name = []
    for fo in foo:
        if fo.find("=") == -1:
            spl_name.append(fo)
    if len(spl_name) == 1:
        return spl_name[0]
    else:
        return str(spl_name)

def get_idx(spl_names):
    names = spl_names.copy()
    for i in range(len(spl_names)):
        if spl_names[i].find("[") != -1:
            names[i] = spl_names[i].replace("[", "")
            names[i] = names[i].replace("]", "")
            names[i] = names[i].replace(" ", "")
            names[i] = names[i].replace("'", "")
            names[i] = names[i].split(",")
    idx = []
    for el in names:
        if type(el) == list:
            idx.extend(el)
        else:
            idx.append(el)
    return idx




