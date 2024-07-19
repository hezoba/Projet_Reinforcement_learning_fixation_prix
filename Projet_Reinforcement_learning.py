#Importation des bibliothèques

import numpy as np
import pandas as pd
import random
from google.colab import drive
drive.mount('/content/drive')

# Fonction de lecture du fichier Excel
def read_excel_data():
    # Lire et renomer les données du fichier Excel
    data = pd.read_excel('/content/drive/My Drive/Projet_RL/Variation_prix.xlsx',skiprows=1)
    data['profit']=(data['prix_entreprise']-data['cout_entreprise'])*data['qte_vendue']
    data=data.round(decimals=1)

    # Extraire les variables pertinentes
    states = data[['prix_entreprise', 'prix_concurrents', 'demande_marche', 'stock_marche_moyen', 'cout_entreprise', 'stock_entreprise','qte_vendue']]
    actions = ['Augmenter le prix', 'Diminuer le prix', 'Maintenir le prix']
    rewards = data[['profit']]
    dates = data[['date']]

    return states, actions, rewards, dates

# Fonction de définition des états possibles
def define_states(states_data):

    possible_states = []

    # Parcourir chaque ligne des données d'état
    for state in states_data.itertuples():
        # Créer un dictionnaire pour représenter l'état actuel
        current_state = {
            'prix_entreprise': state.prix_entreprise,
            'prix_concurrents': state.prix_concurrents,
            'demande_marche': state.demande_marche,
            'stock_marche_moyen': state.stock_marche_moyen,
            'cout_entreprise': state.cout_entreprise,
            'stock_entreprise': state.stock_entreprise,
            'qte_vendue': state.qte_vendue
        }

        # Ajouter l'état actuel à la liste des états possibles
        possible_states.append(current_state)

    return possible_states

# Fonction de définition des actions possibles
def define_actions():

    actions = ['Augmenter le prix', 'Diminuer le prix', 'Maintenir le prix']

    return actions

# Fonction de calcul de la récompense
def calculate_reward(state, action, next_state, rewards_data):

    # Extraire les valeurs des variables d'état
    prix_entreprise = state['prix_entreprise']
    cout_entreprise = state['cout_entreprise']
    demande_marche = state['demande_marche']
    stock_marche_moyen = state['stock_marche_moyen']
    stock_entreprise = state['stock_entreprise']
    qte_vendue = state['qte_vendue']

    # Extraire la récompense de l'état suivant
    next_state_reward = rewards_data.loc[next_state['date']].values[0]

    # Calculer la part de marché
    part_marche = qte_vendue / demande_marche

    # Calculer le profit
    profit = (prix_entreprise - cout_entreprise) * part_marche * demande_marche

    return profit

# Fonction de définition de la fonction de transition
def define_transition_function(states, actions):
    transition_function = {}

    # Parcourir chaque état possible
    for state in states:
        state_key = tuple(state.items())  # Convertir le dictionnaire d'état en un tuple de paires clé-valeur pour le rendre hashable
        # Créer un dictionnaire pour stocker les transitions à partir de l'état actuel
        state_transitions = {}

        # Parcourir chaque action possible
        for action in actions:
            # Déterminer l'état suivant en fonction de l'action appliquée
            next_state = apply_action(state, action)
            next_state_key = tuple(next_state.items())  # Convertir le dictionnaire d'état suivant en un tuple hashable

            # Vérifier si l'état suivant est un état valide
            if next_state_key in states:
                # Ajouter la transition à l'état actuel
                state_transitions[action] = next_state

        # Ajouter les transitions possibles à partir de l'état actuel au dictionnaire global
        transition_function[state_key] = state_transitions

    return transition_function

# Fonction d'application d'une action à un état
def apply_action(state, action):

    # Créer une copie de l'état actuel
    next_state = state.copy()

    # Appliquer l'action à l'état
    if action == 'Augmenter le prix':
        # Demander à l'utilisateur de saisir l'augmentation du prix
        next_state['prix_entreprise'] += 1

    elif action == 'Diminuer le prix':
        # Demander à l'utilisateur de saisir la diminution du prix
        next_state['prix_entreprise'] -= 1
    else:
        # Maintenir le prix actuel
        pass

    # Vérifier que le prix reste dans une plage raisonnable (positif)
    if next_state['prix_entreprise'] < 0:
        next_state['prix_entreprise'] = 0

    return next_state

# Fonction d'initialisation de la table Q
def initialize_q_table(states, actions):
    q_table = {}

    # Parcourir chaque état possible
    for state in states:
        state_key = tuple(state.items())  # Convertir le dictionnaire d'état en un tuple de paires clé-valeur pour le rendre hashable
        # Créer un dictionnaire pour stocker les valeurs Q pour chaque action
        action_values = {}

        # Parcourir chaque action possible
        for action in actions:
            # Initialiser la valeur Q pour chaque action à 0
            action_values[action] = 0

        # Ajouter les valeurs Q pour chaque action à l'état actuel
        q_table[state_key] = action_values

    return q_table

# Fonction d'apprentissage Q-Learning
def q_learning(states, actions, transition_function, reward_function, q_table, alpha, gamma, episodes, max_steps_per_episode):
    # Parcourir le nombre d'épisodes spécifié
    for episode in range(episodes):
        # Initialiser l'état actuel
        current_state = random.choice(states)

        # Parcourir le nombre maximum de pas par épisode
        for step in range(max_steps_per_episode):
            # Sélectionner l'action en fonction de la table Q et de la politique epsilon-greedy
            action = select_action(q_table, current_state, actions, epsilon)

            # Convertir l'état courant en une clé hashable
            current_state_key = tuple(current_state.items())

            # Vérifier si la clé existe dans le dictionnaire transition_function
            if current_state_key not in transition_function:
                # Gérer le cas où la clé n'est pas trouvée
                print(f"La clé {current_state_key} n'existe pas dans transition_function.")
                break

            # Vérifier si l'action est disponible pour l'état actuel
            if action not in transition_function[current_state_key]:
                # Gérer le cas où l'action n'est pas trouvée pour l'état actuel
                print(f"L'action {action} n'est pas disponible pour l'état {current_state_key}.")
                break

            # Appliquer l'action sélectionnée et obtenir l'état suivant
            next_state = transition_function[current_state_key][action]

            # Calculer la récompense
            reward = reward_function(current_state, action, next_state)

            # Mettre à jour la table Q en utilisant la règle de mise à jour Q-Learning
            update_q_table(q_table, current_state_key, action, next_state, reward, alpha, gamma)

            # Définir l'état actuel comme l'état suivant
            current_state = next_state

# Fonction de sélection de l'action basée sur la table Q et la politique epsilon-greedy
def select_action(q_table, state, actions, epsilon):
    state_key = tuple(state.items())  # Convertir un dictionnaire en tuple pour une clé hashable

    # Vérifier si l'état est présent dans la table Q
    if state_key not in q_table:
        # Initialiser les valeurs Q pour l'état s'il n'est pas présent
        q_table[state_key] = {action: 0 for action in actions}

    # Générer une valeur aléatoire entre 0 et 1
    random_value = random.uniform(0, 1)

    # Si la valeur aléatoire est inférieure à epsilon, sélectionner l'action avec la valeur Q la plus élevée
    if random_value < epsilon:
        best_action = max(q_table[state_key], key=q_table[state_key].get)
    else:
        # Sélectionner une action aléatoire parmi les actions possibles
        best_action = random.choice(actions)

    return best_action

# Fonction de mise à jour de la table Q
def update_q_table(q_table, state, action, next_state, reward, alpha, gamma):

    # Récupérer la valeur Q actuelle pour l'état, l'action et l'état suivant
    current_q_value = q_table[state][action]
    next_state_max_q_value = max(q_table[next_state].values())

    # Appliquer la règle de mise à jour Q-Learning
    new_q_value = current_q_value + alpha * (reward + gamma * next_state_max_q_value - current_q_value)

    # Mettre à jour la valeur Q dans la table Q
    q_table[state][action] = new_q_value

# Définition des paramètres d'apprentissage
alpha = 0.2  # Taux d'apprentissage
gamma = 0.9  # Facteur d'actualisation
epsilon = 0.1  # Paramètre epsilon pour la politique epsilon-greedy
episodes = 1000  # Nombre d'épisodes d'apprentissage
max_steps_per_episode = 100  # Nombre maximum de pas par épisode

# Charger les données Excel
states_data, actions, rewards_data, dates_data = read_excel_data()

# Définir les états possibles
possible_states = define_states(states_data)

# Définir les actions possibles
possible_actions = define_actions()

# Définir la fonction de transition
transition_function = define_transition_function(possible_states, possible_actions)

# Définir la fonction de récompense
reward_function = calculate_reward

# Initialiser la table Q
q_table = initialize_q_table(possible_states, possible_actions)

# Lancer l'apprentissage Q-Learning
q_learning(possible_states, possible_actions, transition_function, reward_function, q_table, alpha, gamma, episodes, max_steps_per_episode)

# Afficher la table Q apprise
print("Table Q apprise:")
for state in q_table:
    print(state, q_table[state])

# Utiliser la table Q apprise pour prendre des décisions
def take_action(current_state, q_table, action_space):
    current_state_key = tuple(current_state.items())

    if current_state_key in q_table:
        best_action = max(q_table[current_state_key], key=q_table[current_state_key].get)
        return best_action
    else:
        random_action = random.choice(action_space)
        return random_action

# Exemple d'utilisation de la table Q apprise

current_state = {'prix_entreprise': 6.0, 'prix_concurrents': 4.9, 'demande_marche': 9585.1, 'stock_marche_moyen': 37929.9, 'cout_entreprise': 0.3, 'stock_entreprise': 2452.4, 'qte_vendue': 613.6}
best_action = take_action(current_state, q_table, possible_actions)

print("État actuel:", current_state)
print("Action optimale:", best_action)

np.save('q_table.npy', q_table)

from google.colab import files
files.download('q_table.npy')