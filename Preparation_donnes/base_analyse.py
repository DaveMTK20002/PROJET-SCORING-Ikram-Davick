import pandas as pd
import numpy as np

# Fonction pour calculer le WOE pour chaque bin, maintenant avec des quantiles
def calculate_woe_quantiles(df, col, cible, num_bins=4):
    # Discrétisation en quantiles (chaque bin contiendra environ le même nombre de données)
    bins = pd.qcut(df[col], q=num_bins)  # Utilisation de qcut pour quantiles
    df['bin'] = bins

    # Calcul des événements et non-événements par bin
    bin_stats = df.groupby('bin').agg(
        event_count=(cible, 'sum'),
        non_event_count=(cible, lambda x: (x == 0).sum())
    )

    # Calcul du WOE pour chaque bin
    total_events = df[cible].sum()
    total_non_events = (df[cible] == 0).sum()

    bin_stats['event_ratio'] = bin_stats['event_count'] / total_events
    bin_stats['non_event_ratio'] = bin_stats['non_event_count'] / total_non_events
    bin_stats['WOE'] = np.log(bin_stats['event_ratio'] / bin_stats['non_event_ratio'])

    return bin_stats


# Fonction d'optimisation greedy pour maximiser le WOE
def greedy_woe_discretization(df, col, cible, epsilon=0.01, max_iter=100):
    # Calcul initial du WOE pour les bins
    bin_stats = calculate_woe_quantiles(df, col, cible)
    previous_woe = bin_stats['WOE'].copy()
    iteration = 0
    
    while iteration < max_iter:
        iteration += 1

        # Trouver les bins adjacents qui peuvent être fusionnés
        woe_diff = bin_stats['WOE'].diff().abs()  # Différence absolue des WOE entre les bins adjacents

        # Si la différence de WOE entre deux bins adjacents est plus grande que epsilon, on fusionne
        if woe_diff.max() > epsilon:
            # Identifier les bins voisins à fusionner
            max_diff_idx = woe_diff.idxmax()  # Indice de la plus grande différence

            # Fusionner ces bins
            lower_bin = bin_stats.index.get_loc(max_diff_idx)
            upper_bin = lower_bin + 1
            
            if upper_bin < len(bin_stats):
                # Fusionner les deux bins en recalculant la nouvelle borne
                new_bin_name = f"({bin_stats.index[lower_bin]} - {bin_stats.index[upper_bin]})"
                merged_bin = pd.concat([bin_stats.iloc[lower_bin], bin_stats.iloc[upper_bin]])

                # Réactualiser les statistiques WOE pour la nouvelle configuration
                bin_stats = bin_stats.drop(bin_stats.index[upper_bin])  # Supprimer l'ancien bin
                bin_stats.loc[new_bin_name] = merged_bin  # Ajouter le bin fusionné

        # Calculer les WOE après fusion
        bin_stats = calculate_woe_quantiles(df, col, cible)

        # Si la variation des WOE est inférieure à epsilon, arrêter
        if (bin_stats['WOE'] - previous_woe).abs().max() < epsilon:
            break

        previous_woe = bin_stats['WOE'].copy()

    return df, bin_stats

# Exemple d'utilisation
# Créer un DataFrame exemple
data = {
    'col': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110,130,150],  # Variable continue
    'cible': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0,1,1,1]  # Variable binaire (événement : 1, non-événement : 0)
}

df = pd.DataFrame(data)

# Appliquer la méthode greedy pour maximiser le WOE
df_adjusted, bin_stats_adjusted = greedy_woe_discretization(df, 'col', 'cible', epsilon=0.01)

print("DF avec ajustement progressif des bins et WOE:")
print(df_adjusted)
print("\nStatistiques des bins après ajustement:")
print(bin_stats_adjusted)
