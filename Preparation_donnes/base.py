import pandas as pd
import matplotlib.pyplot as plt

def analyze_stability_by_month(df, variable, cible, date_var):
    """
    Analyser la stabilité des modalités d'une variable en calculant les effectifs et les taux mensuels.
    Arguments :
        df : DataFrame avec les données.
        variable : nom de la variable à analyser.
        cible : nom de la variable cible.
        date_var : nom de la variable représentant la date.
    """

    # Créer un ordre explicite des mois
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 1. Convertir la colonne 'month' en type Categorical avec un ordre précis
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    # 2. Calcul des effectifs mensuels par modalité
    monthly_volumes = df.groupby([variable, 'month']).size().reset_index(name='monthly_count')

    # 3. Calcul des taux de l'événement (cible = 1 et cible = 0) par modalité et par mois
    monthly_rates = df.groupby([variable, 'month']).agg(
        event_count=(cible, 'sum'),
        total_count=(variable, 'size')
    ).reset_index()

    # 4. Ajouter la colonne 'variable' dans monthly_rates
    monthly_rates[variable] = monthly_rates[variable]

    # 5. Calcul des taux d'événement
    monthly_rates['event_rate'] = monthly_rates['event_count'] / monthly_rates['total_count']
    monthly_rates['non_event_rate'] = 1 - monthly_rates['event_rate']

    # 6. Calcul du pourcentage d'effectifs par modalité pour chaque mois
    total_monthly_count = df.groupby('month').size().reset_index(name='total_count_month')
    monthly_volumes = pd.merge(monthly_volumes, total_monthly_count, on='month', how='left')
    monthly_volumes['percentage'] = monthly_volumes['monthly_count'] / monthly_volumes['total_count_month'] * 100

    # 7. Fusionner les deux DataFrames (effectifs et taux) pour chaque modalité et mois
    merged = pd.merge(monthly_volumes, monthly_rates[['variable', 'month', 'event_rate', 'non_event_rate']], on=[variable, 'month'], how='left')

    # 8. Affichage des informations générales
    print("Effectifs mensuels et taux d'événement par modalité :")
    print(merged)

    # 9. Graphiques pour visualiser la stabilité
    plt.figure(figsize=(12, 8))
    
    # Tracer les pourcentages des effectifs par modalité
    plt.subplot(2, 1, 1)
    for modality in df[variable].unique():
        modality_data = merged[merged[variable] == modality]
        plt.plot(modality_data['month'], modality_data['percentage'], label=f'Modalité {modality}')
    plt.title("Pourcentage des effectifs mensuels par modalité")
    plt.xlabel("Mois")
    plt.ylabel("Pourcentage d'effectifs (%)")
    
    # Forcer l'axe x dans l'ordre des mois
    plt.xticks(ticks=month_order, labels=month_order, rotation=45)  # Spécifie explicitement l'ordre des mois sur l'axe X
    plt.legend()

    # Tracer les taux d'événement (cible = 1) et de non-événement (cible = 0) par modalité
    plt.subplot(2, 1, 2)
    for modality in df[variable].unique():
        modality_data = merged[merged[variable] == modality]
        plt.plot(modality_data['month'], modality_data['event_rate'], label=f'Taux d\'événement (1) - Modalité {modality}')
        plt.plot(modality_data['month'], modality_data['non_event_rate'], label=f'Taux de non-événement (0) - Modalité {modality}', linestyle='--')
    plt.title("Taux d'événements et de non-événements mensuel par modalité")
    plt.xlabel("Mois")
    plt.ylabel("Taux (%)")
    plt.xticks(ticks=month_order, labels=month_order, rotation=45)  # Spécifie explicitement l'ordre des mois sur l'axe X
    plt.legend()

    plt.tight_layout()
    plt.show()

    return merged

# Exemple d'utilisation de la fonction
data = {
    'col': ['A', 'B', 'A', 'B', 'C', 'A', 'C', 'B', 'A', 'C'],
    'cible': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    'month': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar', 'Apr', 'Apr', 'May', 'May']
}

df = pd.DataFrame(data)

# Appliquer la fonction pour analyser la stabilité des modalités
merged_data = analyze_stability_by_month(df, 'col', 'cible', 'month')
