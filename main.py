import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm 

data = pd.read_csv('data/human-evolution_dataset.csv')

# 3.1 Cranial_Capacity and Tecno 

data['Tecno_encoded'] = data['Tecno'].map({'yes': 1, 'no': 0, 'likely': None})
filtered_data = data.dropna(subset=['Cranial_Capacity', 'Tecno_encoded'])
cranial_capacity_yes = filtered_data[filtered_data['Tecno_encoded'] == 1]['Cranial_Capacity']
cranial_capacity_no = filtered_data[filtered_data['Tecno_encoded'] == 0]['Cranial_Capacity']
t_stat, p_value = stats.ttest_ind(cranial_capacity_yes, cranial_capacity_no)

# print("t-statistic:", t_stat)
# print("p-value:", p_value)

##

# 3.2 `biped`, `Arms` and `Foots`
biped_mapping = {
    'low probability': 0.1,
    'high probability': 0.5,
    'yes': 0.7,
    'modern': 1
}
data['biped_encoded'] = data['biped'].map(biped_mapping)

arms_mapping = {
    'climbing': 0.1,
    'manipulate with precision': 1,
    'manipulate': 0.5
}
data['arms_encoded'] = data['Arms'].map(arms_mapping)

foots_mapping = {
    'climbing': 0.1,
    'walk': 1
}
data['foots_encoded'] = data['Foots'].map(foots_mapping)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.hist(data['biped_encoded'].dropna(), bins=4, edgecolor='black', alpha=0.7)
plt.title('Distribuția valorilor pentru biped_encoded')

plt.subplot(3, 1, 2)
plt.hist(data['arms_encoded'].dropna(), bins=3, edgecolor='black', alpha=0.7)
plt.title('Distribuția valorilor pentru arms_encoded')

plt.subplot(3, 1, 3)
plt.hist(data['foots_encoded'].dropna(), bins=2, edgecolor='black', alpha=0.7)
plt.title('Distribuția valorilor pentru foots_encoded')

plt.tight_layout()
# plt.show()

#

# 3.3 biped and Cranial_Capacity

## Pearson corelation between 2 variables
correlation, p_value_corr = stats.pearsonr(data['biped_encoded'], data['Cranial_Capacity'])

print("\nCorelație Pearson între biped_encoded și Cranial_Capacity:")
print("Pearson Correlation:", correlation)
print("P-value:", p_value_corr)

if p_value_corr < 0.05:
    print("Există o corelație semnificativă între biped_encoded și Cranial_Capacity.")
else:
    print("Nu există o corelație semnificativă între biped_encoded și Cranial_Capacity.")

# linear regression
X = data['biped_encoded'] 
y = data['Cranial_Capacity']  
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nRezultatele regresiei liniare între biped_encoded și Cranial_Capacity:")
print(model.summary())

#

# 3.4 biped and arms

contingency_table = pd.crosstab(data['biped_encoded'], data['arms_encoded'])

## chi square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Afișăm rezultatele testului
print("\nTest Chi-pătrat pe tabelul de contingență între biped_encoded și arms_encoded:")
print("Chi2 statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Frecvențele așteptate:\n", expected)

# Interpretarea rezultatelor:
if p < 0.05:
    print("\nExistă o asociere semnificativă între biped_encoded și arms_encoded.")
else:
    print("\nNu există o asociere semnificativă între biped_encoded și arms_encoded.")
