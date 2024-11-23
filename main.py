import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data/human-evolution_dataset.csv')


print("Initial Dataset Info:")
print(data.info())  # Overview of dataset structure

print("\nNumber of duplicate rows:", data.duplicated().sum()) 

print("\nUnique Genus & Species:")
print(data["Genus_&_Specie"].unique()) 

print("\nMissing Values per Column:")
print(data.isnull().sum())  

# Fill missing values with placeholders or mean/median for numerical columns
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        data[column].fillna(data[column].median(), inplace=True) 
    elif data[column].dtype == 'object':
        data[column].fillna("Unknown", inplace=True) 

print("\nAfter Handling Missing Values:")
print(data.isnull().sum())  # Check no missing values remain

data.drop_duplicates(inplace=True)
print("\nNumber of rows after removing duplicates:", data.shape[0])

numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
print("\nBasic Statistics for Numerical Columns:")
print(data[numerical_cols].describe())

# Exploring relationships
if 'Age' in data.columns and 'Height' in data.columns:  # scatter plot analysis
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\nVisualizing Relationship Between Age and Height:")
    sns.scatterplot(x=data['Age'], y=data['Height'], hue=data['Genus_&_Specie'])
    plt.title("Age vs. Height by Genus & Species")
    plt.xlabel("Age (in years)")
    plt.ylabel("Height (in cm)")
    plt.legend(loc='upper left')
    plt.show()

if 'Region' in data.columns and 'Age' in data.columns:
    region_stats = data.groupby('Region')['Age'].mean().sort_values(ascending=False)
    print("\nAverage Age by Region:")
    print(region_stats)

# Save cleaned data
output_file = 'data/human-evolution_dataset_cleaned.csv'
data.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")
