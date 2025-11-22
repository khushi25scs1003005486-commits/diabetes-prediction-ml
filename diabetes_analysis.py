import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (update the path if needed)
df = pd.read_csv(r"C:\Users\khush\OneDrive\Documents\diabetes.csv")


# Preview first few rows
print(df.head())

# Check basic info and missing values
print(df.info())
print(df.describe())

# Check how many people have diabetes vs not
print(df['Outcome'].value_counts())

# Pairplot
sns.pairplot(df, hue='Outcome')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Features")
plt.show()
