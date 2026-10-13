import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Sentetik Veri Seti Olusturma
x, y = make_classification(
    n_samples=1000, n_features=5, n_informative=4, n_classes=2, random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 2. Ilk Karar Agaci Modeli (max_depth=4)
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(x_train, y_train)

y_tahmin = dt_model.predict(x_test)
basari = accuracy_score(y_test, y_tahmin)
print(f"Model Basari Orani (Accuracy): {basari * 100:.2f}%")

# Ozellik Onemleri (Decision Tree)
importance_degree = dt_model.feature_importances_
for i, degree in enumerate(importance_degree):
  print(f"Ozellik {i} onem skoru: {degree:.4f}")

# Agac Gorsellestirme
plt.figure(figsize=(12, 8))
plot_tree(
    dt_model,
    filled=True,
    feature_names=[f"Ozellik_{i}" for i in range(x.shape[1])],
    class_names=["Sinif_0", "Sinif_1"],
    rounded=True,
)
plt.title("Decision Tree Gorsellestirmesi")
plt.show()

# 3. Budanmis Karar Agaci Modeli
dt_model_pruned = DecisionTreeClassifier(
    max_depth=3, min_samples_split=10, random_state=42
)
dt_model_pruned.fit(x_train, y_train)

y_pred_pruned = dt_model_pruned.predict(x_test)
basari_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Budanmis Modelin Basari Orani: {basari_pruned * 100:.2f}%")

# 4. Random Forest Modeli
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)
rf_basari = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Modelinin Basari Orani: {rf_basari * 100:.2f}%")

# 5. Random Forest Ozellik Onem Duzeyleri
onemler = rf_model.feature_importances_

ozellik_tablosu = pd.DataFrame(
    {"Ozellik": [f"Ozellik_{i}" for i in range(x.shape[1])], "Onem_Skoru": onemler}
)

ozellik_tablosu = ozellik_tablosu.sort_values(by="Onem_Skoru", ascending=False)
print("\nOzellik Onem Tablosu:")
print(ozellik_tablosu)

# 6. Grafik Gorsellestirme
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Onem_Skoru", y="Ozellik", data=ozellik_tablosu, palette="viridis"
)
plt.title("Random Forest - Ozellik Onem Duzeyleri")
plt.xlabel("Onem Skoru")
plt.ylabel("Ozellikler")
plt.show()