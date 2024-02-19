import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


dados = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/selecao-de-features/main/dados/hotel.csv')

x = dados.drop('booking_status',axis=1)
y = dados['booking_status']

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)
parametros = {
    'max_depth': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 300,
    'random_state': 5
}
modelo = RandomForestClassifier(**parametros)

cv = StratifiedKFold(shuffle=True,random_state=5)

rfecv = RFECV(
    estimator=modelo,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)
rfecv.fit(x_treino,y_treino)

print(rfecv.n_features_)

x_treino_selecionado = rfecv.transform(x_treino)
x_teste_selecionado = rfecv.transform(x_teste)
modelo.fit(x_treino_selecionado,y_treino)

resultado = rfecv.cv_results_

results_df = pd.DataFrame({'Valores':resultado['mean_test_score']})
print(results_df)

plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index+1,y='Valores',data=results_df)