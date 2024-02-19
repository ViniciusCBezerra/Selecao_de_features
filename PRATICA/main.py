
import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix


dados = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/classificacao_multiclasse/main/Dados/dados_estudantes.csv')

encoder = OneHotEncoder(drop='if_binary')
colunas_categoricas = ['Estado civil', 'Migração', 'Sexo', 'Estrangeiro',
       'Necessidades educacionais especiais', 'Devedor',
       'Taxas de matrícula em dia', 'Bolsista',  'Curso', 'Período', 'Qualificação prévia']

df_categoricas = dados[colunas_categoricas]
df_encoded = pd.DataFrame(encoder.fit_transform(df_categoricas).toarray(),columns=encoder.get_feature_names_out(colunas_categoricas))

dados = pd.concat([dados.drop(colunas_categoricas,axis=1),df_encoded],axis=1)

x = dados.drop('Target',axis=1)
y = dados['Target']

x_treino,x_teste,y_treino,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

parametros = {
    'criterion': 'gini', 
    'max_depth': 6, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 100,
    'random_state' : 5
}

modelo = RandomForestClassifier(**parametros)
cv = StratifiedKFold(shuffle=True,random_state=5)

seletor = RFECV(
    estimator=modelo, 
    step=1, 
    min_features_to_select=1, 
    cv=cv, 
    scoring='recall_weighted', 
    n_jobs=-1, 
)
seletor.fit(x_treino,y_treino)
print(seletor.n_features_)

x_treino_selecionado = seletor.transform(x_treino)
x_teste_selecionado = seletor.transform(x_teste)

modelo.fit(x_treino_selecionado,y_treino)
print(modelo.score(x_teste_selecionado,y_teste))

y_pred = seletor.predict(x_teste)
matriz_confusao = confusion_matrix(y_teste,y_pred)
print(matriz_confusao)
