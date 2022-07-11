# Modelo preditivo de classificaçãp para prever o valor de uma variável binária (true ou false) a partir de dados numéricos

# Import dos módulos
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Dados de treino
n_train = 10
np.random.seed(0)
df_train = pd.DataFrame({"var1":np.random.random(n_train), \
                         "var2": np.random.random(n_train), \
                         "var3": np.random.random(n_train), \
                         "var4":np.random.randint(0,2,n_train).astype(bool),\
                         "target":np.random.randint(0,2,n_train).astype(bool)})

# Dados de treino
n_valid = 3
np.random.seed(1)
df_valid = pd.DataFrame({"var1":np.random.random(n_valid), \
                         "var2": np.random.random(n_valid), \
                         "var3": np.random.random(n_valid), \
                         "var4":np.random.randint(0,2,n_valid).astype(bool),\
                         "target":np.random.randint(0,2,n_valid).astype(bool)})
# Reduzindo a dimensionalidade para 3 componentes
pca = PCA(n_components = 3) 

# Aplicando o PCA aos datasets
newdf_train = pca.fit_transform(df_train.drop("target", axis = 1))
newdf_valid = pca.transform(df_valid.drop("target", axis = 1)) 

# Gerando novos datasets
features_train = pd.DataFrame(newdf_train)
features_valid = pd.DataFrame(newdf_valid)  

# Cria o modelo de regressão logística
regr = LogisticRegression() 

# Usando o recurso de pipeline do scikit-learn para encadear 2 algoritmos em um mesmo modelo, no caso PCA e Regressão Logística
pipe = Pipeline([('pca', pca), ('logistic', regr)])
pipe.fit(features_train, df_train["target"])
predictions = pipe.predict(features_valid)

# Imprimindo as previsões
print(predictions)