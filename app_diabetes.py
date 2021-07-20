from flask import Flask,render_template,request
from flask_restful import Resource, Api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# criação da aplicação Flask
app = Flask(__name__)
api = Api(app)

# Método para prever diabetes do user
def previsao_diabetes(lista_valores_formularios):
    prever=np.array(lista_valores_formularios).reshape(1,8)
    modelo_salvo=joblib.load('melhor_modelo.sav')
    resultado=modelo_salvo.predict(prever)
    return resultado

# metodo para calcular a melhor acurácia
def modelos():
    #
    # Dicionário para gravar as acuracias
    dict_acuracia = {}
    dict_obj = {}
    #
    #
    # Leitura dos dados de entrada
    #
    df = pd.read_csv("C:/Users/pamelo/virtualwork/diabetes/entradas/pima-indians-diabetes.csv",header=None)
    #
    # definindo as entradas e saída do modelo
    #
    # entradas = ['0','1','2','3','4','5','6','7']
    entradas = [0,1,2,3,4,5,6,7]
    # saida = ['8']
    saida = [8]
    x = df[entradas]
    y = df[saida]
    #
    # Treinamento e Teste
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #
    # normalizando
    #
    normaliza = MinMaxScaler()
    x_train = normaliza.fit_transform(x_train)
    x_test = normaliza.fit_transform(x_test)
    #
    #   Arvore de Decisao
    #
    tree = DecisionTreeClassifier(random_state=1)
    tree = tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    #
    #   Acurácia
    #
    acuracia_tree = accuracy_score(y_test, y_pred)
    dict_acuracia.update({'tree': acuracia_tree})
    dict_obj.update({'tree':tree})
    #
    #    KNN
    #
    cls_KNN = KNeighborsClassifier(n_neighbors=5)  # implementação usando 5 vizinhos
    cls_KNN.fit(x_train, y_train.values.ravel())  # aplica a classificação
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None,
                         n_neighbors=5, p=2, weights='uniform')
    y_pred = cls_KNN.predict(x_test)
    #
    # Acurácia
    #
    acuracia_knn = accuracy_score(y_test, y_pred)
    dict_acuracia.update({'cls_KNN': acuracia_knn})
    dict_obj.update({'cls_KNN':cls_KNN})
    #
    #  REDES NEURAIS
    #
    clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1)
    clf.fit(x_train, y_train.values.ravel())
    # realiza previsão
    y_pred = clf.predict(x_test)
    #
    # Acurácia
    #
    acuracia_mlp = accuracy_score(y_test, y_pred)
    dict_acuracia.update({'clf': acuracia_mlp})
    dict_obj.update({'clf':clf})
    #
    # Pega o modelo de maior acuracia
    #
    maior = max(dict_acuracia.values())
    chave = max(dict_acuracia, key=dict_acuracia.get)
    objetomodelo = dict_obj[chave]
    # print("O modelo {}".format(chave), "é o que tem a maior acuracia {}".format(maior))
    # print(objetomodelo)
    #
    # gravando os dados do modelo
    #
    arquivo = 'melhor_modelo.sav'
    joblib.dump(objetomodelo, arquivo)

    return dict_acuracia

class home(Resource):
    def get(self):
        return render_template('index.html')


api.add_resource(home,'/diabetes')

if __name__ == '__main__':
    #
    # instancia o método dos modelos
    #
    dados = modelos()
    # print(dados)
    #
    # Inicializa a aplicação
    #
    app.run(debug=True)