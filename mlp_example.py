from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X,y = iris.data, iris.target

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3,random_state=1)

mlp=MLPClassifier(solver='adam',hidden_layer_sizes=(5,),random_state=1,learning_rate='constant',learning_rate_init=0.01,max_iter=400,activation='logistic',momentum=0.1,verbose=True,early_stopping=True,validation_fraction=0.3,tol=0.0001)

mlp.fit(X_treino, y_treino)
saidas=mlp.predict(X_teste)

print('----------')
print('saída rede:\t',saidas)
print('saída desejada:\t',y_teste)
print('----------')
print('Score: ',(saidas == y_teste).sum()/len(X_teste))
print('Score: ', mlp.score(X_teste, y_teste))
