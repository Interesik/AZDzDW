import numpy as np
from dtaidistance import dtw
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#toy dataset 
X = np.random.random((100,10))
y = np.random.randint(0,2, (100))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#custom metric
def DTW(vector_1,vector_2):
        return dtw.distance(vector_1, vector_2)
#train
parameters = {'n_neighbors':[2, 4, 8]}
clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
clf.fit(X_train, y_train)

def dtw_kernel(series_1, series_2):
    """
    Funkcja jądra oparta na metryce DTW dla SVM.
    Zwraca macierz podobieństw opartą na odległości DTW.
    """
    series_1_size = series_1.shape[0]
    series_2_size = series_2.shape[0]
    kernel_matrix = np.zeros((series_1_size, series_2_size))
    
    for i in range(series_1_size):
        for j in range(series_2_size):
            # Zwracamy ujemną odległość DTW jako miarę podobieństwa
            kernel_matrix[i, j] = np.exp(-dtw.distance(series_1[i], series_2[j]))
    
    return kernel_matrix

#evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
