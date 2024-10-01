from sklearn.metrics import confusion_matrix
import numpy as np
#혼동행렬 이해를 위한 기초
# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'
y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])

conf_mat=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred,
                          labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")

# 이제 저 모델을 2개로 만들어보는 실습

y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred1 = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])
y_pred2 = np.array(['C', 'A', 'A', 'A', 'C', 'C', 'C'])
conf_mat1=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred1,
                          labels=["A", "C"])

conf_mat1

conf_mat2=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred2,
                          labels=["A", "C"])

conf_mat2
from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat1,
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")


p=ConfusionMatrixDisplay(confusion_matrix=conf_mat2,
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")