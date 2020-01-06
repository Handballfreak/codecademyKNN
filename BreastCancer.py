import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#saving the Data from sklearn.datasets 
breast_cancer_data=load_breast_cancer()

#getting overview of the data
#print(breast_cancer_data.data[0],breast_cancer_data.feature_names)
#print(breast_cancer_data.target,breast_cancer_data.target_names)

#Splitting Dataset
x_train, x_test, y_train, y_test=train_test_split(breast_cancer_data.data,breast_cancer_data.target, test_size=0.2, random_state=100)

#Test succesfully split
#print(len(x_train),len(x_test))

#adding accuracy for k=1-100
accuracy_list=[]
for k in range(1,101):
  classifier=KNeighborsClassifier(n_neighbors=k)
  classifier.fit(x_train,y_train)
  accuracy_list.append(classifier.score(x_test,y_test))

#Visualization of the different Accuracy depends on k
plt.plot(range(1,101),accuracy_list)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifeir Accuracy")
plt.show()
