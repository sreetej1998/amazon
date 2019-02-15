
# coding: utf-8

# # KNN TO DETERMINE AMAZON FOOD REVIEWS

# <b>DATA PREPROCESSING</b>

# In[1]:


#loading the dataset
import sqlite3 as sq
import pandas as pd
data=sq.connect(r"C:\Users\sreetej\Downloads\Amazon\database.sqlite");

amazon_reviews=pd.read_sql_query("select * from Reviews where Score!=3",data);


# In[2]:


#renaming the score as a binary format i.e positive or negitive
def rename_score(x):
    if x>3:
        return "positive";
    else:
        return "negitive";
amazon_reviews["Score"]=amazon_reviews["Score"].map(rename_score)


# In[3]:


#eliminating redundant data
data_valid=amazon_reviews.drop_duplicates(subset=["Userid","Summary","Time"])


# In[4]:


#taking samples of the dataset and then storing them in single dataset
data_pos=data_valid[data_valid["Score"]=="positive"].sample(n=5000)
data_neg=data_valid[data_valid["Score"]=="negitive"].sample(n=5000)
dataset=data_pos.append(data_neg)


# In[5]:


#shuffling the data
dataset=dataset.sample(frac=1).reset_index(drop=True)


# In[7]:


#doing a timebased sorting of the dataset for getting the test and train vectors  
dataset=dataset.sort_values("Time")
dataset["Text"]=dataset["Text"].map(clean)


# In[6]:


#clean function and getting a list of sentences for word2vec
import nltk.stem
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stopword=set(stopwords.words("english"))
snow=nltk.stem.SnowballStemmer('english')
list_sent=[]
import re
def clean(x):
    cle=re.compile(r'<.*?>')
    cle1=re.sub(cle,r' ',x)
    cle2 = re.sub(r'[?|!|\'|"|#]',r'',cle1)
    cle3 = re.sub(r'[.|,|)|(|\|/]',r' ',cle2)
    return cle3

    
for i in range(0,10000):
    x=[]
    
    for j in dataset["Text"].iloc[i].split():
        
        if j.isalpha() and len(j)>2:
            if j not in stopword: 
                x.append(snow.stem(j.lower()))
            else:
                continue
        else:
            continue
            
    list_sent.append(x)   



train_string=list_sent[0:7000]
test_string=list_sent[7000:10001]


# In[8]:


#vectorizing the data
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
w2v_model=gensim.models.Word2Vec(train_string,min_count=5,size=50, workers=4)   
w2v_model_test=gensim.models.Word2Vec(test_string,min_count=5,size=50, workers=4)   


# In[9]:


#obtaining w2v test and train data
import numpy as np

def vect_convert(string,model):
    sentencevect=[]
    for k in string:
        count=0
        sent=np.zeros(50);
        for u in k:
            try:
                wordvec=model.wv[u]
                sent+=wordvec
                count+=1
            
            except:
                continue
    
    
            sent/=count
    
        sentencevect.append(sent)
    return sentencevect
    
train_vect=vect_convert(train_string,w2v_model)
test_vect=vect_convert(test_string,w2v_model)
len(test_vect)


# In[10]:


#obtaning input for tfidf and bow vectors
import nltk.stem
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
snow=nltk.stem.SnowballStemmer('english')
stopword=set(stopwords.words("english"))
def sent_to_tfidf(x):
    final=[]
    str1=""
    s=""
    for sent in x:
        lis=[]
        for word in sent.split():
            if word.isalpha() and len(word)>2:
                if word not in stopword:
                
                    s=snow.stem(word.lower()).encode("utf-8")
                    lis.append(s)
                else:
                    continue;
            else:
                continue;
        
        str1=b" ".join(lis);
        final.append(str1)
    return final
final=sent_to_tfidf(dataset["Text"])
final_train=final[0:7000]
final_test=final[7000:10001]


# In[11]:


#obtained tfidf vectors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
tfidf_train = tf_idf_vect.fit_transform(final_train)
tfidf_test=tf_idf_vect.transform(final_test)
test_label=dataset["Score"][7000:10001]
train_label=dataset["Score"][0:7000]


# In[12]:


tfidf_test.shape


# In[13]:


#obtained bog vectors
count_vect = CountVectorizer() 
bog = count_vect.fit_transform(final_train)
bog_test = count_vect.transform(final_test)
bog_train= bog


# In[25]:


# OBTAINED TFIDF-W2V VECTORS

tfidf_feat = tf_idf_vect.get_feature_names() 


def vec(data):
    tfidf_sent_vectors = []; 
    row=0;
    for sent in data: 
        sent_vec = np.zeros(50) 
        weight_sum =0; 
        for word in sent:
            try:
                vec = w2v_model.wv[word]
                tfidf1 = tfidf_train[row, tfidf_feat.index(word)]
                sent_vec += (vec * tfidf1)
                weight_sum += tfidf1
            except:
                pass
        if weight_sum==0:
            tfidf_sent_vectors.append(np.zeros(50))
            row += 1
            continue
        sent_vec /= weight_sum
        tfidf_sent_vectors.append(sent_vec)
        row += 1
    return tfidf_sent_vectors      
    
w2v_tfidf_train=vec(train_string)
w2v_tfidf_test=vec(test_string)
    


# <b>KNN ACCURACY FOR EACH MODEL</b>

# In[30]:


#creating a class for calculating the accuracy of any model 
#each model will act as an object
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
class Knn:
    

    
    def __init__(self,train,test,test_label,train_label):#INITTIALIZES VARIABLES
        self.optimal=0;
        self.train=train
        self.test=test
        self.test_label=test_label
        self.train_label=train_label
        
    def train_knn(self):#calculates optimal k 
        myList = list(range(0,70))
        neighbors = list(filter(lambda x: x % 2 != 0, myList))
        cv_scores = []
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, self.train, self.train_label, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
        error = [1 - x for x in cv_scores]
        self.optimal = neighbors[error.index(min(error))]
        print("training accuracy is")
        print(max(cv_scores)*100)
        plt.plot(neighbors,error)
        plt.title("hyperparameter vs error")
        plt.xlabel("neighbor")
        plt.ylabel("misclassificationn error")
        plt.show()
        print("optimal k is")
        print(self.optimal)
        
    def test_knn(self):#GIVES THE ACCURACY OF MODEL ON THE TESTED DATA
        k=0
        knn=KNeighborsClassifier(n_neighbors=self.optimal)
        knn.fit(self.train, self.train_label)
        x=knn.predict(self.test)
        acc = accuracy_score(test_label, x, normalize=True) * float(100)
        print("test accuracy is:")
        print(acc)
        print("confusion matrix is")
        print(confusion_matrix(self.test_label,x))
        
        
    
        


# <b>CREATING OBJECTS FOR EACH MODEL AND TESTING THEM</b>

# In[31]:


#training  tfidf vectors
word=Knn(tfidf_train,tfidf_test,test_label,train_label)
word.train_knn()
word.test_knn()


#  <h3>Observations</h3><br>
#  1.Traning and test accuracies are almost similar so there is no problem of overfitting or underfitting.<br>
#  2.optimal k is 69 from 70 the curve will rise.<br>
#  3.there are more false negitives than false positives the model is not as good in classifying negitive class.<br>
#  
# 

# 
# 
# 
# 

# In[32]:


#traning bog vectors
bog_obj=Knn(bog_train,bog_test,test_label,train_label)
bog_obj.train_knn()
bog_obj.test_knn()


#  <h3>Observations</h3><br>
#  1.Traning and test accuracies are almost similar so there is no problem of overfitting or underfitting.<br>
#  2.optimal k is 27 from 27 the curve is rising.<br>
#  3.there ae more true negitives than true positives the model is not as good as classifying positive class.<br>
#  
# 

# In[33]:


#traning w2v
w2v_obj=Knn(train_vect,test_vect,test_label,train_label)
w2v_obj.train_knn()
w2v_obj.test_knn()


#  <h3>Observations</h3><br>
#  1.This model overall performance is poor.<br>
#  2.optimal k is 13.there are a lot of hills and valeys in this graph.<br>
#  3.there are almost same numbers in confusion matrix very poor results this may be due to lack of vocabulary.<br>
#  as the dataset is confined due to hardware restrictions.
# 

# In[34]:


#traning tfidf_w2v vectors
w2v_tfidf_obj=Knn(w2v_tfidf_train,w2v_tfidf_test,test_label,train_label)
w2v_tfidf_obj.train_knn()
w2v_tfidf_obj.test_knn()


#  <h3>Observations</h3><br>
#  1.This model overall performance isvery poor.<br>
#  2.optimal k is 69.<br>
#  3.there are more misclassified datapoints especially negitive ones very poor results this also may be due to lack of<br> vocabulary.
#  as the dataset is confined due to hardware restrictions.
# 

# <h1>Table for accuracies under different models</h2>

# In[1]:


from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Model","Hyperparameter","train_accuracy","test_accuracy"]

x.add_row(["tfidf",])
x.add_row()
x.add_row()
x.add_row()

print(x)

