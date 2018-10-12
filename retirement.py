
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('basics').getOrCreate()


# In[2]:


df = spark.read.csv('dataset/best places for retirement in usa.csv', header=True, inferSchema=True)


# In[3]:


df.show()
print("Total data points:", df.count())


# In[4]:


df1 = spark.read.csv('dataset/state abbreviation.csv', header=True, inferSchema=True)


# In[5]:


df1.show()
print("Total data points:", df.count())


# In[6]:


df1.drop(df1.Postal).show()


# In[7]:


df1.drop('Capital City','Postal').show()


# In[8]:


df1.selectExpr("State as STATE","Standard as Abbreviation") .show()


# In[9]:


df2=df1.selectExpr("State as STATE","Standard as Abbreviation")


# In[10]:


df2.show()


# In[11]:


df3=df2.dropna()


# In[12]:


df3.show()


# In[13]:


df.show()


# In[14]:


df3.show()


# In[15]:


df_join = df.join(df3, df.STATE == df3.STATE, "inner")


# In[16]:


df_join.show()


# In[17]:


df_cleaned=df_join.drop('STATE')


# In[18]:


df_cleaned.show()


# In[19]:


from pyspark.ml.feature import VectorAssembler
vector_assembler = VectorAssembler(inputCols = ['COST OF LIVING', 'CRIME', 'CULTURE', 'HEALTH CARE QUALITY', 'TAXES', 'WEATHER', 'WELL-BEING'], outputCol = 'features')
vector_output = vector_assembler.transform(df_cleaned)
vector_output.printSchema()
vector_output.head(1)
label_features = vector_output.select("features", "OVERALL RANK").toDF('features','label')


# In[20]:


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(label_features)
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))
df_cleaned_summary =lrModel.summary
print("RMSE: " + str(df_cleaned_summary.rootMeanSquaredError))
print("R2: " + str(df_cleaned_summary.r2))


# In[21]:


train_data,test_data = vector_output.randomSplit([0.7,0.3])


# In[22]:


df.cache()
df.printSchema()


# In[23]:


df_iteration=df_cleaned.select('OVERALL RANK', 'COST OF LIVING')


# In[24]:


df_iteration.show()


# In[25]:


from pyspark.ml.feature import VectorAssembler
vector_assembler = VectorAssembler(inputCols = ['COST OF LIVING'], outputCol = 'features')
vector_output = vector_assembler.transform(df_iteration)
vector_output.printSchema()
vector_output.head(1)
label_features = vector_output.select("features", "OVERALL RANK").toDF('features','label')


# In[26]:


from pyspark.ml.regression import LinearRegression
lr1 = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel1 = lr.fit(label_features)
print("Coefficients: %s" % str(lrModel1.coefficients))
print("Intercept: %s" % str(lrModel1.intercept))

