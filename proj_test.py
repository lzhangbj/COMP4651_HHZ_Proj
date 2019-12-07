# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Data Processing
# MAGIC #### 1.1 Data Load

# COMMAND ----------

train_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/train.csv')
building_metadata_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/building_metadata.csv')
weather_train_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/weather_train.csv')

# COMMAND ----------

test_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/test.csv')
building_metadata_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/building_metadata.csv')
weather_test_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/weather_test.csv')

# COMMAND ----------

meta_train_df =  building_metadata_df.join(train_df, (building_metadata_df['building_id'] == train_df['building_id']))
cond = [weather_train_df.site_id == meta_train_df.site_id, weather_train_df.timestamp == meta_train_df.timestamp]
trainDF =  weather_train_df.join(meta_train_df, cond)

# COMMAND ----------

meta_test_df =  building_metadata_df.join(test_df, (building_metadata_df['building_id'] == test_df['building_id']))
cond = [weather_test_df.site_id == meta_test_df.site_id, weather_test_df.timestamp == meta_test_df.timestamp]
testDF =  weather_test_df.join(meta_test_df, cond)

# COMMAND ----------

datasetDF = trainDF.drop("timestamp", "site_id", "building_id")
datasetDF = datasetDF.na.fill(0)

# COMMAND ----------

datasetTestDF = testDF.drop("timestamp", "site_id", "building_id")
datasetTestDF = datasetTestDF.na.fill(0)

# COMMAND ----------

split15DF, split85DF = datasetDF.randomSplit([15., 85.], seed=190)

# Let's cache these datasets for performance
testSetDF = split15DF.cache()
trainingSetDF = split85DF.cache()
submitSetDF = datasetTestDF.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. ML and Evaluation Set Up

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql.functions import col, log, avg
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
from math import sqrt
from statistics import mean
# cross validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# ***** vectorizer MODEL ****
from pyspark.ml.feature import VectorAssembler

vectorizer = VectorAssembler()
vectorizer.setInputCols(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", 
                         "wind_direction", "wind_speed", "square_feet", "year_built", "floor_count", "meter"])
vectorizer.setOutputCol("features")

class RMSLEEvaluator(Evaluator):

    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        """
        Returns a random number. 
        Implement here the true metric
        """
        new_dataset = dataset.withColumn('result_'+self.predictionCol, log(col(self.predictionCol)+1))
        new_dataset = new_dataset.withColumn('result_'+self.labelCol, log(col(self.labelCol)+1))
        new_dataset = new_dataset.withColumn('result', (col('result_'+self.predictionCol) - col('result_'+self.labelCol))**2)
        
        result = new_dataset.agg(avg(col("result")))
        result = result.collect()[0]["avg(result)"]
        return sqrt(result)
      
    def isLargerBetter(self):
        return True

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Linear Regression
# MAGIC ##### 2.1.1 Linear Regression Pipeline

# COMMAND ----------

# ***** LINEAR REGRESSION MODEL ****
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
# Let's initialize our linear regression learner
lr = LinearRegression()

# Now we set the parameters for the method
lr.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setMaxIter(100)\
  .setRegParam(0.15)

# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()

lrPipeline.setStages([vectorizer, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.2 Linear Regression Simple Fitting

# COMMAND ----------

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.3 Linear Regression Simple Evaluation

# COMMAND ----------

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "meter_reading" and col != "primary_use"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)

resultsDF = lrModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Simple Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.1.4 Linear Regression Cross Validation Fitting

# COMMAND ----------

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our regularization parameter from 0.05 to 0.50
regParam = [x / 200.0 for x in range(1, 11)]

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, regParam)
             .build())
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
cvModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.1.5 Linear Regression Cross Validation Evaluation

# COMMAND ----------

# evaluation
resultsDF = lrModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Cross Validated Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Decision Tree
# MAGIC #### 2.2.1 Decision Tree Pipeline

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

# Create a DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setFeaturesCol("features")\
  .setMaxBins(100)

# Create a Pipeline
dtPipeline = Pipeline()

# Set the stages of the Pipeline
dtPipeline.setStages([vectorizer, dt])

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2.2 Decision Tree Simple Fitting

# COMMAND ----------

# Let's first train on the entire dataset to see what we get
dtModel = dtPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2.3 Decision Tree Simple Evaluation

# COMMAND ----------

resultsDF = dtModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Simple Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# dtSubmitted = dtModel.transform(datasetTestDF)
# dtSubmitted = dtSubmitted.select('row_id', 'predicted_meter_reading')
# dtSubmitted = dtSubmitted.withColumnRenamed('predicted_meter_reading', "meter_reading")

# COMMAND ----------

# file='/FileStore/tables/my.csv'
# dtSubmitted = dtSubmitted.repartition(1)
   
# dtSubmitted.write.format("com.databricks.spark.csv")\
#    .option("header", "true")\
#    .save(file)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2.4 Decision Tree Cross Validation Fitting

# COMMAND ----------

# Let's just reuse our CrossValidator with the new dtPipeline, RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(dtPipeline)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2,3,4,5])
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
dtModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2.5 Decision Tree Cross Validation Evaluation

# COMMAND ----------

# evaluation
resultsDF = dtModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Cross Validated Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Random Forest
# MAGIC #### 2.3.1 Random Forest Pipeline

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Create a DecisionTreeRegressor
rf = RandomForestRegressor()

rf.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setFeaturesCol("features")\
  .setMaxBins(100)

# Create a Pipeline
rfPipeline = Pipeline()

# Set the stages of the Pipeline
rfPipeline.setStages([vectorizer, rf])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.2 Random Forest Simple Fitting

# COMMAND ----------

# Let's first train on the entire dataset to see what we get
rfModel = rfPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.3 Random Forest Simple Evaluation

# COMMAND ----------

resultsDF = rfModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Simple Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.4 Random Forest Cross Validation Fitting

# COMMAND ----------

# Let's just reuse our CrossValidator with the new dtPipeline, RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(rfPipeline)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2,3,4,5])
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
rfModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.5 Random Forest Cross Validation Evaluation

# COMMAND ----------

# evaluation
resultsDF = rfModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Cross Validated Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Gradient-Boosted Trees
# MAGIC #### 2.4.1 Gradient-Boosted Trees Pipeline

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

# Create a DecisionTreeRegressor
gbt = GBTRegressor()

gbt.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setFeaturesCol("features")\
  .setMaxBins(100)\
  .setMaxIter(100)

# Create a Pipeline
gbtPipeline = Pipeline()

# Set the stages of the Pipeline
gbtPipeline.setStages([vectorizer, gbt])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.2 Gradient-Boosted Trees Simple Fitting

# COMMAND ----------

# Let's first train on the entire dataset to see what we get
gbtModel = gbtPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.3 Gradient-Boosted Trees Simple Evaluation

# COMMAND ----------

resultsDF = gbtModel.transform(testSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Simple Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.4 Gradient-Boosted Trees Cross Validation Fitting

# COMMAND ----------

# # Let's just reuse our CrossValidator with the new dtPipeline, RegressionEvaluator regEval, and 3 fold cross validation
# crossval.setEstimator(gbtPipeline)

# # Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
# paramGrid = (ParamGridBuilder()
#              .addGrid(gbt.maxDepth, [2,3,4,5])
#              .build())

# # Add the grid to the CrossValidator
# crossval.setEstimatorParamMaps(paramGrid)

# # Now let's find and return the best model
# gbtModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.5 Gradient-Boosted Trees Cross Validation Evaluation

# COMMAND ----------

# # evaluation
# resultsDF = gbtModel.transform(testSetDF)
# # Create an RMSE evaluator using the label and predicted columns
# regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# # Run the evaluator on the DataFrame
# rmsle = regEval.evaluate(resultsDF)

# print("Cross Validated Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Ensemble Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Data Preparation

# COMMAND ----------

# lrModel
# dtModel
# rfModel
# gbtModel
from pyspark.sql.functions import monotonically_increasing_id 

lrTrainingSetDF = lrModel.transform(trainingSetDF).withColumnRenamed('predicted_meter_reading', 'lr_predicted').select('lr_predicted', "meter_reading")\
                          .withColumn('id', monotonically_increasing_id())
dtTrainingSetDF = dtModel.transform(trainingSetDF).withColumnRenamed('predicted_meter_reading', 'dt_predicted').select('dt_predicted')\
                          .withColumn('id', monotonically_increasing_id())
rfTrainingSetDF = rfModel.transform(trainingSetDF).withColumnRenamed('predicted_meter_reading', 'rf_predicted').select('rf_predicted')\
                          .withColumn('id', monotonically_increasing_id())
gbtTrainingSetDF = gbtModel.transform(trainingSetDF).withColumnRenamed('predicted_meter_reading', 'gbt_predicted').select('gbt_predicted')\
                          .withColumn('id', monotonically_increasing_id())

lrTestSetDF = lrModel.transform(testSetDF).withColumnRenamed('predicted_meter_reading', 'lr_predicted').select('lr_predicted')\
                          .withColumn('id', monotonically_increasing_id())
dtTestSetDF = dtModel.transform(testSetDF).withColumnRenamed('predicted_meter_reading', 'dt_predicted').select('dt_predicted')\
                          .withColumn('id', monotonically_increasing_id())
rfTestSetDF = rfModel.transform(testSetDF).withColumnRenamed('predicted_meter_reading', 'rf_predicted').select('rf_predicted')\
                          .withColumn('id', monotonically_increasing_id())
gbtTestSetDF = gbtModel.transform(testSetDF).withColumnRenamed('predicted_meter_reading', 'gbt_predicted').select('gbt_predicted', "meter_reading")\
                          .withColumn('id', monotonically_increasing_id())

# COMMAND ----------

from pyspark.sql.types import StructType

# schema = StructType([])
# leTrainingSetDF = sqlContext.createDataFrame(sc.emptyRDD(), schema)
# leTrainingSetDF = leTrainingSetDF.join(lrTrainingSetDF, how="left_outer")
leTrainingSetDF = lrTrainingSetDF.join(dtTrainingSetDF, "id")
leTrainingSetDF = leTrainingSetDF.join(rfTrainingSetDF, "id")
leTrainingSetDF = leTrainingSetDF.join(gbtTrainingSetDF, "id")
leTrainingSetDF = leTrainingSetDF.drop('id')

# leTestSetDF = sqlContext.createDataFrame(sc.emptyRDD(), schema)
# leTestSetDF = leTestSetDF.join(lrTrainingSetDF, how="left_outer")
leTestSetDF = lrTestSetDF.join(dtTestSetDF, "id")
leTestSetDF = leTestSetDF.join(rfTestSetDF, "id")
leTestSetDF = leTestSetDF.join(gbtTestSetDF, "id")
leTestSetDF = leTestSetDF.drop('id')

leVectorizer = VectorAssembler()
leVectorizer.setInputCols(["lr_predicted", "dt_predicted", "rf_predicted", "gbt_predicted"])
leVectorizer.setOutputCol("features")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Linear Ensemble
# MAGIC ##### 3.2.1 Linear Ensemble PipeLine Set Up

# COMMAND ----------

le = LinearRegression(fitIntercept=False)

# Now we set the parameters for the method
le.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setMaxIter(100)\
  .setRegParam(0.15)

lePipeline = Pipeline()

lePipeline.setStages([leVectorizer, le])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.2.2 Linear Ensemble Simple Fitting

# COMMAND ----------

leModel = lePipeline.fit(leTrainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.2.3 Linear Ensemble Simple Evaluation

# COMMAND ----------

# The intercept is as follows:
# intercept = leModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:

weights = leModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in leTrainingSetDF.columns if col != "meter_reading"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

#equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

print("Linear Regression Equation: " + equation)

resultsDF = leModel.transform(leTestSetDF)
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmsle = regEval.evaluate(resultsDF)

print("Simple Root Mean Squared Logarithmtic Error: %.2f" % rmsle)

