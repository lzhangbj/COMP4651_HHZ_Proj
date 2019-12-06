# Databricks notebook source
train_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/train.csv')
building_metadata_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/building_metadata.csv')
weather_train_df = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/weather_train.csv')

# COMMAND ----------

meta_train_df =  building_metadata_df.join(train_df, (building_metadata_df['building_id'] == train_df['building_id']))
cond = [weather_train_df.site_id == meta_train_df.site_id, weather_train_df.timestamp == meta_train_df.timestamp]
trainDF =  weather_train_df.join(meta_train_df, cond)

# COMMAND ----------

datasetDF = trainDF.drop("timestamp", "site_id", "building_id")
datasetDF = datasetDF.na.fill(0)

# COMMAND ----------

# ***** vectorizer MODEL ****
from pyspark.ml.feature import VectorAssembler

vectorizer = VectorAssembler()
vectorizer.setInputCols(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", 
                         "wind_direction", "wind_speed", "square_feet", "year_built", "floor_count", "meter"])
vectorizer.setOutputCol("features")

# COMMAND ----------

split15DF, split85DF = datasetDF.randomSplit([15., 85.], seed=190)

# Let's cache these datasets for performance
testSetDF = split15DF#.cache()
trainingSetDF = split85DF#.cache()

# COMMAND ----------

# ***** LINEAR REGRESSION MODEL ****

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

# Let's initialize our linear regression learner
lr = LinearRegression()

# COMMAND ----------

# Now we set the parameters for the method
lr.setPredictionCol("predicted_meter_reading")\
  .setLabelCol("meter_reading")\
  .setMaxIter(100)\
  .setRegParam(0.15)


# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()

lrPipeline.setStages([vectorizer, lr])

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)


# COMMAND ----------

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "meter_reading"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

# Now let's sort the coefficients from greatest absolute weight most to the least absolute weight

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)


# COMMAND ----------

resultsDF = lrModel.transform(testSetDF)#.select("AT", "V", "AP", "RH", "PE", "Prediction_PE")

# COMMAND ----------

# t = resultsDF.groupBy().agg({'predicted_meter_reading': "mean"})
from pyspark.sql.functions import *
# predictionCol = "predicted_meter_reading"
# labelCol = "meter_reading"
# dataset = resultsDF
# dataset = dataset.withColumn('result_'+predictionCol, log(col(predictionCol)+1))
# dataset = dataset.withColumn('result_'+labelCol, log(col(labelCol)+1))
# dataset = dataset.withColumn('result', (col('result_'+predictionCol) - col('result_'+labelCol)))
# result = dataset.agg(avg(col("result")))
# result = result.collect()[0]["avg(result)"]
# print(result)

# COMMAND ----------



# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
from math import sqrt
from statistics import mean

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
# Create an RMSE evaluator using the label and predicted columns
regEval = RMSLEEvaluator(predictionCol="predicted_meter_reading", labelCol="meter_reading")

# Run the evaluator on the DataFrame
rmse = regEval.evaluate(resultsDF)

print("Root Mean Squared Error: %.2f" % rmse)
