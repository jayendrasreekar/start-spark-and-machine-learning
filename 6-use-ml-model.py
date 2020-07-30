import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages "com.memsql:memsql-spark-connector_2.11:3.0.0-spark-2.4.4" pyspark-shell'

import mlflow.spark

import pyspark
spark = pyspark.sql.SparkSession.builder.master("local[*]").config("spark.jars.packages", "org.mlflow.mlflow-spark").getOrCreate()

spark.conf.set("spark.datasource.memsql.ddlEndpoint", "memsql")
spark.conf.set("spark.datasource.memsql.user", "root")
spark.conf.set("spark.datasource.memsql.password", "")

#this table is rowstore with a primary key (order, linenumber), and nullable prediction column
data = spark.read.format("memsql").load("tpch.lineitem").select("*").filter("prediction is null")

data_features_only = data.drop("prediction")

feature_columns = ['l_partkey','l_suppkey','l_quantity','l_discount','l_tax']
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
data_to_predict = assembler.transform(data_features_only)

from pyspark.ml.regression import LinearRegression
algo = LinearRegression(featuresCol="features", labelCol="prediction")

model = mlflow.spark.load_model("spark-model")

predictions = model.transform(data_to_predict)
results = predictions.drop("features")

results.write.format("memsql").option("overwriteBehavior", "merge").option("loadDataCompression", "LZ4").mode("overwrite").save("tpch.lineitem")
