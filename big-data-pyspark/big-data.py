import pyspark
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('LoansDataset.csv',header=True,inferSchema=True)

df = df.withColumn("Months since last delinquent",f.col("Months since last delinquent").cast(IntegerType()))

for col in df.columns:
    if df.schema[col].dataType != StringType():
        mean = df.agg(f.avg(f.col(col)).alias('avg')).collect()[0]['avg']
        df = df.fillna(mean)

df = df.withColumn("Loan Status", when(f.col("Loan Status") == "other", "Other").otherwise(f.col("Loan Status")))
df = df.withColumn("Home Ownership", when(f.col("Home Ownership") == "HaveMortgage", "Home Mortgage").otherwise(f.col("Home Ownership")))

df = df.withColumn("Credit Score", when(f.col("Credit Score") > 800, 800).otherwise(f.col("Credit Score")))
df = df.dropDuplicates(["Loan ID"])
df = df.filter(f.col("Current Loan Amount")!= 99999999)

df = df.withColumn("Credit Used", f.col("Current Credit Balance") / f.col("Maximum Open Credit"))
df = df.withColumn("Credit to income %", f.col("Current Credit Balance") / f.col("Anual Income"))