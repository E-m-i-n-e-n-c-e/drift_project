"""Optional feature-window layer: Kafka `sensors` -> Spark Structured Streaming 10 s/5 s windows
(mean/std per sensor) -> console (or Kafka topic `features`).
Run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 streaming/spark_features.py"""
from pyspark.sql import SparkSession, functions as F, types as T

spark = SparkSession.builder.appName("drift-features").getOrCreate()
schema = T.StructType([T.StructField("t", T.LongType()), T.StructField("ts", T.DoubleType()),
                       T.StructField("features", T.MapType(T.StringType(), T.DoubleType())),
                       T.StructField("label", T.IntegerType())])
raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092")
       .option("subscribe", "sensors").load())
df = (raw.select(F.from_json(F.col("value").cast("string"), schema).alias("r")).select("r.*")
      .withColumn("event_time", F.col("ts").cast("timestamp")))
agg = df.withWatermark("event_time", "10 seconds").groupBy(F.window("event_time", "10 seconds", "5 seconds")).agg(
    F.count("*").alias("n"), F.avg(F.col("features")["0"]).alias("mean_f0"), F.stddev(F.col("features")["0"]).alias("std_f0"),
    F.avg(F.col("label")).alias("attack_rate"))
agg.writeStream.outputMode("update").format("console").option("truncate", False).start().awaitTermination()
