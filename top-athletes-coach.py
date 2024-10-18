from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, when, sum as _sum, collect_list, row_number
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("OlympicsAnalysis").getOrCreate()

# Load datasets
athletes_2012 = spark.read.csv("athletes_2012.csv", header=True, inferSchema=True)
athletes_2016 = spark.read.csv("athletes_2016.csv", header=True, inferSchema=True)
athletes_2020 = spark.read.csv("athletes_2020.csv", header=True, inferSchema=True)
coaches = spark.read.csv("coaches.csv", header=True, inferSchema=True)
medals = spark.read.csv("medals.csv", header=True, inferSchema=True)

# Standardize the casing
athletes_2012 = athletes_2012.select([upper(col(column)).alias(column.upper()) for column in athletes_2012.columns])
athletes_2016 = athletes_2016.select([upper(col(column)).alias(column.upper()) for column in athletes_2016.columns])
athletes_2020 = athletes_2020.select([upper(col(column)).alias(column.upper()) for column in athletes_2020.columns])
coaches = coaches.select([upper(col(column)).alias(column.upper()) for column in coaches.columns])
medals = medals.select([upper(col(column)).alias(column.upper()) for column in medals.columns])

# Combine athletes data
athletes = athletes_2012.union(athletes_2016).union(athletes_2020)

# Assign points based on medals
points_2012 = medals.filter(col("YEAR") == 2012).withColumn("POINTS",
    when(col("MEDAL") == "GOLD", 20).when(col("MEDAL") == "SILVER", 15).when(col("MEDAL") == "BRONZE", 10))
points_2016 = medals.filter(col("YEAR") == 2016).withColumn("POINTS",
    when(col("MEDAL") == "GOLD", 12).when(col("MEDAL") == "SILVER", 8).when(col("MEDAL") == "BRONZE", 6))
points_2020 = medals.filter(col("YEAR") == 2020).withColumn("POINTS",
    when(col("MEDAL") == "GOLD", 15).when(col("MEDAL") == "SILVER", 12).when(col("MEDAL") == "BRONZE", 7))

# Combine points data
combined_points = points_2012.union(points_2016).union(points_2020)

# Calculate total points per athlete
athlete_points = combined_points.groupBy("ID", "SPORT").agg(
    _sum("POINTS").alias("TOTAL_POINTS"),
    _sum(when(col("MEDAL") == "GOLD", 1).otherwise(0)).alias("GOLD_COUNT"),
    _sum(when(col("MEDAL") == "SILVER", 1).otherwise(0)).alias("SILVER_COUNT"),
    _sum(when(col("MEDAL") == "BRONZE", 1).otherwise(0)).alias("BRONZE_COUNT")
)

# Join with athletes to get the NAME column
athlete_points = athlete_points.join(athletes.select("ID", "NAME"), on="ID", how="left")

# Rank athletes by points, medals, and name
windowSpec = Window.partitionBy("SPORT").orderBy(
    col("TOTAL_POINTS").desc(),
    col("GOLD_COUNT").desc(),
    col("SILVER_COUNT").desc(),
    col("BRONZE_COUNT").desc(),
    col("NAME")
)

top_athletes = athlete_points.withColumn("RANK", row_number().over(windowSpec)).filter(col("RANK") == 1).drop("RANK")

# Collect top athletes' names
top_athletes_names = top_athletes.orderBy(col("NAME")).select(collect_list("NAME")).collect()[0][0]

# Filter coaches from China, India, USA and calculate total points
eligible_coaches = coaches.filter(col("COUNTRY").isin(["CHINA", "INDIA", "USA"]))

# Ensure sports match and join dataframes
coach_athlete_points = eligible_coaches.join(
    athletes.withColumnRenamed("NAME", "ATHLETE_NAME"), 
    (eligible_coaches.ID == athletes.COACH_ID) & (eligible_coaches.SPORT == athletes.SPORT), 
    "inner"
).join(
    combined_points, 
    athletes.ID == combined_points.ID, 
    "inner"
)

# Aggregate points per coach
coach_points = coach_athlete_points.groupBy("COACH_ID", eligible_coaches["NAME"], eligible_coaches["COUNTRY"]).agg(
    _sum("POINTS").alias("TOTAL_POINTS"),
    _sum(when(col("MEDAL") == "GOLD", 1).otherwise(0)).alias("GOLD_COUNT"),
    _sum(when(col("MEDAL") == "SILVER", 1).otherwise(0)).alias("SILVER_COUNT"),
    _sum(when(col("MEDAL") == "BRONZE", 1).otherwise(0)).alias("BRONZE_COUNT")
)

# Rank coaches and handle tie cases
windowSpec = Window.partitionBy("COUNTRY").orderBy(
    col("TOTAL_POINTS").desc(),
    col("GOLD_COUNT").desc(),
    col("SILVER_COUNT").desc(),
    col("BRONZE_COUNT").desc(),
    col("NAME")
)

top_coaches = coach_points.withColumn("RANK", row_number().over(windowSpec)).filter(col("RANK") <= 5).drop("RANK")

# Collect top coaches' names for each country
top_coaches_china = top_coaches.filter(col("COUNTRY") == "CHINA").orderBy(col("NAME")).select(collect_list("NAME")).collect()[0][0]
top_coaches_india = top_coaches.filter(col("COUNTRY") == "INDIA").orderBy(col("NAME")).select(collect_list("NAME")).collect()[0][0]
top_coaches_usa = top_coaches.filter(col("COUNTRY") == "USA").orderBy(col("NAME")).select(collect_list("NAME")).collect()[0][0]

# Combine top coaches' names in the correct order
top_coaches_names = top_coaches_china + top_coaches_india + top_coaches_usa

# Print results in the required format
result = (top_athletes_names, top_coaches_names)
print(result)
