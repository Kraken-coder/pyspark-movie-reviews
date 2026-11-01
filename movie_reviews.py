import argparse
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
from pyspark.sql.types import StringType, StructField, StructType


def main():
    parser = argparse.ArgumentParser(description="Process IMDb movie reviews.")
    parser.add_argument("input", nargs="?", default="IMDB Dataset.csv")
    parser.add_argument("output", nargs="?", default="cleaned_movie_reviews.csv")
    args = parser.parse_args()

    spark = SparkSession.builder.master("local[*]").appName("Movie Reviews").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        schema = StructType([
            StructField("review", StringType(), True),
            StructField("sentiment", StringType(), True),
        ])
        reviews = spark.read.csv(
            args.input, header=True, schema=schema, multiLine=True,
            quote='"', escape='"', mode="FAILFAST",
        )

        reviews.printSchema()
        reviews.show(5, truncate=100)

        cleaned = reviews.filter(
            col("review").isNotNull() & ~col("review").rlike(r"(?U)^\s*$")
        )

        print("Total reviews:", cleaned.count())
        cleaned.groupBy("sentiment").count().orderBy("sentiment").show()

        transformed = cleaned.withColumn("review_length", length(col("review")))
        transformed.filter(col("review_length") > 500).show(truncate=100)

        with open(args.output, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(transformed.columns)
            writer.writerows(transformed.toLocalIterator())
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
