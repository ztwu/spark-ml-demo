package com.iflytek.scala.tjml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

object ALS {
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("als")
      .getOrCreate()
    import spark.implicits._

    val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // # 冷启动策略使用"drop"，不对NaN进行评估
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // # 对每个用户推荐top 10的movie
    val userRecs = model.recommendForAllUsers(10)
    // # 对每部电影推荐top 10的user
    val movieRecs = model.recommendForAllItems(10)

//    // # 为指定的用户组推荐top 10的电影
//    val users = ratings.select(als.getUserCol).distinct().limit(3)
//    val userSubsetRecs = model.recommendForUserSubset(users, 10)
//    // # 为指定的电影组推荐top 10的用户
//    val movies = ratings.select(als.getItemCol).distinct().limit(3)
//    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)

    userRecs.show()
    movieRecs.show()

  }

}
