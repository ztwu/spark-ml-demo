package com.iflytek.scala.tjml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object Kmeans {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("als")
//      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._
    //todo 使用als模型获取基于评分的 用户\物品 特征向量
    //获取数据
    val rating = spark.table("dw.dw_user_rating")
      .select($"gid", $"game_id", $"rating")
    //利用StringIndexer获取映射模型
    val index_1 = new StringIndexer()
      .setInputCol("gid")
      .setOutputCol("gid_indexed")
      .setHandleInvalid("keep")
    val index_2 = new StringIndexer()
      .setInputCol("game_id")
      .setOutputCol("game_id_indexed")
      .setHandleInvalid("keep")
    // 获取管道模型
    val stringIndexers = new Pipeline()
      .setStages(Array(index_1, index_2))
      .fit(rating)
    val indexDF = stringIndexers.transform(rating)

    //数据处理
    val transData = indexDF.select($"gid_indexed".cast("int"),
      $"game_id_indexed".cast("int"),
      $"rating".cast("float"))
      .toDF("gid_indexed", "game_id_indexed", "rating")

    //ALS模型训练
    val alsObj = new ALS()
      .setRank(10)
      .setMaxIter(10)
      .setRegParam(0.01)
      .setUserCol("gid_indexed")
      .setItemCol("game_id_indexed")
      .setRatingCol("rating")
    val alsModel = alsObj.fit(transData)

    //todo
    //用户特征向量数据处理
    val userFactors = alsModel.userFactors
    //获取到的用户特征向量为array[Float]类型，需要转换成DenseVector[Double]类
    val inputDF = userFactors.rdd.map(row => {
        (row.getAs[Int]("id"),
        Vectors.dense(row.getAs[Seq[Float]]("features").toArray.map(_.toDouble)))
    }).toDF("id", "features")

    //Kmeans模型训练
    val kmsObj = new KMeans()
      .setFeaturesCol("features")
      .setK(1000)
      .setMaxIter(10)
    val kmsModel = kmsObj.fit(inputDF)
    //模型中心点集合
    val clusterCenters = kmsModel.clusterCenters
    //计算每个用户特征向量到中心点的欧式距离,以离中心点最近的数据作为推荐模板
    val df = kmsModel.transform(inputDF).rdd.map(row => {
      val feature = row.getAs[DenseVector]("features").toArray
      var minDis: Double = Double.MaxValue
      var center = Vectors.zeros(0)
      clusterCenters.foreach(v => {
        val dis = v.toArray.zip(feature)
          .map(t => (t._1 - t._2) * (t._1 - t._2))
//          .map(t => math.pow((t._1 - t._2),2))
          .reduce(_ + _)
        if (minDis >= dis) {
          center = v
          minDis = dis
        }
      })
      (
        row.getAs[Int]("id"),
        row.getAs[DenseVector]("features").toArray.map(_.toFloat),
        row.getAs[Int]("prediction"), minDis, center
      )
    }).toDF("id", "features", "prediction", "min_dis", "center")

    //数据处理
    val refDF = df.select($"id", $"features", $"min_dis",
      $"prediction", $"center",
      row_number() over Window.partitionBy($"prediction").orderBy(asc("min_dis")) as "row_number")
      .where($"row_number" === 1)
    val test = refDF.select($"id", $"features")
    //ALSModel中的userFactor变量为私有变量，因此需要利用反射机制修改
    val field = classOf[ALSModel].getDeclaredField("userFactors")
    println(field.getName)
    field.setAccessible(true)
    field.set(alsModel, test)

    //result test
    val test_1 = alsModel.recommendForAllUsers(10)
      .join(refDF.select($"id" as "gid", $"prediction"), Seq("gid"), "left")
      .select($"prediction", $"recommendations")
      .join(df.select($"id", $"prediction"), Seq("prediction"), "right")
      .select($"id" as "gid", $"recommendations" as "test_rec", $"prediction")
  }

}
