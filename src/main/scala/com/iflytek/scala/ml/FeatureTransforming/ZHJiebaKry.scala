package com.iflytek.scala.ml.FeatureTransforming

import com.huaban.analysis.jieba.{JiebaSegmenter, SegToken}
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object JiebaKry {
  def main(args: Array[String]): Unit = {
    //    定义结巴分词类的序列化
    val conf = new SparkConf()
      .registerKryoClasses(Array(classOf[JiebaSegmenter]))
      .set("spark.rpc.message.maxSize","800")
//    KryoSerialization速度快，可以配置为任何org.apache.spark.serializer的子类。
//    但Kryo也不支持所有实现了 java.io.Serializable 接口的类型，
//    它需要你在程序中 register 需要序列化的类型，以得到最佳性能
//      .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
//      .set("spark.serializer","org.apache.spark.serializer.JavaSerialization") 默认
    //    建立sparkSession,并传入定义好的Conf
    val spark = SparkSession
      .builder()
      .appName("Jieba UDF")
      .master("local")
      .config(conf)
      .getOrCreate()

    // 定义结巴分词的方法，传入的是DataFrame，输出也是DataFrame多一列seg（分好词的一列）
    def jieba_seg(df:DataFrame,colname:String): DataFrame ={
      val segmenter = new JiebaSegmenter()
      val seg = spark.sparkContext.broadcast(segmenter)
      val jieba_udf = udf{(sentence:String)=>
        val segV = seg.value
        segV.process(sentence.toString,SegMode.INDEX)
          .toArray().map(_.asInstanceOf[SegToken].word)
          .filter(_.length>1).mkString("/")
      }
      df.withColumn("seg",jieba_udf(col(colname)))
    }

    val df = spark.read
      .format("text")
      .load("data/mydata/jiebaword.txt")
    val df_seg = jieba_seg(df,"value")
    df_seg.show()
  }
}
