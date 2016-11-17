package se.kth.spark.lab1.task6

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SQLContext, DataFrame }
import org.apache.spark.ml.PipelineModel

import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.PolynomialExpansion

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    
//    //Tests made for the Helpers
//    val double = 2.0
//    val v1 = Vectors.dense(Array(2.0,4.0,3.0)) 
//    val v2 = Vectors.dense(Array(3.0,2.0,5.0))
//    println(VectorHelper.dot(v1, v2))
//    println(VectorHelper.dot(v1, double))
//    println(VectorHelper.sum(v1, v2))
//    println(VectorHelper.fill(5, 1.5))
    
    
val obsDF: DataFrame = sqlContext.read.text(filePath).toDF("rawSongs")

    val regexTokenizer = new RegexTokenizer()
    .setInputCol("rawSongs")
    .setOutputCol("arraySongs")
    .setPattern(",")
    
    val arr2Vect = new Array2Vector()
    .setInputCol("arraySongs").setOutputCol("vectorSongs")
    
    val lSlicer = new VectorSlicer()
    .setInputCol("vectorSongs")
    .setOutputCol("years")
    .setIndices(Array(0))
    
    val v2d = new Vector2DoubleUDF((x: org.apache.spark.ml.linalg.Vector) => x(0).toDouble)
    .setInputCol("vectorSongs").setOutputCol("label_double")
    
    val smallest_year = 1922.0
    val lShifter = new DoubleUDF((a:Double) => a - smallest_year)
    .setInputCol("label_double").setOutputCol("label")
    
    val fSlicer = new VectorSlicer()
    .setInputCol("vectorSongs")
    .setOutputCol("features")
    .setIndices(Array(1,2,3))
    
    val myLearningAlg = new MyLinearRegressionImpl().setPredictionCol("prediction")
    .setLabelCol("label")
    .setFeaturesCol("features")
    
    val arrayStages = Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLearningAlg)
    val pipeline = new Pipeline().setStages(arrayStages)    
    val model = pipeline.fit(obsDF).stages(6).asInstanceOf[MyLinearModelImpl]
    //RMSE error : 
    println("Training error (RMSE) in the different iterations")
    model.trainingError.foreach(println)
    // Last iteration RMSE: 19.076770311012933 
  }
}