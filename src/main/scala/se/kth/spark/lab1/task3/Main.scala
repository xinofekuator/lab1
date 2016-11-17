package se.kth.spark.lab1.task3

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
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
    
    val arrayStages = Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer)
    val pipeline = new Pipeline().setStages(arrayStages)    
    
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val data = pipelineModel.transform(obsDF).drop("rawSongs","arraySongs","vectorSongs","years","label_double")
    
    //set the required paramaters
    val learningAlg1 = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.1)
    .setElasticNetParam(0.1)
    
    val learningAlg2 = new LinearRegression()
    .setMaxIter(50)
    .setRegParam(0.1)
    .setElasticNetParam(0.1)
    
    val learningAlg3 = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.9)
    .setElasticNetParam(0.1)
    
    val learningAlg4 = new LinearRegression()
    .setMaxIter(50)
    .setRegParam(0.9)
    .setElasticNetParam(0.1)
    
    val model1 = learningAlg1.fit(data)
    val model2 = learningAlg2.fit(data)
    val model3 = learningAlg3.fit(data)
    val model4 = learningAlg4.fit(data)
    
    def getModelSummary(m : LinearRegressionModel) = {
      val modelSummary = m.summary
      println(s"numIterations: ${modelSummary.totalIterations}")
      println(s"RMSE: ${modelSummary.rootMeanSquaredError}")
      //coefficient of determination: indicates  the proportion of the variance in the dependent variable that is predictable from the independent variable
      println(s"r2: ${modelSummary.r2}")
    }
    
    getModelSummary(model1)
    getModelSummary(model2)
    getModelSummary(model3)
    getModelSummary(model4)
    
    //make predictions on testing data
    val test1 = model1.transform(data)
    println("Predicted values of the first 5 elements with model1:")
    test1.take(5).foreach(println)
    val test2 = model2.transform(data)
    println("Predicted values of the first 5 elements with model2:")
    test2.take(5).foreach(println)
    val test3 = model3.transform(data)
    println("Predicted values of the first 5 elements with model3:")
    test3.take(5).foreach(println)
    val test4 = model4.transform(data)
    println("Predicted values of the first 5 elements with model4:")
    test4.take(5).foreach(println)
  }
}