package se.kth.spark.lab1.task4

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel

import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator

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
   
    //set the required paramaters
    val learningAlg = new LinearRegression().setPredictionCol("prediction")
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setRegParam(0.5)
    .setElasticNetParam(0.1)
    
    val arrayStages = Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,learningAlg)
    val pipeline = new Pipeline().setStages(arrayStages)    
    
    //build the parameter grid by setting the values for maxIter and regParam
    val paramGrid = new ParamGridBuilder()
    .addGrid(learningAlg.maxIter, Array(4,6,8,12,14,16))
    .addGrid(learningAlg.regParam, Array(0.15,0.3,0.45,0.6,0.75,0.9))
    .build()
    val evaluator = new RegressionEvaluator()
    //create the cross validator and set estimator, evaluator, paramGrid
    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid).setNumFolds(3)
    val cvModel = cv.fit(obsDF)
    
    def getModelSummary(m : LinearRegressionModel) = {
      val modelSummary = m.summary
      println(s"numIterations: ${modelSummary.totalIterations}")
      println(s"RMSE: ${modelSummary.rootMeanSquaredError}")
      //coefficient of determination: indicates  the proportion of the variance in the dependent variable that is predictable from the independent variable
      println(s"r2: ${modelSummary.r2}")
    }
    
    val bestModelSummary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel]
    //print best model RMSE to compare to previous
    println(s"Best model maxIter: ${bestModelSummary.getMaxIter}")
    println(s"Best model regParam: ${bestModelSummary.getRegParam}")
    println(s"Best model elasticNetParam: ${bestModelSummary.getElasticNetParam}")
    //maxIter: 8  regParam:0.15  ElasticNetParam:0.1
    getModelSummary(bestModelSummary)
    //VALUES OBTAINED: numIterations: 9 RMSE: 17.349420399063767 r2: 0.34385282938404127
    //print rmse of our model
    val model = pipeline.fit(obsDF).stages(6).asInstanceOf[LinearRegressionModel]
    getModelSummary(model)
    //VALUES OBTAINED: numIterations: 8 RMSE: 17.351541791002635 r2: 0.34369235942179976

    //do prediction - print first k
    val testBestModel = cvModel.bestModel.transform(obsDF)
    println("Predicted values of the first 5 elements with best model:")
    testBestModel.drop("rawSongs","arraySongs","vectorSongs","years","label_double").take(5).foreach(println)
    val testModel = pipeline.fit(obsDF).transform(obsDF)
    println("Predicted values of the first 5 elements with first model:")
    testModel.drop("rawSongs","arraySongs","vectorSongs","years","label_double").take(5).foreach(println)
  }
}