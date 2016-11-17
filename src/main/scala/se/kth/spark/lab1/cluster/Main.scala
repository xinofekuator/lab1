package se.kth.spark.lab1.cluster

import se.kth.spark.lab1._
import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer

//This is the Main class specified in the build.sbt assembly
object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val conf = new SparkConf().setAppName("lab1")
    
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    //val filePath = "src/main/resources/millionsongClusterFormat.txt"
    val filePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/all.txt"
    val obsDF: DataFrame = sqlContext.read.text(filePath).toDF("rawSongs")
//    val parsedRDD = obsDF.rdd.map(line => Row(line.toString().replace("\"", "")))
//    val parsedDF = 
//    parsedDF.take(3).foreach(println)
    
    val regexTokenizer = new RegexTokenizer()
    .setInputCol("rawSongs")
    .setOutputCol("arraySongs")
    .setPattern("\",\"|\"")
    
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
    .setIndices(Array.range(1, 13))
    //.setIndices(1 until 13 toArray)
    
    val arrayStages = Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer)
    val pipeline = new Pipeline().setStages(arrayStages)    
    
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val data = pipelineModel.transform(obsDF).drop("rawSongs","arraySongs","vectorSongs","years","label_double")
    
    //set the required paramaters (values obtained from task4)
    val learningAlg = new LinearRegression()
    .setMaxIter(8)
    .setRegParam(0.15)
    .setElasticNetParam(0.1)
    
    val model = learningAlg.fit(data)
    
    def getModelSummary(m : LinearRegressionModel) = {
      val modelSummary = m.summary
      println(s"numIterations: ${modelSummary.totalIterations}")
      println(s"RMSE: ${modelSummary.rootMeanSquaredError}")
      //coefficient of determination: indicates  the proportion of the variance in the dependent variable that is predictable from the independent variable
      println(s"r2: ${modelSummary.r2}")
    }
    
    getModelSummary(model)
    //Result: numIterations: 9 RMSE: 10.196397704373915 r2: 0.12981493307373249
    //make predictions on testing data
    val test1 = model.transform(data)
    println("Predicted values of the first 5 elements with model1:")
    test1.take(5).foreach(println)
    //Results:
    //[82.0,[412.49914,-3.89,0.223,412.499,145.029,6.0,0.0,0.49754690321416023,0.01,1.0,7.72750744853739,-266.37],81.36985129086139]
    //[83.0,[580.70159,-4.523,5.294,580.702,146.331,0.0,1.0,0.34371729849726773,0.004,1.0,3.3754281933060164,-235.827],81.25890747535479]
    //[87.0,[214.33424,-6.934,0.142,203.639,97.974,1.0,1.0,0.3993503120411163,0.005,1.0,3.146814151982389,-178.243],78.64480757596412]
    //[68.0,[238.34077,-8.831,0.102,227.347,96.928,5.0,1.0,0.2812941501103765,0.002,1.0,1.8835393671817502,-201.19],77.3490503187083]
    //[53.0,[127.65995,-11.829,0.427,114.347,119.257,5.0,1.0,0.29277104166666723,0.006,1.0,5.150343125000018,-375.263],74.54182124096002]
    sc.stop()
  }
}