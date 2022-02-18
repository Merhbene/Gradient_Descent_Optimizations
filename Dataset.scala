// Databricks notebook source
import scala.collection.mutable.ArrayBuffer


// COMMAND ----------

//My Data
val data_shape=1000
case class data(x1 : Double, x2:Double , y:Double)

val D=for (i <- 1 to data_shape) yield (data(i.asInstanceOf[Double],(i+1).asInstanceOf[Double],(5*i+2).asInstanceOf[Double]))
val DS=D.toDS()
DS.show()


// COMMAND ----------

DS.rdd.getNumPartitions

// COMMAND ----------

val ds = DS.repartition(10)

// COMMAND ----------

ds.rdd.getNumPartitions

// COMMAND ----------

ds.show()

// COMMAND ----------

import org.apache.spark.sql.functions.spark_partition_id
var Dataset =ds.withColumn("partitionID", spark_partition_id)


// COMMAND ----------

Dataset.show()

// COMMAND ----------

def sum(a:ArrayBuffer[Double], b:ArrayBuffer[Double])=
{
  var c = ArrayBuffer[Double]()
  for (i <- 0 until a.size)
  {
    c+=a(i)+b(i)
  }
  c

}
val a = ArrayBuffer(1.0,2.0,3.0)
val b= ArrayBuffer(2.0,4.0,1.0)
sum(a,b)

// COMMAND ----------

def prod_by_scal(a:ArrayBuffer[Double],b:Double)={
  val n=a.size
  var c=ArrayBuffer[Double]()
  for (i<-0 until n )
     { 
       c+=a(i)*b
     }
   c
}

prod_by_scal(a,10)

// COMMAND ----------

def prod_scal (x:ArrayBuffer[Double], y:ArrayBuffer[Double]):Double =
{
  val T=(x,y).zipped map(_*_) 
  var sum=0.0
  for (i<-0 until T.length)
  {
    sum+=T(i)
  }
  sum
}

prod_scal(a,b)

// COMMAND ----------

//substract two arrays 
def minus(a:ArrayBuffer[Double],b:ArrayBuffer[Double])=
{ 
  var c=ArrayBuffer[Double]()
  for (i<-0 until b.length )
     { 
       c+=a(i)-b(i)
     }
   c
}

// COMMAND ----------

def Gradient(a:(ArrayBuffer[Double],Double),w:ArrayBuffer[Double])=
{
   var f=2*(prod_scal(a._1,w)-a._2)
   var grad=prod_by_scal(a._1,f) 
   grad 
  
}

// COMMAND ----------

//SGD
def SGD( DS:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double , W:ArrayBuffer[Double] )={
  var new_W  = W
  var L=DS.select($"x1",$"x2",$"y").collect()
  for (i<-L)

  { 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((x,y),new_W)
    new_W =  minus(new_W ,prod_by_scal(grad,eta )) 
  }
  
  new_W
}


var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 3
val num_partition=10

for (i<-1 to epoch)
{
  for(i<- 0 to num_partition-1)
  {

    val curr_DS=Dataset.filter($"partitionID" === i )
    val nww=SGD( curr_DS , eta , nw )
    nw=nww
    println(nw)
  }
}

// COMMAND ----------

//Mini Batch SGD
def SGD_miniBatch( DS:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double , W:ArrayBuffer[Double] )={
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}
  
  var L=DS.select($"x1",$"x2",$"y").collect()


  for (i<-L)

  { 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((x,y),new_W)
    
    gradient = sum(gradient , grad )
    
  }
  
  gradient=prod_by_scal(gradient,1/(DS.count.toFloat))
  new_W =  minus(new_W ,prod_by_scal(gradient,eta ))
  new_W
}


var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch=3
val mini_batch_rate=10

for (j<-1 to epoch)
 {
   
  for(i<- 0 to mini_batch_rate-1){

    val curr_DS=Dataset.filter($"partitionID" === i )
    val nww=SGD_miniBatch( curr_DS , eta , nw )
    nw=nww
    println(nw)
  }
}

// COMMAND ----------

//Momentum
def Momentum( DS:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double, Gamma:Double, V:ArrayBuffer[Double], W:ArrayBuffer[Double] )=
{
  var new_W  = W
  var new_V=V
  
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}
  
  
  var L=DS.select($"x1",$"x2",$"y").collect()
  for (i<-L)

  { 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((x,y),new_W)
    new_V=sum(prod_by_scal(new_V,Gamma),prod_by_scal(grad,eta))
    new_W =  minus(new_W ,new_V)
    
  }
  (new_W,new_V)
}



var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val Gamma=0.9

var new_V = ArrayBuffer(0.0,0.0)

val epoch=3
for (j<-1 to epoch)
{
  
    for(i<- 0 to 9)
  {

      val curr_DS=Dataset.filter($"partitionID" === i )
      val (nww,nwV)=Momentum( curr_DS , eta ,Gamma , new_V , nw )
      nw=nww
      new_V=nwV
      println(nw)
   }
}

// COMMAND ----------

//Adagrad

import scala.math.sqrt
def Adagrad( DS:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double,  W:ArrayBuffer[Double] )=
{
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  
  var L=DS.select($"x1",$"x2",$"y").collect()
  for (i<-L)

  { 
    var x1=i.get(0).asInstanceOf[Double]
    var x2=i.get(1).asInstanceOf[Double]
    var x=ArrayBuffer(x1,x2)
    var y =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((x,y),new_W)
    
    var G=prod_scal(grad,grad)+ 1e-8
    var new_eta = eta * (1/sqrt(G))
    new_W=minus(new_W,prod_by_scal(grad,new_eta))
  }
  
  new_W
}


var nw=ArrayBuffer(0.0,0.0)
val eta = 0.0025
var new_eta=eta

val epoch=3
for (j<-1 to epoch)
{
  for(i<- 0 to 9)
  {

      val curr_DS=Dataset.filter($"partitionID" === i )
      val nww=Adagrad( curr_DS , eta , nw )
      nw=nww
      println(nw)
  }

}

// COMMAND ----------
