// Databricks notebook source
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt


// COMMAND ----------

val data=(1 to 1000).toArray
val pairs=data.map(x=>(ArrayBuffer(x.asInstanceOf[Double],(x+1).asInstanceOf[Double]),(5*x+2).asInstanceOf[Double]))

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

//calcul du gradient
def Gradient(a:(ArrayBuffer[Double],Double),w:ArrayBuffer[Double])=
{
   var f=2*(prod_scal(a._1,w)-a._2)
   var grad=prod_by_scal(a._1,f) 
   grad 
  
}
var c = ArrayBuffer(3.0,1.0,5.0)
val a = ArrayBuffer(1.0,2.0,3.0)
Gradient((a,3),c)  


// COMMAND ----------

//GD
def GD( p:Array[(ArrayBuffer[Double] , Double)],W:ArrayBuffer[Double] )={
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}

  for (i <- p)
  {
    var grad = Gradient(i,new_W)
    gradient = sum(gradient , grad )
    
  }
  
  gradient
}


val rdds=sc.parallelize(pairs,1).repartition(10)
val partitions=rdds.glom.zipWithIndex
var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 20
val num_part=10
 
var gradient = new ArrayBuffer[Double](nw.length)
for (j<-1 to nw.length ){gradient+=0.0}
var new_W=nw

for (i<-1 to epoch)
{
  for(i<- 0 to num_part-1)
  {

    val currpart=partitions.filter(p=>p._2==i)
    val grad=currpart.map(x=>GD( x._1 ,nw )).collect
    gradient = sum(gradient , grad(0) )
  }
  
  gradient=prod_by_scal(gradient,1/(pairs.length.toFloat))
  new_W =  minus(new_W ,prod_by_scal(gradient,eta )) 
  nw=new_W
  println(nw)
}

// COMMAND ----------

//SGD
def SGD( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double , W:ArrayBuffer[Double] )={
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (i <- p)
  {
    var grad = Gradient(i,new_W)
    new_W =  minus(new_W ,prod_by_scal(grad,eta )) 
  }
  
  new_W
}


val rdds=sc.parallelize(pairs,1).repartition(10)
val partitions=rdds.glom.zipWithIndex
var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 3
val num_part=10

for (i<-1 to epoch)
{
  for(i<- 0 to num_part-1)
  {

    val currpart=partitions.filter(p=>p._2==i)
    val nww=currpart.map(x=>SGD( x._1 , eta , nw )).collect
    nw=nww(0)
    println(nw)
  }
}

// COMMAND ----------

//SGD MINI BATCH
def SGD_miniBatch( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double , W:ArrayBuffer[Double] )={
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}

  for (i <- p)
  {
    var grad = Gradient(i,new_W)
    gradient = sum(gradient , grad )
    
  }
  
  gradient=prod_by_scal(gradient,1/(p.length.toFloat))
  new_W =  minus(new_W ,prod_by_scal(gradient,eta ))
  new_W
}

val rdds=sc.parallelize(pairs,1).repartition(10)
val partitions=rdds.glom.zipWithIndex
val mini_batch_rate=10

var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch=3
for (j<-1 to epoch)
 {
   
  for(i<- 0 to mini_batch_rate-1){

    val currpart=partitions.filter(p=>p._2==i)
    val nww=currpart.map(x=>SGD_miniBatch( x._1 , eta , nw )).collect
    nw=nww(0)
    println(nw)
  }
}

// COMMAND ----------

//MOMENTUM
def Momentum( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double, Gamma:Double, V:ArrayBuffer[Double], W:ArrayBuffer[Double] )=
{
  var new_W  = W
  var new_V=V
  
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}
  
  
  for (i <- p)
  {
    var grad = Gradient(i,new_W)
    new_V=sum(prod_by_scal(new_V,Gamma),prod_by_scal(grad,eta))
    new_W =  minus(new_W ,new_V)
    
  }
  (new_W,new_V)
}

val rdds=sc.parallelize(pairs,1).repartition(10)
val partitions=rdds.glom.zipWithIndex
val num_part=10

var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val Gamma=0.9

var new_V = ArrayBuffer(0.0,0.0)

val epoch=3
for (j<-1 to epoch)
{
  
    for(i<- 0 to num_part-1)
  {

      val currpart=partitions.filter(p=>p._2==i)
      val nww=currpart.map(x=>Momentum( x._1 , eta ,Gamma , new_V , nw )).collect
      nw=nww(0)._1
      new_V=nww(0)._2
      println(nw)
   }
}

// COMMAND ----------

//ADAGRAD
def Adagrad( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double,  W:ArrayBuffer[Double] )=
{
  var new_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (i <- p)
  {
    
    var grad = Gradient(i,new_W)
    var G=prod_scal(grad,grad)+ 1e-8
    var new_eta = eta * (1/sqrt(G))
    new_W=minus(new_W,prod_by_scal(grad,new_eta))
  }
  
  new_W
}


val rdds=sc.parallelize(pairs,1).repartition(10)
val partitions=rdds.glom.zipWithIndex
val num_part=10

var nw=ArrayBuffer(0.0,0.0)
val eta = 0.0025
var new_eta=eta

val epoch=3
for (j<-1 to epoch)
{
  for(i<- 0 to num_part-1)
  {

      val currpart=partitions.filter(p=>p._2==i)
      val nww=currpart.map(x=>Adagrad( x._1 , eta , nw )).collect
      nw=nww(0)
      println(nw)
  }

}



// COMMAND ----------
