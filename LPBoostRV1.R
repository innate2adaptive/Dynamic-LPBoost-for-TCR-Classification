#function model for LP Boost in R
## Original code by: Dimitrios Athanasakis (dathanasakis [at] gotbim [dot] com)
# and Yuxin Sun: yuxin.sun [at] ucl [dot] ac [dot] uk; using cvx package in Matlab

#X is a matrix of weak learners; each columns is a weak learner; each row  is a sample. 
#NOTE X is normally normalised for number of features per sample : either 1norm (divide by sum of features) or two norm (divide by sqrt(sum(features^2))).
#NOTE X MUST be centred : combine X with -X, so you end up with twice the number of nominal features but doubled. 

#Y is a classification vector; this must correspond to the order of rows in X, and must be {1,-1}.
#D is a weight for the penalty ;it should be set at 1/mv where m is the number of samples and v [0,1]  
#iter is the number of iterations before stopping 
#OUTPUT of function is a list of three elements containing 
#[[1]] the primal (sample weights);
#[[2]][1:L] the dual (feature) weights where L is the lenght of 
#[[3]] The list of features selected

LPBoostYS <-function (X, Y, D, iter){
#requires package lpSolve
library (lpSolve)
start_time<-proc.time()
# Output:
# model: structure variable of LPBoost model

#Improved code
# Original code by: Dimitrios Athanasakis (dathanasakis [at] gotbim [dot] com)
# Yuxin Sun: yuxin.sun [at] ucl [dot] ac [dot] uk

#[M, ~] = size(X);
M<-dim(X)[1]

# Initialise
counter <- 1;
a <- matrix(0,nrow=1,ncol=M)
beta <- 0;
u = rep(1,M)/M
eps<-1e-6
#calculate which triplet gives best discrimination by maximising uY*X
index<-c()
hypo<-c()
val<-1
#rm(u,beta)
while (counter <= iter & val > beta+eps)   { 
  #recalculate which triplet gives best discrimination by maximising uY*X
  uY<-u*Y
  weak_L<-uY %*% X
  val=max(weak_L)
  ind<-which.max(weak_L)
  index<-c(index,ind)
#  if (counter >1) 
  hypo = cbind(hypo,X[, ind])

  model<-LPcvx(hypo, Y, D)

#print iterations  
  cat("\nITERATION ",counter,"\t beta",  beta, "\t Val", val)
  counter <- counter + 1
  u<-model[[1]]
  beta<-model[[2]]
  #Z <- apply (hypo, 2 , function(x){x*Y})
  
}
cat ("\n user system elapsed \n",proc.time()[1]-start_time[1],"  ",proc.time()[2:3]-start_time[2:3],"\n")
#outputs both primal ("a") and dual "u" variables
l<-length(index)-1
list(model[[1]],model[[3]][1:l],index[1:l])
}

#######################################################################################################################
#this function does convex minimization to find the solution of the dual problem for the weak learner
LPcvx <-function(X1,Y,D){ 
# Input:
#X: hypothesis space up to date
# Y: label
# D: regularisation parameter
# Output:

# u: misclassification cost
# a: coefficients
# betaa: objective
#X1<-hypo

M<-dim(X1)[1]
N<-dim(X1)[2]
Z <- apply (X1, 2 , function(x){x*Y})
#these are the constraints that sum less than beta
beta.con<-cbind(t(Z),rep(-1,N))
#constraint that sum u =1
sumu.con<-c(rep(1,M),0)
#constraint that all u < D
UltD.con<-cbind(diag(M),rep(0,M))
#combine these constraints 
f.con<-rbind(beta.con,sumu.con,UltD.con)
f.dir<-c(rep("<=",N),"==",rep("<=",M))
f.rhs<-c(rep(0,N),1,rep(D,M))
#beta<-100
#trip<-c()

#the objective function is to minimize beta, the last coefficient
f.obj<-c(rep(0,M),1)
LP<-lp("min",f.obj,f.con,f.dir,f.rhs,compute.sens=TRUE)
#LP<-lp("min",f.obj,f.con,f.dir,f.rhs)
beta<-LP$objval
u<-LP$solution[1:M]
#duals coefficients are before  dual constraints ?
end <-dim(f.con)[1]+1
a<-LP$duals
list(u,beta,LP$duals)
        }
#a(a<=eps)=0;