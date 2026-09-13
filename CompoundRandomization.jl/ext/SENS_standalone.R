#' The SENS procedure for controlling the false discovery rate
#'
#' @docType package
#' @name CEN
library(HDInterval);

#' SENS
#'
#'This function runs the SENS procedure, constructing the calibrated empirical null and the test data, and constructing auxiliary statistics under heteroskedasticity, choosing the cutoff and selectiing the locations.
#'
#' @param X the matrix or data frame of observation
#' @param alpha targeted FDR (false discovery rate) level
#' @param option Gasssian case for the null distribution or General case for the null distribution
#' #'
#' @return A list containing the following components:
#' \item{de}{decision for each location (0 or 1)}
#' \item{th}{threshold for SENS procedure}
#'
#' @examples
#' X <- matrix(rep(c(0,3),c(800,200))+rnorm(1000),ncol=5,nrow=200);
#' SENS(X,0.05,'Gaussian');
#'
#' @importFrom density
#' @importFrom stats density dt pnorm pt qnorm var rnorm
#' @export
SENS <- function(X,alpha,option=c('Gaussian','General')){
  #Validate input types.
  if (is.data.frame(X)){
    X.names=names(X)
    X = as.matrix(X,rownames.force=F)
  } else if (is.matrix(X)){
    X.names=colnames(X)
  } else{
    stop('Input X must be a matrix or data frame')}

  #Validate input dimensions.
  n <- ncol(X); m <- nrow(X);
  stopifnot(n>1)

  #Split the observations into two parts
  n1 <- ceiling(n/2); n2=n-n1;
  n11 <- sample(1:n,size=n1,replace=F);
  X1 <- X[,n11]; X2 <- X[,-n11];

  if(n>=4){
    #Calculate the mean of X1, X2 for each location
    X1.mean <- rowMeans(X1);
    X2.mean <- rowMeans(X2);

    V <- sqrt(n1*n2/n)*(X1.mean+X2.mean);
    V0 <- sqrt(n1*n2/n)*(X1.mean-X2.mean);
    #Calculate the standard deviation of X1, X2 for each location
    X1.var <- apply(X1,1,var);
    X2.var <- apply(X2,1,var);
    #Calculate the main statistics used later
    S <- sqrt(((n1-1)*X1.var+(n2-1)*X2.var)/(n-2));
  }else if(n==3){
    #Calculate the mean of X1, X2 for each location
    X1.mean <- rowMeans(X1);
    X2.mean <- X2;

    V <- sqrt(n1*n2/n)*(X1.mean+X2.mean);
    V0 <- sqrt(n1*n2/n)*(X1.mean-X2.mean);
    #Calculate the standard deviation of X1, X2 for each location
    X1.var <- apply(X1,1,var);
    #Calculate the main statistics used later
    S <- X1.var;
  }else{
    V <- sqrt(n1*n2/n)*(X1+X2);
    V0 <- sqrt(n1*n2/n)*(X1-X2);
    S <-rep(1,m);
  }
  t <- V/S; t0 <- V0/S;
  t[which(is.na(t))]<-0; t0[which(is.na(t0))]<-0

  # Use t and t0 as the test and calibration
  # Calculate the estimated SENS statistics based on density estimation for denominator
  if(n>=3){
  t <- inverseCDF(pt(t,df=n-2),pnorm); t0 <- inverseCDF(pt(t0,df=n-2),pnorm);}

  if (option=='Gaussian'){
    Em<-EstNull.func(c(t,t0))
    density_values0<-dnorm(c(t,t0),Em$mu,Em$s)
  }else{
    t00 <- ifelse(abs(t) < abs(t0), t, t0)
    density_values0 <- numeric(2*m)
    bw0<-density(c(t00,-t00))$bw
    for (i in 1:(2*m)){
      density_values0[i]<-sum(dnorm((c(t,t0)[i]-c(t00,-t00))/bw0))/(2*m*bw0)
    }
  }

  sens.numerator <- density_values0
  sens.denominator <- approxfun(density(c(t,t0))$x,density(c(t,t0))$y)(c(t,t0));
  sens.Est <- sens.numerator[1:m]/sens.denominator[1:m];
  sens.Est0 <- sens.numerator[(m+1):(2*m)]/sens.denominator[(m+1):(2*m)];
  t<-sapply(1:m, function(i) sign(sens.Est0[i]-sens.Est[i])*(max(exp(-sens.Est[i]),exp(-sens.Est0[i]))))
  y = bc.func(t,alpha)
  return(y)
}

bc.func<-function(W,q){
  m=length(W)
  # The decision rule:
  hps <- rep(0, m)
  candidate <- c()
  for (i in 1:m) {
    denom <- (W <= -abs(W[i]))
    numer <- (W >= abs(W[i]))
    if ((1+sum(denom)) / max(sum(numer), 1) <= q) {
      candidate <- c(candidate, abs(W[i]))
    }
  }

  if (length(candidate) == 0) {
    threshold <- Inf
    reject <- integer(0)
  } else {
    threshold <- min(candidate)
    reject <- which(W >= threshold)
  }

  hps[reject] <- 1

  y <- list(th = threshold, re = reject, de = hps)
  return(y)
}

EstNull.func<-function (x,gamma=0.08){
  # x is a vector of z-values
  # gamma is a parameter, default is 0.1
  # output the estimated mean and standard deviation

  n = length(x)
  t = c(1:10000)/200

  gan    = n^(-gamma)
  that   = 0
  shat   = 0
  uhat   = 0
  epshat = 0

  phiplus   = rep(1,10000)
  phiminus  = rep(1,10000)
  dphiplus  = rep(1,10000)
  dphiminus = rep(1,10000)
  phi       = rep(1,10000)
  dphi      = rep(1,10000)

  for (i in 1:10000) {
    s = t[i]
    phiplus[i]   = mean(cos(s*x))
    phiminus[i]  = mean(sin(s*x))
    dphiplus[i]  = -mean(x*sin(s*x))
    dphiminus[i] = mean(x*cos(s*x))
    phi[i]       = sqrt(phiplus[i]^2 + phiminus[i]^2)
  }

  ind = min(c(1:10000)[(phi - gan) <= 0])
  tt = t[ind]
  a  = phiplus[ind]
  b  = phiminus[ind]
  da = dphiplus[ind]
  db = dphiminus[ind]
  c  = phi[ind]

  that   = tt
  shat   = -(a*da + b*db)/(tt*c*c)
  shat   = sqrt(shat)
  uhat   = -(da*b - db*a)/(c*c)
  epshat = 1 - c*exp((tt*shat)^2/2)

  return(musigma=list(mu=uhat,s=shat))
}
