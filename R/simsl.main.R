
#' Single-index models with a surface-link (main function)
#'
#' \code{simsl} is the wrapper function for fitting a single-index model with a surface-link (SIMSL).
#' The function estimates a linear combination (a single-index) of baseline covariates X, and models a nonlinear interactive structure between the single-index and a treatment variable defined on a continuum, by estimating a smooth link function on the index-treatment domain. The resulting \code{simsl} object can be used to estimate an optimal dose rule for a new patient with baseline clinical information.
#'
#' SIMSL captures the effect of covariates via a single-index and their interaction with the treatment via a 2-dimensional smooth link function.
#' Interaction effects are determined by shapes of the link surface.
#' The SIMSL allows comparing different individual treatment levels and constructing individual treatment rules,
#' as functions of a biomarker signature (single-index), efficiently utilizing information on patient’s characteristics.
#' The resulting \code{simsl} object can be used to estimate an optimal dose rule for a new patient with baseline clinical information.
#'
#'
#' @param y  a n-by-1 vector of treatment outcomes; y is a member of the exponential family; any distribution supported by \code{mgcv::gam}; y can also be an ordinal categorial response with \code{R} categories taking a value from 1 to \code{R}.
#' @param A  a n-by-1 vector of treatment variable; each element is assumed to take a value on a continuum.
#' @param X  a n-by-p matrix of baseline covarates.
#' @param Xm  a n-by-q design matrix associated with an X main effect model; the defult is \code{NULL} and it is taken as a vector of zeros
#' @param family  specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; can be any family supported by \code{mgcv::gam}; can also be "ordinal", for an ordinal categorical response y.
#' @param R   the number of response categories for the case of family = "ordinal".
#' @param bs basis type for the treatment (A) and single-index domains, respectively; the defult is "ps" (p-splines); any basis supported by \code{mgcv::gam} can be used, e.g., "cr" (cubic regression splines); see \code{mgcv::s} for detail.
#' @param k  basis dimension for the treatment (A) and single-index domains, respectively.
#' @param m  a length 2 list (e.g., m=list(c(2,3), c(2,2))), for the treatment (A) and single-index domains, respectively, where each element specifies the order of basis and penalty (note, for bs="ps", c(2,3) means a 2nd order P-spline basis (cubic spline) and a 3rd order difference penalty; the default "NA" sets c(2,2) for each domain); see \code{mgcv::s} for details.
#' @param sp  a vector of smoothing parameters;  Smoothing parameters must be supplied in the order that the smooth terms appear in the model formula (i.e., A, and then the single-index); negative elements indicate that the parameter should be estimated, and hence a mixture of fixed and estimated parameters is possible; see \code{mgcv::gam} for detail.
#' @param knots  a list containing user-specified knot values to be used for basis construction, for the treatment (A) and single-index domains, respectively.
#' @param sep.A.effect   If \code{TRUE}, the g term of SIMSL is further decomposed into: the A main effect + the A-by-X interaction effect; the default is \code{FALSE}.
#' @param mc  a length 2 vector indicating which marginals (i.e., A and the single-index, respectively) should have centering (i.e., the sum-to-zero) constraints applied; the default is \code{mc = c(TRUE, FALSE)} (see \code{mgcv::te} for detail of the constraint), which is sufficient for the so-called "orthogonality" constraint of the SIMSL.
#' @param method  the smoothing parameter estimation method; "GCV.Cp" to use GCV for unknown scale parameter and Mallows' Cp/UBRE/AIC for known scale; any method supported by \code{mgcv::gam} can be used.
#' @param beta.ini  an initial value for \code{beta.coef}; a p-by-1 vector; the defult is \code{NULL}, in which case a linear model estimate is used.
#' @param ind.to.be.positive  for identifiability of the solution \code{beta.coef}, the user can restrict the jth (e.g., j=1) component of \code{beta.coef} to be positive; by default, we match the "overall" sign of \code{beta.coef} with that of the linear estimate (i.e., the initial estimate), by restricting the inner product between the two to be positive.
#' @param random.effect  if \code{TRUE}, as part of the main effects, the user can incorporate z-specific random intercepts.
#' @param z  a factor that specifies the random intercepts when \code{random.effect = TRUE}.
#' @param gamma  increase this beyond 1 to produce smoother models. \code{gamma} multiplies the effective degrees of freedom in the GCV or UBRE/AIC (see \code{mgcv::gam} for detail); the default is 1.
#' @param pen.order 0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of \code{beta.coef}.
#' @param lambda  a regularization parameter associated with the penalized LS for \code{beta.coef} update.
#' @param max.iter  an integer specifying the maximum number of iterations for \code{beta.coef} update.
#' @param eps.iter a value specifying the convergence criterion of algorithm.
#' @param trace.iter if \code{TRUE}, trace the estimation process and print the differences in \code{beta.coef}.
#' @param center.X   if \code{TRUE}, center X to have zero mean.
#' @param scale.X    if \code{TRUE}, scale X to have unit variance.
#' @param uncons.final.fit    if \code{TRUE}, once the convergence in the estimates of \code{beta.coef} is reached, include the main effect associated with the fitted single-index (beta.coef'X) to the final surface-link estimate.
#' @param bootstrap if \code{TRUE}, compute bootstrap confidence intervals for the single-index coefficients, \code{beta.coef}; the default is \code{FALSE}.
#' @param boot.conf  a value specifying the confidence level of the bootstrap confidence intervals; the defult is \code{boot.conf = 0.95}.
#' @param nboot  when \code{bootstrap=TRUE}, a value specifying the number of bootstrap replications.
#' @param seed  when  \code{bootstrap=TRUE}, randomization seed used in bootstrap resampling.
#'
#'
#' @return a list of information of the fitted SIMSL including
#'  \item{beta.coef}{ the estimated single-index coefficients.} \item{g.fit}{a \code{mgcv:gam} object containing information about the estimated 2-dimensional link function.} \item{beta.ini}{the initial value used in the estimation of \code{beta.coef}} \item{beta.path}{solution path of \code{beta.coef} over the iterations} \item{d.beta}{records the change in \code{beta.coef} over the solution path, \code{beta.path}} \item{X.scale}{sd of pretreatment covariates X} \item{X.center}{mean of pretreatment covariates X} \item{A.range}{range of the observed treatment variable A} \item{p}{number of baseline covariates X} \item{n}{number of subjects} \item{boot.ci}{\code{boot.conf}-level bootstrap CIs (LB, UB) associated with \code{beta.coef}} \item{boot.mat}{a (nboot x p) matrix of bootstrap estimates of  \code{beta.coef}}
#'
#' @author Park, Petkova, Tarpey, Ogden
#' @import mgcv stats
#' @seealso \code{pred.simsl},  \code{fit.simsl}
#' @export
#'
#' @examples
#'

#'set.seed(1234)
#'n <- 200
#'n.test <- 500
#'
#'
#'## simulation 1
#'# generate training data
#'p <- 30
#'X <- matrix(runif(n*p,-1,1),ncol=p)
#'A <- runif(n,0,2)
#'D_opt <- 1 + 0.5*X[,2] + 0.5*X[,1]
#'mean.fn <- function(X, D_opt, A){ 8 + 4*X[,1] - 2*X[,2] - 2*X[,3] - 25*((D_opt-A)^2) }
#'mu <-   mean.fn(X, D_opt, A)
#'y <- rnorm(length(mu),mu,1)
#'# fit SIMSL
#'simsl.obj <- simsl(y=y, A=A, X=X)
#'
#'# generate testing data
#'X.test <- matrix(runif(n.test*p,-1,1),ncol=p)
#'A.test <- runif(n.test,0,2)
#'f_opt.test <- 1 + 0.5*X.test[,2] + 0.5*X.test[,1]
#'pred <- pred.simsl(simsl.obj, newX= X.test)  # make prediction based on the estimated SIMSL
#'value <- mean(8 + 4*X.test[,1] - 2*X.test[,2] - 2*X.test[,3] - 25*((f_opt.test- pred$trt.rule)^2))
#'value  # "value" of the estimated treatment rule; the "oracle" value is 8.
#'
#'
#'## simulation 2
#'p <- 10
#'# generate training data
#'X <- matrix(runif(n*p,-1,1),ncol=p)
#'A <- runif(n,0,2)
#'f_opt <- I(X[,1] > -0.5)*I(X[,1] < 0.5)*0.6 + 1.2*I(X[,1] > 0.5) +
#'  1.2*I(X[,1] < -0.5) + X[,4]^2 + 0.5*log(abs(X[,7])+1) - 0.6
#'mu <-   8 + 4*cos(2*pi*X[,2]) - 2*X[,4] - 8*X[,5]^3 - 15*abs(f_opt-A)
#'y  <- rnorm(length(mu),mu,1)
#'Xq <- cbind(X, X^2)  # include a quadratic term
#'# fit SIMSL
#'simsl.obj <- simsl(y=y, A=A, X=Xq)
#'
#'# generate testing data
#'X.test <- matrix(runif(n.test*p,-1,1),ncol=p)
#'A.test <- runif(n.test,0,2)
#'f_opt.test <- I(X.test[,1] > -0.5)*I(X.test[,1] < 0.5)*0.6 + 1.2*I(X.test[,1] > 0.5) +
#'  1.2*I(X.test[,1] < -0.5) + X.test[,4]^2 + 0.5*log(abs(X.test[,7])+1) - 0.6
#'Xq.test <- cbind(X.test, X.test^2)
#'pred <- pred.simsl(simsl.obj, newX= Xq.test)  # make prediction based on the estimated SIMSL
#'value <- mean(8 + 4*cos(2*pi*X.test[,2]) - 2*X.test[,4] - 8*X.test[,5]^3 -
#'               15*abs(f_opt.test-pred$trt.rule))
#'value  # "value" of the estimated treatment rule; the "oracle" value is 8.
#'
#'
#'\donttest{
#'  ### air pollution data application
#'  data(chicago); head(chicago)
#'  chicago <- chicago[,-3][complete.cases(chicago[,-3]), ]
#'  chicago <- chicago[-c(2856:2859), ]  # get rid of the gross outliers in y
#'  chicago <- chicago[-which.max(chicago$pm10median), ]  # get rid of the gross outliers in x
#'
#'  # create lagged variables
#'  lagard <- function(x,n.lag=5) {
#'    n <- length(x); X <- matrix(NA,n,n.lag)
#'    for (i in 1:n.lag) X[i:n,i] <- x[i:n-i+1]
#'    X
#'  }
#'  chicago$pm10 <- lagard(chicago$pm10median)
#'  chicago <- chicago[complete.cases(chicago), ]
#'  # create season varaible
#'  chicago$time.day <- round(chicago$time %%  365)
#'
#'  # fit SIMSL for modeling the season-by-pm10 interactions on their effects on outcomes
#'  simsl.obj <- simsl(y=chicago$death, A=chicago$time.day, X=chicago[,7], bs=c("cc","ps"),
#'                     ind.to.be.positive = 1, family="poisson", method = "REML",
#'                     bootstrap =TRUE, nboot=1) # nboot =500
#'  simsl.obj$beta.coef  # the estimated single-index coefficients
#'  summary(simsl.obj$g.fit)
#'  round(simsl.obj$boot.ci,3)
#'
#'
#'  additive.fit  <- mgcv::gam(chicago$death ~
#'                               s(simsl.obj$g.fit$model[,3], k=8, bs="ps") +
#'                               s(chicago$time.day, k=8, bs="cc"),
#'                             family = poisson(), method = "REML")
#'  plot(additive.fit, shift= additive.fit$coefficients[1], select=2,
#'       ylab= "Linear predictor", xlab= "A", main = expression(paste("Individual A effect")))
#'  plot(additive.fit, shift= additive.fit$coefficients[1], select = 1,
#'       xlab= expression(paste(beta*minute,"x")), ylab= " ",
#'       main = expression(paste("Individual ", beta*minute,"x effect")))
#'  mgcv::vis.gam(simsl.obj$g.fit, view=c("A","single.index"), theta=-135, phi = 30,
#'                color="heat", se=2,ylab = "single-index", zlab = " ",
#'                main=expression(paste("Interaction surface ")))
#'
#'
#'
#'  ### Warfarin data application
#'  data(warfarin)
#'  X <- warfarin$X
#'  A <- warfarin$A
#'  y <- -abs(warfarin$INR - 2.5)  # the target INR is 2.5
#'  X[,1:3] <- scale(X[,1:3]) # standardize continuous variables
#'
#'  # Estimate the main effect, using an additive model
#'  mu.fit <- mgcv::gam(y-mean(y)  ~ X[, 4:13] +
#'                        s(X[,1], k=5, bs="ps")+
#'                        s(X[,2], k=5, bs="ps") +
#'                        s(X[,3], k=5, bs="ps"), method="REML")
#'  summary(mu.fit)
#'  mu.hat <- predict(mu.fit)
#'  # fit SIMSL
#'  simsl.obj <- simsl(y, A, X, Xm= mu.hat, scale.X = FALSE, center.X=FALSE, method="REML",
#'                     bootstrap = TRUE, nboot=1) # nboot = 500
#'  simsl.obj$beta.coef
#'  round(simsl.obj$boot.ci,3)
#'  mgcv::vis.gam(simsl.obj$g.fit, view=c("A","single.index"), theta=55, phi = 30,
#'                color="heat", se=2, ylab = "single-index", zlab = "Y",
#'                main=expression(paste("Interaction surface ")))
#'}
#'
simsl <- function(y, # a n-by-1 vector of treatment outcomes; y is a member of the exponential family; any distribution supported by mgcv::gam.
                  A, # a n-by-1 vector of treatment (i.e., dose) variable (assumed to be on continumm).
                  X, # a n-by-p matrix of covariates.
                  Xm = NULL,  # a n-by-q design matrix assocaited with the X main effect.
                  family = "gaussian",  # specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; can be any family supported by mgcv::gam; can also be "ordinal", for an ordianal categorical response y.
                  R = NULL,    # for the case of family = "ordinal", the number of response catergories.
                  bs = c("ps", "ps"),  # basis type for the treatment (A) and the single-index domains, respectively; the defult is "ps" (p-splines); any basis supported by mgcv::gam can be used, e.g., "cr" (cubic regression splines).
                  k = c(8, 8),  # basis dimension for the treatment (A) and the single-index domains, respectively.
                  m = list(NA, NA),  # a list of length 2 (e.g., m=list(c(2,3), c(2,2))), each element pecifying the order of basis and penality (for bs="ps", c(2,3) specifies a 2nd order P-spline basis (cubic spline) with a 3rd order difference penalty; the default (i.e., "NA") sets c(2,2) to each domain), for the treatment (A) and the single-index domains, respectively; see mgcv::s for details.
                  sp = NULL, # a vector of smoothing parameters;  Smoothing parameters must be supplied in the order that the smooth terms appear in the model formula (i.e., A and then the single-index); negative elements indicate that the parameter should be estimated, and hence a mixture of fixed and estimated parameters is possible; see mgcv::gam for detail.
                  knots = NULL,   # a list containing user-specified knot values to be used for basis construction, for the treatment (A) and the single-index domains, respectively.
                  sep.A.effect = FALSE,  # If TRUE, the g term of SIMSL is further decomposed into: the A main effect term + the A-by-X interaction effect term.
                  mc = c(TRUE, FALSE), # a length 2 vector indicating which marginals (i.e., A and the single-index, respectively) should have centering (i.e., sum-to-zero) constraints applied; the default is mc = c(TRUE, FALSE) (see mgcv::te for detail of the constraint), which is sufficient for the so-called "orthogonality" constraint of the SIMSL.
                  method = "GCV.Cp",  # the smoothing parameter estimation method; can be "GCV.Cp", "REML" and "ML"; see mgcv::gam for detail.
                  beta.ini = NULL,  # an initial value for beta.coef; a p-by-1 vector; the defult is NULL.
                  ind.to.be.positive = NULL,  # for identifiability of the solution beta.coef, we can restrict the jth (e.g., j=1) component of beta.coef to be positive; by default, we match the "overall" sign of the single-index with that of the linear estimate (i.e., the initial estimate), by keeping the inner product between the two to be positive.
                  random.effect = FALSE,  # if TRUE, incorporate the z-specific random intercepts, as part of the main effects.
                  z = NULL, # a factor that specifies random intercepts, if random.effect = TRUE.
                  gamma =1,  # increase this beyond 1 to produce smoother models. gamma multiplies the effective degrees of freedom in the GCV or UBRE/AIC (see mgcv::gam for detail).
                  pen.order = 0,  # pen.order 0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of beta.coef.
                  lambda = 0,     # a regularziation parameter associated with the penalized LS of beta.coef.
                  max.iter = 10,  # an integer specifying the maximum number of iterations for beta.coef update.
                  eps.iter = 0.01, # a value specifying the convergence criterion of algorithm.
                  trace.iter = TRUE, # if TRUE, trace the estimation process and print the differences in beta.coef.
                  center.X=TRUE,  # if TRUE, center X to have zero mean.
                  scale.X=TRUE,   # if TRUE, scale X to have unit variance.
                  uncons.final.fit = TRUE, # if TRUE, once the convergece in the estimates of beta.coef is reached, include the main effect associated with the fitted single-index (beta.coef'X) to the final surface-link estimate.
                  bootstrap = FALSE,  # if TRUE, compute bootstrap confidence intervals for the single-index coefficients, beta.coef; the default is FALSE.
                  nboot= 200,  # when bootstrap=TRUE, a value specifying the number of bootstrap replications.
                  boot.conf = 0.95,  #a value specifying the confidence level of the bootstrap confidence intervals; the defult is boot.conf = 0.95.
                  seed= 1357)  # when bootstrap=TRUE, randomization seed used in bootstrap resampling.
{

  simsl.obj <- fit.simsl(y=y, A=A, X=X, Xm=Xm,
                         family=family, R=R,
                         bs =bs, k = k, m=m, knots=knots, sp=sp,
                         sep.A.effect=sep.A.effect,
                         mc=mc, method=method,
                         beta.ini = beta.ini,
                         ind.to.be.positive=ind.to.be.positive,
                         random.effect=random.effect, z=z,
                         pen.order = pen.order, lambda = lambda,
                         max.iter = max.iter, gamma=gamma, trace.iter=trace.iter,
                         center.X= center.X, scale.X= scale.X,
                         uncons.final.fit= uncons.final.fit)


  boot.mat = boot.ci <- NULL
  if(bootstrap){
    set.seed(seed)
    indices <- 1:simsl.obj$n
    if(is.null(Xm)) Xm <- rep(0,simsl.obj$n)
    if(is.null(z))  z <-  rep(0,simsl.obj$n)
    Xm <- as.matrix(Xm)
    boot.mat <- matrix(0, nboot, simsl.obj$p)
    for(i in 1:nboot){
      boot.indices <- sample(indices, simsl.obj$n, replace = TRUE)
      tmp  <- fit.simsl(y=y[boot.indices], A = A[boot.indices], X = X[boot.indices,],
                        Xm = Xm[boot.indices,],
                        family=family, R=R,
                        bs =bs, k = k, m=m, knots=knots, sp= sp,
                        sep.A.effect=sep.A.effect,
                        mc=mc, method= method,
                        beta.ini = beta.ini,
                        ind.to.be.positive=ind.to.be.positive,
                        random.effect= random.effect, z=z[boot.indices],
                        pen.order = pen.order, lambda = lambda, max.iter = max.iter, gamma=gamma, trace.iter=trace.iter,
                        center.X= center.X, scale.X= scale.X,
                        uncons.final.fit = uncons.final.fit
      )$beta.coef

      if(simsl.obj$beta.coef%*%tmp > simsl.obj$beta.coef%*%(-tmp)){
        boot.mat[i,] <-  tmp
      }else{
        boot.mat[i,] <- -tmp
      }
    }

    boot.mat.abs <- abs(boot.mat)
    var.t0.abs <- apply(boot.mat.abs, 2, var)
    boot.ci <- cbind(simsl.obj$beta.coef-qnorm((1+boot.conf)/2)*sqrt(var.t0.abs),
                     simsl.obj$beta.coef+qnorm((1+boot.conf)/2)*sqrt(var.t0.abs))

    boot.ci <- cbind(simsl.obj$beta.coef, boot.ci, (boot.ci[,1] > 0 | boot.ci[,2] < 0) )
    colnames(boot.ci) <- c("coef", "LB", "UB", " ***")
    rownames(boot.ci) <- colnames(X)
  }
  simsl.obj$boot.mat <- boot.mat
  simsl.obj$boot.ci <- boot.ci

  return(simsl.obj)
}










#' Single-index models with a surface-link (workhorse function)
#'
#' \code{fit.simsl} is the workhorse function for Single-index models with a surface-link (SIMSL).
#'
#' The function estimates a linear combination (a single-index) of covariates X, and captures a nonlinear interactive structure between the single-index and the treatment defined on a continuum via a smooth surface-link on the index-treatment domain.
#'
#' SIMSL captures the effect of covariates via a single-index and their interaction with the treatment via a 2-dimensional smooth link function.
#' Interaction effects are determined by shapes of the link function.
#' The model allows comparing different individual treatment levels and constructing individual treatment rules,
#' as functions of a biomarker signature (single-index), efficiently utilizing information on patient’s characteristics.
#' The resulting \code{simsl} object can be used to estimate an optimal dose rule for a new patient with pretreatment clinical information.
#'
#' @param y  a n-by-1 vector of treatment outcomes; y is a member of the exponential family; any distribution supported by \code{mgcv::gam}; y can also be an ordinal categorial response with \code{R} categories taking a value from 1 to \code{R}.
#' @param A  a n-by-1 vector of treatment variable; each element is assumed to take a value on a continuum.
#' @param X  a n-by-p matrix of baseline covarates.
#' @param Xm  a n-by-q design matrix associated with an X main effect model; the defult is \code{NULL} and it is taken as a vector of zeros
#' @param family  specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; can be any family supported by \code{mgcv::gam}; can also be "ordinal", for an ordinal categorical response y.
#' @param R   the number of response categories for the case of family = "ordinal".
#' @param bs basis type for the treatment (A) and single-index domains, respectively; the defult is "ps" (p-splines); any basis supported by \code{mgcv::gam} can be used, e.g., "cr" (cubic regression splines); see \code{mgcv::s} for detail.
#' @param k  basis dimension for the treatment (A) and single-index domains, respectively.
#' @param m  a length 2 list (e.g., m=list(c(2,3), c(2,2))), for the treatment (A) and single-index domains, respectively, where each element specifies the order of basis and penalty (note, for bs="ps", c(2,3) means a 2nd order P-spline basis (cubic spline) and a 3rd order difference penalty; the default "NA" sets c(2,2) for each domain); see \code{mgcv::s} for details.
#' @param sp  a vector of smoothing parameters;  Smoothing parameters must be supplied in the order that the smooth terms appear in the model formula (i.e., A, and then the single-index); negative elements indicate that the parameter should be estimated, and hence a mixture of fixed and estimated parameters is possible; see \code{mgcv::gam} for detail.
#' @param knots  a list containing user-specified knot values to be used for basis construction, for the treatment (A) and single-index domains, respectively.
#' @param sep.A.effect   If \code{TRUE}, the g term of SIMSL is further decomposed into: the A main effect + the A-by-X interaction effect; the default is \code{FALSE}.
#' @param mc  a length 2 vector indicating which marginals (i.e., A and the single-index, respectively) should have centering (i.e., the sum-to-zero) constraints applied; the default is \code{mc = c(TRUE, FALSE)} (see \code{mgcv::te} for detail of the constraint), which is sufficient for the so-called "orthogonality" constraint of the SIMSL.
#' @param method  the smoothing parameter estimation method; "GCV.Cp" to use GCV for unknown scale parameter and Mallows' Cp/UBRE/AIC for known scale; any method supported by \code{mgcv::gam} can be used.
#' @param beta.ini  an initial value for \code{beta.coef}; a p-by-1 vector; the defult is \code{NULL}, in which case a linear model estimate is used.
#' @param ind.to.be.positive  for identifiability of the solution \code{beta.coef}, the user can restrict the jth (e.g., j=1) component of \code{beta.coef} to be positive; by default, we match the "overall" sign of \code{beta.coef} with that of the linear estimate (i.e., the initial estimate), by restricting the inner product between the two to be positive.
#' @param random.effect  if \code{TRUE}, as part of the main effects, the user can incorporate z-specific random intercepts.
#' @param z  a factor that specifies the random intercepts when \code{random.effect = TRUE}.
#' @param gamma  increase this beyond 1 to produce smoother models. \code{gamma} multiplies the effective degrees of freedom in the GCV or UBRE/AIC (see \code{mgcv::gam} for detail); the default is 1.
#' @param pen.order 0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of \code{beta.coef}.
#' @param lambda  a regularization parameter associated with the penalized LS for \code{beta.coef} update.
#' @param max.iter  an integer specifying the maximum number of iterations for \code{beta.coef} update.
#' @param eps.iter a value specifying the convergence criterion of algorithm.
#' @param trace.iter if \code{TRUE}, trace the estimation process and print the differences in \code{beta.coef}.
#' @param center.X   if \code{TRUE}, center X to have zero mean.
#' @param scale.X    if \code{TRUE}, scale X to have unit variance.
#' @param uncons.final.fit    if \code{TRUE}, once the convergence in the estimates of \code{beta.coef} is reached, include the main effect associated with the fitted single-index (beta.coef'X) to the final surface-link estimate.
#'
#'
#' @return a list of information of the fitted SIMSL including
#'  \item{beta.coef}{ the estimated single-index coefficients.} \item{g.fit}{a \code{mgcv:gam} object containing information about the estimated 2-dimensional link function as well as the X main effect model.} \item{beta.ini}{the initial value used in the estimation of \code{beta.coef}} \item{beta.path}{solution path of \code{beta.coef} over the iterations} \item{d.beta}{records the change in \code{beta.coef} over the solution path, \code{beta.path}} \item{X.scale}{sd of pretreatment covariates X} \item{X.center}{mean of pretreatment covariates X} \item{A.range}{range of the observed treatment variable A} \item{p}{number of baseline covariates X} \item{n}{number of subjects}
#'
#' @author Park, Petkova, Tarpey, Ogden
#' @import mgcv
#' @seealso \code{pred.simsl},  \code{fit.simsl}
#' @export
#'
fit.simsl  <- function(y, # a n-by-1 vector of treatment outcomes; y is a member of the exponential family; any distribution supported by mgcv::gam.
                       A, # a n-by-1 vector of treatment (i.e., dose) variable (assumed to be on continumm).
                       X, # a n-by-p matrix of covariates.
                       Xm = NULL,  # a n-by-q design matrix assocaited with the X main effect.
                       family = "gaussian",  # specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; can be any family supported by mgcv::gam; can also be "ordinal", for an ordianal categorical response y.
                       R = NULL,    # for the case of family = "ordinal", the number of response catergories.
                       bs = c("ps", "ps"),  # basis type for the treatment (A) and the single-index domains, respectively; the defult is "ps" (p-splines); any basis supported by mgcv::gam can be used, e.g., "cr" (cubic regression splines).
                       k = c(8, 8),  # basis dimension for the treatment (A) and the single-index domains, respectively.
                       m = list(NA, NA),  # a list of length 2 (e.g., m=list(c(2,3), c(2,2))), each element pecifying the order of basis and penality (for bs="ps", c(2,3) specifies a 2nd order P-spline basis (cubic spline) with a 3rd order difference penalty; the default (i.e., "NA") sets c(2,2) to each domain), for the treatment (A) and the single-index domains, respectively; see mgcv::s for details.
                       sp = NULL, # a vector of smoothing parameters;  Smoothing parameters must be supplied in the order that the smooth terms appear in the model formula (i.e., A and then the single-index); negative elements indicate that the parameter should be estimated, and hence a mixture of fixed and estimated parameters is possible; see mgcv::gam for detail.
                       knots = NULL,   # a list containing user-specified knot values to be used for basis construction, for the treatment (A) and the single-index domains, respectively.
                       sep.A.effect = FALSE,  # If TRUE, the g term of SIMSL is further decomposed into: the A main effect term + the A-by-X interaction effect term.
                       mc = c(TRUE, FALSE), # a length 2 vector indicating which marginals (i.e., A and the single-index, respectively) should have centering (i.e., sum-to-zero) constraints applied; the default is mc = c(TRUE, FALSE) (see mgcv::te for detail of the constraint), which is sufficient for the so-called "orthogonality" constraint of the SIMSL.
                       method = "GCV.Cp",  # the smoothing parameter estimation method; can be "GCV.Cp", "REML" and "ML"; see mgcv::gam for detail.
                       beta.ini = NULL,  # an initial value for beta.coef; a p-by-1 vector; the defult is NULL.
                       ind.to.be.positive = NULL,  # for identifiability of the solution beta.coef, we can restrict the jth (e.g., j=1) component of beta.coef to be positive; by default, we match the "overall" sign of the single-index with that of the linear estimate (i.e., the initial estimate), by keeping the inner product between the two to be positive.
                       random.effect = FALSE,  # if TRUE, incorporate the z-specific random intercepts, as part of the main effects.
                       z = NULL, # a factor that specifies random intercepts, if random.effect = TRUE.
                       gamma =1,  # increase this beyond 1 to produce smoother models. gamma multiplies the effective degrees of freedom in the GCV or UBRE/AIC (see mgcv::gam for detail).
                       pen.order = 0,  # pen.order 0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of beta.coef.
                       lambda = 0,     # a regularziation parameter associated with the penalized LS of beta.coef.
                       max.iter = 10,  # an integer specifying the maximum number of iterations for beta.coef update.
                       eps.iter = 0.01, # a value specifying the convergence criterion of algorithm.
                       trace.iter = TRUE, # if TRUE, trace the estimation process and print the differences in beta.coef.
                       center.X=TRUE,  # if TRUE, center X to have zero mean.
                       scale.X=TRUE,   # if TRUE, scale X to have unit variance.
                       uncons.final.fit = TRUE) # if TRUE, once the convergece in the estimates of beta.coef is reached, include the main effect associated with the fitted single-index (beta.coef'X) to the final surface-link estimate.
{


  y <- as.vector(y)
  n <- length(y)
  p <- ncol(X)
  A.range <- range(A, na.rm = TRUE)  # the observed range of A

  ## Center and scale X
  Xc <- scale(X, center = center.X, scale = scale.X)
  X.center <- attr(Xc, "scaled:center")
  X.scale  <- attr(Xc, "scaled:scale")

  ## If not provided by the user, the efficiency augmentation vector (corresponding to the X main effect) is set to be a zero vector.
  if(is.null(Xm)) Xm <- rep(0,n)

  ## Specify a penalty matrix associated with the penalized least squares for estimating beta.coef.
  D <- diag(p);  if(pen.order != 0)  for(j in 1:pen.order) D <- diff(D);
  Pen <- sqrt(lambda)*D

  ## special case of an ordinal categorical response
  if(family=="ordinal"){
    if(is.null(R)) R <- length(unique(y))
    family=ocat(R=R)
  }

  ## initialize the single-index coefficient and the single-index.
  if(is.null(beta.ini)){
    Ac <- A; if(mc[1]) Ac <- A-mean(A)
    if(random.effect){
      tmp <- gam(y~ Xm + s(z, bs="re") + s(A, k=k[1], bs=bs[1], m=m[[1]]) + Ac:Xc, family = family, knots = knots)$coef   # specify the basis type bs and the form of the penalty m
    }else{
      tmp <- gam(y~ Xm + s(A, k=k[1], bs=bs[1], m=m[[1]]) + Ac:Xc, family = family, knots = knots)$coef   # specify the basis type bs and the form of the penalty m
    }
    beta.ini <- tmp[grep("Ac:" ,names(tmp))]
  }
  beta.ini[which(is.na(beta.ini))] <- 0

  beta.coef <- beta.ini/sqrt(sum(beta.ini^2))  # enforce unit L2 norm
  if(!is.null(ind.to.be.positive)){
    if(beta.coef[ind.to.be.positive] < 0) beta.coef <- -1*beta.coef      # for the (sign) identifiability
  }
  single.index <- as.vector(Xc %*% beta.coef)


  ## the following chunk initializes the surface link, through the ti() term.

  if(sep.A.effect){   # if sep.A.effect==TRUE, in ti(), "mc" should be c(T, T), to exclude both A and X main effects from the ti() term.
    mc <- c(TRUE, TRUE)
    if(!is.null(sp)) sp <- c(sp[1], sp)
    if(random.effect){
      g.fit <- gam(y ~ Xm + s(z, bs="re") + s(A, bs=bs[1], k=k[1], m=m[[1]]) + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),
                   knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
    }else{
      g.fit <- gam(y ~ Xm + s(A, bs=bs[1], k=k[1], m=m[[1]]) + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),
                   knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
    }
  }else{  # if sep.A.effect==FALSE, in ti(), "mc" should be c(T, F) (which is the default mode), to allow the A main effect (and exclude only the X main effect) in the ti() term.
    if(random.effect){
      g.fit <- gam(y ~ Xm + s(z, bs="re") + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),  # this models the heterogenous A effect as a function of the single-index
                   knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
    }else{
      g.fit <- gam(y ~ Xm + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),  # this models the heterogenous A effect as a function of the single-index
                   knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
    }
  }


  beta.path <- beta.coef
  d.beta <- NULL

  ## Start iteration
  for (it in 2:max.iter)
  {

    # take the 1st deriavative of the 2D smooths, w.r.t. the single.index.`
    g.der <- der.link(g.fit)

    ## Update beta.coef and intercept through lsfit
    # adjusted responses, adjusted for the nonlinearity associated with the smooth
    y.star    <- residuals(g.fit, type="working") + g.der*single.index
    # adjusetd covariates, adjusted for the nonlinearity of the smooth
    X.tilda   <- diag(g.der) %*% Xc
    nix       <- rep(0, nrow(D))
    X.p       <- rbind(X.tilda, Pen)
    y.p       <- c(y.star, nix)
    # perform a (penalized) WLS
    beta.fit   <- stats::lsfit(X.p, y.p, wt =c(g.fit$weights, (nix + 1)))
    beta.fit$coefficients
    # for the identifiability
    beta.new <- beta.fit$coef[-1]/sqrt(sum(beta.fit$coef[-1]^2))
    if(is.null(ind.to.be.positive)){
      if(beta.ini %*%beta.new  < 0)  beta.new <- -1*beta.new
    }else{
      if(beta.new[ind.to.be.positive] < 0) beta.new <- -1*beta.new      # for the (sign) identifiability
    }
    beta.path <- rbind(beta.path, beta.new)

    ## Check the convergence of beta
    d.beta   <- c(d.beta, sum((beta.new-beta.coef)^2))
    if(trace.iter){
      cat("iter:", it, " "); cat(" difference in beta: ", d.beta[(it-1)], "\n")
    }
    beta.coef <- beta.new
    single.index <- as.vector(Xc %*% beta.coef)

    if (d.beta[(it-1)] < eps.iter)
      break

    # Update the surface-link for the heterogenous A effects, subject to the "orthogonality" constraint

    if(sep.A.effect){   # if sep.A.effect==TRUE, in ti(), "mc" should be c(T, T), to exclude both A and X main effects from the ti() term.
      if(random.effect){
        g.fit <- gam(y ~ Xm + s(z, bs="re") + s(A, bs=bs[1], k=k[1], m=m[[1]]) + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),
                     knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
      }else{
        g.fit <- gam(y ~ Xm + s(A, bs=bs[1], k=k[1], m=m[[1]]) + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),
                     knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
      }
    }else{  # if sep.A.effect==FALSE, in ti(), "mc" should be c(T, F) (which is the default mode), to allow the A main effect (and exclude only the X main effect) in the ti() term.
      if(random.effect){
        g.fit <- gam(y ~ Xm + s(z, bs="re") + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),  # this models the heterogenous A effect as a function of the single-index
                     knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
      }else{
        g.fit <- gam(y ~ Xm + ti(A, single.index, bs=bs, k=k, m=m, mc=mc),  # this models the heterogenous A effect as a function of the single-index
                     knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
      }
    }

  }

  if(uncons.final.fit){
    if(length(sp)>2) sp <- sp[-1]
    g.fit <- gam(y ~ Xm + te(A, single.index, bs=bs, k=k, m=m), knots = knots, family =  family, gamma=gamma, sp=sp, method=method)
  }


  results <- list(beta.coef = beta.coef,
                  beta.ini = beta.ini, d.beta=d.beta, beta.path=beta.path,
                  g.fit= g.fit,
                  beta.fit=beta.fit,
                  X.scale=X.scale, X.center = X.center, y=y, A=A, X=X, Xm = Xm,
                  z=z, random.effect=random.effect,
                  A.range=A.range, p=p, n=n, bs=bs, k=k, m=m, mc=mc, gamma=gamma)
  class(results) <- c("simsl", "list")

  return(results)
}





#' A subfunction used in estimation
#'
#' This function computes the 1st derivative of the surface-link function with respect to the argument associated with the pure interaction effect term of the smooth, using finite difference.
#'
#' @param g.fit  a \code{mgcv::gam} object
#' @param eps a small finite difference used in numerical differentiation.
#' @seealso \code{fit.simsl}, \code{simsl}
#'
der.link <- function(g.fit, #arg.number=3,
                     eps=10^(-4))
{
  m.terms <- attr(stats::terms(g.fit), "term.labels")
  newD <- stats::model.frame(g.fit)[, m.terms, drop = FALSE]
  newD
  newDF <- data.frame(newD)  # needs to be a data frame for predict
  X0 <- predict.gam(g.fit, newDF, type = "lpmatrix")
  newDF[,"single.index"] <- newDF[,"single.index"] + eps
  X1 <- predict.gam(g.fit, newDF, type = "lpmatrix")
  Xp <- (X1 - X0) / eps
  Xi <- Xp * 0
  want <- grep("A,single.index", colnames(X1))  # take only the pure interaction effect term
  Xi[, want] <- Xp[, want]
  g.der  <- as.vector(Xi %*% stats::coef(g.fit))  # the first derivative of the link function
  g.der
  return(g.der)
}



#' SIMSL prediction function
#'
#' This function makes predictions from an estimated SIMSL, given a (new) set of covariates.
#' The function returns a set of predicted outcomes given the treatment values in a dense grid of treatment levels for each individual, and a recommended treatment level (assuming a larger value of the outcome is better).
#'
#' @param simsl.obj  a \code{simsl} object
#' @param newX  a (n-by-p) matrix of new values for the covariates X at which predictions are to be made.
#' @param newA  a (n-by-L) matrix of new values for the treatment A at which predictions are to be made.
#' @param newXm a (n-by-q) matrix of new values for the covariates associated with the fitted main effect Xm at which predictions are to be made.
#' @param single.index  a length n vector specifying new values for the single-index at which predictions are to be made; the default is \code{NULL}.
#' @param L when \code{newA=NULL}, a value specifying the length of the grid of A at which predictions are to be made.
#' @param type the type of prediction required; the default "response" is on the scale of the response variable; the alternative "link" is on the scale of the linear predictors.
#' @param maximize the default is \code{TRUE}, assuming a larger value of the outcome is better; if \code{FALSE}, a smaller value is assumed to be prefered.
#'
#' @return
#' \item{pred.new}{a (n-by-L) matrix of predicted values; each column represents a treatment dose.}
#' \item{trt.rule}{a (n-by-1) vector of suggested treatment assignments}
#'
#'
#' @author Park, Petkova, Tarpey, Ogden
#' @seealso \code{simsl},\code{fit.simsl}
#' @export
#'
pred.simsl  <-  function(simsl.obj, newX=NULL, newA =NULL, newXm =NULL, single.index=NULL, L=50, type = "link", maximize=TRUE)
{
  #if(!inherits(simsl.obj, "simsl"))   # checks input
  #  stop("obj must be of class `simsl'")

  if(is.null(single.index)){

    if(is.null(newX)){
      newX  <- simsl.obj$X
      if(is.null(newXm)) newXm <- simsl.obj$Xm
    }else{
      if(is.null(newXm)){
        if(is.matrix(simsl.obj$Xm)){ newXm <- matrix(0, nrow(newX), ncol(simsl.obj$Xm)) }
        else{ newXm <- rep(0, nrow(newX)) }
      }
    }
    if(ncol(newX) != simsl.obj$p) stop("newX needs to be of p columns ")

    if(is.null(simsl.obj$X.scale)){
      if(is.null(simsl.obj$X.center)){
        newX.scaled <- scale(newX, center = rep(0,simsl.obj$p), scale = rep(1,simsl.obj$p))
      }else{
        newX.scaled <- scale(newX, center = simsl.obj$X.center, scale = rep(1,simsl.obj$p))
      }
    }else{
      newX.scaled <- scale(newX, center = simsl.obj$X.center, scale = simsl.obj$X.scale)
    }
    single.index  <- newX.scaled %*% simsl.obj$beta.coef

  }else{
    if(is.null(newXm)){
      if(is.matrix(simsl.obj$Xm)){ newXm <- matrix(0, length(single.index), ncol(simsl.obj$Xm)) }
      else{ newXm <- rep(0, length(single.index)) }
    }
  }

  # compute treatment (A)-specific predicted outcomes
  if(is.null(newA)){
    A.grid <- seq(from =simsl.obj$A.range[1], to =simsl.obj$A.range[2], length.out =L)
    newA <-  matrix(A.grid, length(single.index), L, byrow = TRUE)
  }else{
    newA <- as.matrix(newA)
    L <- ncol(newA)
  }

  pred.new <- matrix(0, length(single.index), L)
  for(a in 1:L){
    if(simsl.obj$random.effect){
      newD <- list(Xm= newXm, A= newA[,a], single.index=single.index,z=rep(simsl.obj$g.fit$model$z[1], length(single.index)))
    }else{
      newD <- list(Xm= newXm, A= newA[,a], single.index=single.index)
    }
    pred.new[ ,a] <- predict.gam(simsl.obj$g.fit, newD, type =type)
    rm(newD)
  }

  # compute optimal treatment assignment
  if(maximize){
    opt.trt.index <- apply(pred.new, 1, which.max)
  }else{
    opt.trt.index <- apply(pred.new, 1, which.min)
  }

  trt.rule <- rep(NA, nrow(pred.new))
  for(i in 1:nrow(pred.new)){
    trt.rule[i] <- newA[i, opt.trt.index[i]]
  }

  return(list(trt.rule = trt.rule, pred.new = pred.new))
}




######################################################################
## END OF THE FILE
######################################################################
