% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simsl.main.R
\name{fit.simsl}
\alias{fit.simsl}
\title{Single-index models with a surface-link (workhorse function)}
\usage{
fit.simsl(y, A, X, Xm = NULL, family = "gaussian", R = NULL,
  bs = c("ps", "ps"), k = c(8, 8), m = list(NA, NA), sp = NULL,
  knots = NULL, sep.A.effect = FALSE, mc = c(TRUE, FALSE),
  method = "GCV.Cp", beta.ini = NULL, ind.to.be.positive = NULL,
  random.effect = FALSE, z = NULL, gamma = 1, pen.order = 0,
  lambda = 0, max.iter = 10, eps.iter = 0.01, trace.iter = TRUE,
  center.X = TRUE, scale.X = TRUE, uncons.final.fit = TRUE)
}
\arguments{
\item{y}{a n-by-1 vector of treatment outcomes; y is a member of the exponential family; any distribution supported by \code{mgcv::gam}; y can also be an ordinal categorial response with \code{R} categories taking a value from 1 to \code{R}.}

\item{A}{a n-by-1 vector of treatment variable; each element is assumed to take a value on a continuum.}

\item{X}{a n-by-p matrix of baseline covarates.}

\item{Xm}{a n-by-q design matrix associated with an X main effect model; the defult is \code{NULL} and it is taken as a vector of zeros}

\item{family}{specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; can be any family supported by \code{mgcv::gam}; can also be "ordinal", for an ordinal categorical response y.}

\item{R}{the number of response categories for the case of family = "ordinal".}

\item{bs}{basis type for the treatment (A) and single-index domains, respectively; the defult is "ps" (p-splines); any basis supported by \code{mgcv::gam} can be used, e.g., "cr" (cubic regression splines); see \code{mgcv::s} for detail.}

\item{k}{basis dimension for the treatment (A) and single-index domains, respectively.}

\item{m}{a length 2 list (e.g., m=list(c(2,3), c(2,2))), for the treatment (A) and single-index domains, respectively, where each element specifies the order of basis and penalty (note, for bs="ps", c(2,3) means a 2nd order P-spline basis (cubic spline) and a 3rd order difference penalty; the default "NA" sets c(2,2) for each domain); see \code{mgcv::s} for details.}

\item{sp}{a vector of smoothing parameters;  Smoothing parameters must be supplied in the order that the smooth terms appear in the model formula (i.e., A, and then the single-index); negative elements indicate that the parameter should be estimated, and hence a mixture of fixed and estimated parameters is possible; see \code{mgcv::gam} for detail.}

\item{knots}{a list containing user-specified knot values to be used for basis construction, for the treatment (A) and single-index domains, respectively.}

\item{sep.A.effect}{If \code{TRUE}, the g term of SIMSL is further decomposed into: the A main effect + the A-by-X interaction effect; the default is \code{FALSE}.}

\item{mc}{a length 2 vector indicating which marginals (i.e., A and the single-index, respectively) should have centering (i.e., the sum-to-zero) constraints applied; the default is \code{mc = c(TRUE, FALSE)} (see \code{mgcv::te} for detail of the constraint), which is sufficient for the so-called "orthogonality" constraint of the SIMSL.}

\item{method}{the smoothing parameter estimation method; "GCV.Cp" to use GCV for unknown scale parameter and Mallows' Cp/UBRE/AIC for known scale; any method supported by \code{mgcv::gam} can be used.}

\item{beta.ini}{an initial value for \code{beta.coef}; a p-by-1 vector; the defult is \code{NULL}, in which case a linear model estimate is used.}

\item{ind.to.be.positive}{for identifiability of the solution \code{beta.coef}, the user can restrict the jth (e.g., j=1) component of \code{beta.coef} to be positive; by default, we match the "overall" sign of \code{beta.coef} with that of the linear estimate (i.e., the initial estimate), by restricting the inner product between the two to be positive.}

\item{random.effect}{if \code{TRUE}, as part of the main effects, the user can incorporate z-specific random intercepts.}

\item{z}{a factor that specifies the random intercepts when \code{random.effect = TRUE}.}

\item{gamma}{increase this beyond 1 to produce smoother models. \code{gamma} multiplies the effective degrees of freedom in the GCV or UBRE/AIC (see \code{mgcv::gam} for detail); the default is 1.}

\item{pen.order}{0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of \code{beta.coef}.}

\item{lambda}{a regularization parameter associated with the penalized LS for \code{beta.coef} update.}

\item{max.iter}{an integer specifying the maximum number of iterations for \code{beta.coef} update.}

\item{eps.iter}{a value specifying the convergence criterion of algorithm.}

\item{trace.iter}{if \code{TRUE}, trace the estimation process and print the differences in \code{beta.coef}.}

\item{center.X}{if \code{TRUE}, center X to have zero mean.}

\item{scale.X}{if \code{TRUE}, scale X to have unit variance.}

\item{uncons.final.fit}{if \code{TRUE}, once the convergence in the estimates of \code{beta.coef} is reached, include the main effect associated with the fitted single-index (beta.coef'X) to the final surface-link estimate.}
}
\value{
a list of information of the fitted SIMSL including
 \item{beta.coef}{ the estimated single-index coefficients.} \item{g.fit}{a \code{mgcv:gam} object containing information about the estimated 2-dimensional link function as well as the X main effect model.} \item{beta.ini}{the initial value used in the estimation of \code{beta.coef}} \item{beta.path}{solution path of \code{beta.coef} over the iterations} \item{d.beta}{records the change in \code{beta.coef} over the solution path, \code{beta.path}} \item{X.scale}{sd of pretreatment covariates X} \item{X.center}{mean of pretreatment covariates X} \item{A.range}{range of the observed treatment variable A} \item{p}{number of baseline covariates X} \item{n}{number of subjects}
}
\description{
\code{fit.simsl} is the workhorse function for Single-index models with a surface-link (SIMSL).
}
\details{
The function estimates a linear combination (a single-index) of covariates X, and captures a nonlinear interactive structure between the single-index and the treatment defined on a continuum via a smooth surface-link on the index-treatment domain.

SIMSL captures the effect of covariates via a single-index and their interaction with the treatment via a 2-dimensional smooth link function.
Interaction effects are determined by shapes of the link function.
The model allows comparing different individual treatment levels and constructing individual treatment rules,
as functions of a biomarker signature (single-index), efficiently utilizing information on patient’s characteristics.
The resulting \code{simsl} object can be used to estimate an optimal dose rule for a new patient with pretreatment clinical information.
}
\seealso{
\code{pred.simsl},  \code{fit.simsl}
}
\author{
Park, Petkova, Tarpey, Ogden
}
