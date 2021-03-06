% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/chicago.R
\docType{data}
\name{chicago}
\alias{chicago}
\title{Air pollution dataset}
\format{A data frame with 7 columns and 5114 rows; each row refers to one day; the columns correspond to:
\describe{
\item{death}{ total deaths (per day).}
\item{pm10median}{ median particles in 2.5-10 per cubic m}
\item{pm25median}{ median particles < 2.5 mg per cubic m (more dangerous).}
\item{o3median}{ Ozone in parts per billion}
\item{so2median}{ Median Sulpher dioxide measurement}
\item{time}{ time in days}
\item{tmpd}{ temperature in fahrenheit}
}}
\source{
The \code{chicago} dataset is available from package \code{gamair} (Wood, 2019).
}
\description{
Daily air pollution and death rate data for Chicago
}
\details{
The data are from Peng and Welty (2004) and are available from R (R Core Team, 2019) package \code{gamair} (Wood, 2019).

The daily death in the city of Chicago is recorded over a number of years (about 14 years). Each observation is a time series of daily mortality counts, indicating the number of deaths that occurred on each day.
}
\references{
Peng, R.D. and Welty, L.J. (2004) The NMMAPSdata package. R News 4(2)

Wood, S.N. (2017) Generalized Additive Models: An Introduction with R

Wood, S.N. (2019) gamair: Data for ’GAMs: An introduction with R’. R package version 1.0.2
}
\keyword{dataset}
