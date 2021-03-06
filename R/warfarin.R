#' @title Warfarin dataset
#'
#' @description The dataset provided by International Warfarin Pharmacogenetics Consortium et al. (2009). Warfarin is an anticoagulant agent widely used as a medicine to treat blood clots and prevent forming new harmful blood clots.
#'
#' @details
#' The dataset onsists of 1780 subjects (after removing patients with missing data and data cleaning), including information on patient covariates (X), final therapeutic warfarin dosages (A), and patient outcomes (INR, International Normalized Ratio).
#'
#' There are 13 covariates in the dataset: weight (X1), height (X2), age (X3), use of the cytochrome P450 enzyme inducers (X4; the enzyme inducers considered in this analysis includes phenytoin, carbamazepine, and rifampin), use of amiodarone (X5), gender (X6; 1 for male, 0 for female), African or black race (X7), Asian race (X8), the VKORC1 A/G genotype (X9), the VKORC1 A/A genotype (X10), the CYP2C9 1/2 genotype (X11), the CYP2C9 1/3 genotype (X12), and the other CYP2C9 genotypes (except the CYP2C9 1/1 genotype which is taken as the baseline genotype) (X13).
#'
#' The details of these covariate information are given in International Warfarin Pharmacogenetics Consortium et al. (2009).
#' @docType data
#' @keywords dataset
#' @name warfarin
#' @format A list containing \code{INR}, \code{A}, \code{X}:
#' \describe{
#' \item{INR}{a vector of treatment outcomes of the study (INR; International Normalized Ratio)}
#' \item{A}{a vector of therapeutic warfarin dosages}
#' \item{X}{a data frame consist of 13 patient characteristics}
#' }
#'
#' @source The data can be downloaded from https://www.pharmgkb.org/downloads/.
#'
#' @references
#' International Warfarin Pharmacogenetics Consortium, Klein, T., Altman, R., Eriksson, N., Gage, B., Kimmel, S., Lee, M., Limdi, N., Page, D., Roden, D., Wagner, M., Caldwell, M., and Johnson, J. (2009). Estimation of the warfarin dose with clinical and pharmacogenetic data. The New England Journal of Medicine 360:753–674
#'
#' Chen, G., Zeng, D., and Kosorok, M. R. (2016). Personalized dose finding using outcome wieghted learning. Journal of the American Medical Association 111:1509–1547.
NULL
