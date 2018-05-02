# Predicting-age-from-the-transcriptome-of-human-dermal-fibroblasts

This project generates Figure 2 from the paper "Predicting age from the transcriptome of human dermal fibroblasts", in sumbission, 2018.

## Abstract
There is a marked heterogeneity in human lifespan and health outcomes for people of the same chronological age. Thus, one fundamental challenge is to identify molecular and cellular biomarkers of aging that could predict lifespan and be useful in evaluating lifestyle changes and therapeutic strategies in the pursuit of healthy aging. Here, we developed a computational method to predict biological age from gene expression data in skin fibroblast cells using an ensemble of machine learning classifiers. We generated an extensive RNA-seq dataset of fibroblast cell lines derived from 133 healthy individuals whose ages range from 1 to 94 years, and 10 patients with Hutchinson-Gilford Progeria Syndrome (HGPS), a premature aging disease. On this dataset, our method predicted chronological age with a median error of 4 years, outperforming algorithms proposed by prior studies that predicted age from DNA methylation [1–5] and gene expression data [3,6] for fibroblasts. Importantly, our method consistently predicted higher ages for Progeria patients compared to age-matched controls, suggesting that our algorithm can identify accelerated aging in humans. These results show that the transcriptome of skin fibroblasts retains important age-related signatures. Our computational tool may also be applicable to predicting age from other genome-wide datasets.

1.    Horvath, S. (2013). DNA methylation age of human tissues and cell types. Genome Biology 14, 3156.
2.    Hannum, G., Guinney, J., Zhao, L., Zhang, L., Hughes, G., Sadda, S., Klotzle, B., Bibikova, M., Fan, J.-B., Gao, Y., et al. (2013). Genome-wide Methylation Profiles Reveal Quantitative Views of Human Aging Rates. Molecular Cell 49, 359–367.
3.    Peters, M.J., Joehanes, R., Pilling, L.C., Schurmann, C., Conneely, K.N., Powell, J., Reinmaa, E., Sutphin, G.L., Zhernakova, A., Schramm, K., et al. (2015). The transcriptional landscape of age in human peripheral blood. Nature Communications 6, 8570.
4.    Xu, C., Qu, H., Wang, G., Xie, B., Shi, Y., Yang, Y., Zhao, Z., Hu, L., Fang, X., Yan, J., et al. (2015). A novel strategy for forensic age prediction by DNA methylation and support vector regression model. Scientific Reports 5, 17788.
5.    Zhang, Y., Wilson, R., Heiss, J., Breitling, L.P., Saum, K.-U., Schöttker, B., Holleczek, B., Waldenberger, M., Peters, A., and Brenner, H. (2017). DNA methylation signatures in peripheral blood strongly predict all-cause mortality. Nature Communications 8, 14617.
6.    Holly Alice C., Melzer David, Pilling Luke C., Henley William, Hernandez Dena G., Singleton Andrew B., Bandinelli Stefania, Guralnik Jack M., Ferrucci Luigi, and Harries Lorna W. (2013). Towards a gene expression biomarker set for human biological age. Aging Cell 12, 324–326.


## Getting started

### Prerequisites

While it may work with different versions of these packages, the software was tested in a GNU Linux environment using the following packages:

* Jupyter notebook 5.0.0
* Python 2.7.9
* Pandas 0.22
* Scipy 0.19
* Matplotlib 2.1.0
* Seaborn 0.7.1


### Using this software

Open the notebook "Generate panels for figure.ipynb" in Jupyter.

To regenerate figures from the saved runs, simply select "Cell -> Run All" from the pull down menus.

To run the analysis again from scratch using the fpkm tables and metadata distributed in this repository, go to the cell where the function make_figs() is declared.  In the next 4 cells after that declaration are the calls to make_figs() that generate the data.  Inside those calls are three lines that are commented out, that need to be uncommented before selecting "Run All" from teh pull down menus.  The lines like:

```
svregr = make_figs( 'Support vector regression', model=subsvr,
                  #search_cval=search_cval, parameters=parameters, # uncomment these lines
                  #plot_cval=LeaveOneOut(),                        # to rerun the analysis from scratch
                  #lcurve_cval=lcurve_cval,                        # instead of loading results from disk
                  njobs=njobs)   
```

Be aware that running the LDA ensemble from scratch will take many days of compute time, as documented in code comments.

### Code structure

* A custom function handles the loading of the FPKM and metadata.
* Sublasses of scikit-learn functions are implemented that do gene subsetting during the .fit() call for all the regression algorithms.
* A class is implemented that handles the staggered age-bin ensenble.
* Thereafter, standard scikit-learn libraries are used to do parameter search and cross-validation.
* make_figs() function handles the creation of the models, stores the model & predictions for later so that figures can be re-generated/modified from previous runs, and creates the figures themselves.   


## Author

[**Jason G. Fleischer**](https://github.com/jasongfleischer)

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

As always, we stand on the shoulders of giants.  This project would not have been possible without the  many developers of scikit-learn, scipy, numpy, pandas, seaborn, matplotlib, jupyter, and python.

