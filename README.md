# csxrTungstenAnalysis
Scripts used to characterize/regress unknown tungsten line at ~.39648 nm 

In depth description of the work can be found in the following pdfs:
https://github.com/icfaust/csxrTungstenAnalysis/raw/master/presentations/initialAnalysisIFaust.pdf
https://github.com/icfaust/csxrTungstenAnalysis/raw/master/presentations/IFaust_slides.pdf

<p align="center">
  <img src="https://github.com/icfaust/csxrTungstenAnalysis/blob/master/presentations/output_forward.png" alt="not loaded" width="50%"/>
</p>
This repository is the collection of scripts I used to analyze an unknown line of tungsten in the spectrum of the Compact X-ray Spectrometer at the ASDEX-Upgrade tokamak (discussed in detail in this thesis https://edoc.ub.uni-muenchen.de/12056/1/Sertoli_Marco.pdf).  In plasmas, intensities of the observed spectral lines can be used to understand the underlying density or concentration.  However, this requires knowing certain characteristics, namely: what is the line emission emissivity versus the electron temperature. This comes down for soft x-rays to knowing the 'charge state', or how many electrons is missing from the atom.  The spectrum changes significantly as the number of possible states an electron can take changes.  This work was to characterize this soft x-ray line's charge state, which was expected to be 46+.

This work definitively showed that the line was 45+, the method relied on generating and analyzing large datasets from 4 years of ASDEX-Upgrade discharges (500k usable samples). This data was analyzed using standard machine-learning methods and techniques, which are detailed in depth in the presentations available in the 'presentations' folder. Both a forward/simplified Support Vector Machine model, and a non-linear 0th and 2nd order Tikhonov regularized/ cross-validated solutions yielded that the line was 45+, which can be seen in the three images below.

<p align="center">
<img src="https://github.com/icfaust/csxrTungstenAnalysis/blob/master/presentations/val7_inverse.png" alt="not loaded" width="50%"/>
</p>

Several SQL databases were generated for the final machine learning solution, which includes a maximum-likelihood non-linear spectral fitting code written partially in C using the C/Python/Numpy API to speed the initial generation of the spectral datasets.  Secondary datasets were generated using TRIPPy and other plasma data, and can be found in the genSQL folder. These were instrumental, as it saved significant IO to the underlying dataservers at the IPP.

As a consquence, the datasets are of significant size that I did not want to upload them to github publicly (and I am not sure of the legal ramifications).  Similarly, this contains the use of several other packages which are not included, namely the python 'dd', 'eqtools', and 'TRIPPy' packages, most of which can be found in my other repositories.

<p align="center">
<img src="https://github.com/icfaust/csxrTungstenAnalysis/blob/master/presentations/output_SVM.png" alt="not loaded" width="50%"/>
</p>
