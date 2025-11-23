# IsoFlow-Orbitrap (DOI: 10.5281/zenodo.17687577)

### An automated Python-based data extraction and processing workflow for Isotopic Ratio Analysis using Flow Injection ESI-Orbitrap 

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**IsoFlow-Orbitrap** is a Python-based data processing workflow designed to extract, process, and visualize high-precision stable isotope ratios (e.g., $\delta^{15}\text{N}$, $\delta^{18}\text{O}$) from Thermo Fisher Orbitrap `.raw` files. 

Designed specifically for **Flow Injection Analysis (FIA)**, this tool automates the extraction of transient ion signals, applies standard bracketing for drift correction, and performs rigorous statistical analysis to verify if measurements are limited by counting statistics (shot noise).

---

## Key Features

* **Batch Processing:** Multithreaded extraction of `.raw` files using `MSFileReader` libraries.
* **Drift Correction:** Implements a **Reference-Sample-Reference** bracketing method to correct for instrumental mass bias and drift over time.
* **Dual Analysis Modes:**
    * **`VALIDATION` Mode:** For method development. Compares calculated $\delta$ values against known certified values to report accuracy (Error %) and precision.
    * **`UNKNOWN` Mode:** For routine analysis. Calculates $\delta$ values for samples without known certified values.
* **Shot Noise Diagnostics:** Automatically generates cumulative log-log plots comparing experimental Standard Error (SE) against the theoretical Poisson shot noise limit ($1/\sqrt{N}$).
* **Configurable Targets:** While pre-configured for Nitrate ($\text{NO}_3^-$), the script is fully modular and can be adapted for Sulfate, Phosphate, or organic molecules by simply updating the mass list.

---

## Prerequisites & Installation

Since this tool relies on Thermo's proprietary libraries to read `.raw` files, it is designed to run on **Windows**.

### 1. Install MSFileReader 
This script requires the **Thermo MSFileReader** COM libraries. 
* You can find the installers and detailed instructions in the `pymsfilereader` repository:
    * [Download MSFileReader via pymsfilereader GitHub](https://github.com/frallain/pymsfilereader/tree/master)
    * *Note: Ensure you install the version matching your Python architecture (usually 64-bit).*

### 2. Clone the Repository
```bash
git clone [https://github.com/isoeremita23/IsoFlow-Orbitrap.git](https://github.com/isoeremita23/IsoFlow-Orbitrap.git)
cd IsoFlow-Orbitrap
