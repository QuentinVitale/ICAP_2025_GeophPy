# GeophPy Workshop - Instructor's Guide

## 1. High-Level Plan

* **Topic:** Introduction to `geophpy` for geophysical data processing.
* **Duration:** 120 minutes.
* **Target Audience:** Students or professionals with a basic knowledge of Python and Jupyter Notebooks.
* **Main Asset:** `notebooks/01_Basic_Workflow.ipynb`.
* **Required Data:** `data/Mag_ex1.dat`, `data/GPS_ex1.csv`.

---
## 2. Detailed Agenda & Instructor Notes

### **Part 1: Introduction & Setup (Total: 15 min)**

**[0-5 min] Welcome & Goals**
* Welcome participants.
* Introduce yourself and the `geophpy` package.
* **Key Message:** "Today, we're going on a journey from a raw, messy data file to a clean, georeferenced map that's ready for a GIS."
* Briefly show the agenda from the first cell of the notebook.

**[5-15 min] Environment Check**
* **Action:** Ask everyone to open `01_Basic_Workflow.ipynb` and run the first code cell (the imports).
* **Instructor Note:** This is the most critical step to ensure a smooth workshop. Walk around the room or check the chat for issues. The first cell should print the `geophpy` version. If it fails for anyone, they have an installation problem. Be ready to assist.

---
### **Part 2: Core Processing Workflow (Total: 80 min)**

**[15-35 min] Module A: Loading & Visualizing Raw Data**
* **Action:** Guide them through the data loading cells.
* **Instructor Note:** Explain each parameter of `DataSet.from_file()`.
* **Action:** Run the plotting cell.
* **Key Question for Audience:** "Look at this first map. What problems can you see? Notice the vertical stripes and the slight misalignment between lines. These are the issues we're going to fix."

**[35-80 min] Module B: Applying Correction Filters**
* **Instructor Note:** This is the core of the workshop. Explain the *geophysical meaning* of each filter before showing the code. Show the "before" and "after" plot for each step to highlight the improvement.

* **(Despike - `peakfilt`)**
    * **Concept:** "First, let's remove extreme outliers or 'spikes' that don't represent the geology."
    * **Action:** Run the `peakfilt` cell.

* **(Destaggering - `festoonfilt`)**
    * **Concept:** "This filter corrects for small shifts in the starting position of each survey line, which creates a 'festoon' or 'stagger' effect."
    * **Action:** Run the `festoonfilt` cell. Show the improved alignment.

* **(Destriping - `destripecon`)**
    * **Concept:** "Vertical stripes are often caused by slight differences between sensors in a multi-sensor array. This filter harmonizes the lines."
    * **Action:** Run the `destripecon` cell. The stripes should disappear.

* **(Contrast Enhancement - `wallisfilt`)**
    * **Concept:** "Finally, a Wallis filter enhances local contrast, making subtle geological features much more visible."
    * **Action:** Run the `wallisfilt` cell and show the final, clean map.

---
### **Part 3: Georeferencing & Export (Total: 20 min)**

**[80-90 min] Module C: Georeferencing**
* **Concept:** "Our map looks great, but it lives in a local grid system (like 'Step 1, Profile 1'). To make it useful, we need to place it in the real world using real coordinates."
* **Instructor Note:** Briefly explain what a GCP (Ground Control Point) file is.
* **Action:** Guide them through loading `GPS_ex1.csv` and applying the transformation with `setgeoref()`.

**[90-100 min] Module D: Exporting**
* **Concept:** "The final step is to export our work into standard formats that other software can use."
* **Action:** Run the `to_kml()` and `to_raster()` cells.
* **Instructor Note:** Show them the `dataset_final.kml` and `dataset_final.tiff` files that have been created in their directory.

---
### **Part 4: Bonus & Wrap-up (Total: 5 min + Flex Time)**

**[100-110 min] Bonus: QGIS Integration (If time allows)**
* **Instructor Note:** This is your buffer. If you are running short on time, skip this and just mention that the GeoTIFF can be opened in any GIS software.
* **Action (Live Demo):** Have QGIS open. Drag and drop the `dataset_final.tiff` file into QGIS. Show them the georeferenced map overlaid on a base map (like OpenStreetMap). This is a very powerful final visual.

**[Until 120 min] Q&A and Next Steps**
* Open the floor for questions.
* **Key Message:** "You now have the fundamental workflow for processing data with `geophpy`. For more advanced topics, please check out the official documentation and the other example notebooks in the repository."