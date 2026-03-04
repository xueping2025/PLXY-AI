Semantic Segmentation Workflow
1. Input
    Users should place the images to be segmented into the Input folder. This folder serves as the unified input directory for the entire workflow. All subsequent batch processing will be performed based on the images contained in this folder.
2. Notebook Execution (1–8)
    Users can selectively run one or more of the eight .ipynb notebooks according to their specific research requirements.
2.1 Environment Setup
    This notebook is used to configure the runtime environment. The recommended environment for this project is:Python 3.7、PyTorch 1.10
    Please execute this notebook before running any other notebooks to ensure that all required dependencies are properly installed.
2.2 Task-Specific Batch Prediction Notebooks
    The following notebooks correspond to different anatomical structures and section types for semantic segmentation and area quantification. Detailed operation instructions can be found in the internal comments of each notebook.
    2. Batch prediction of trans-xylem.ipynb: Used for xylem segmentation and area quantification in transverse sections, mainly targeting Populus species.
    3. Batch prediction of trans-vessel.ipynb: Used for vessel segmentation and area quantification in transverse sections.
    4. Batch prediction of trans-fiber.ipynb: Used for fiber segmentation and area quantification in transverse sections.
    5. Batch prediction of trans-ray.ipynb: Used for ray segmentation and area quantification in transverse sections.
    6. Batch prediction of tan-ray.ipynb: Used for ray segmentation and area quantification in tangential sections.
    7. Batch prediction of tan-ray-type.ipynb: Used for segmentation and area quantification of ray types in tangential sections.
    8. Batch prediction of trans-Xylem_Angiosperms.ipynb: Used for xylem segmentation and area quantification in transverse sections for non-Populus angiosperm species.
If the automatic segmentation performance is unsatisfactory, manual xylem segmentation using PLXY AI is recommended.
3. Output
    After running the selected notebooks, the segmentation results and area quantification outputs will be automatically saved to the Output folder. Users can download and archive the results as needed.
Important Notes
    Since output files are saved based on image file names, files with identical names may cause result overwriting. Therefore, it is strongly recommended to:
1.Run only one specific task at a time;
2.Download and save the results from the Output folder after each task;
3.Then proceed to the next task to avoid accidental overwriting.