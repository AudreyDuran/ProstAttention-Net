# ProstAttention-Net: A deep attention model for prostate cancer segmentation by aggressiveness in MRI scans

See the associated paper published in Medical Image Analysis :
https://www.sciencedirect.com/science/article/pii/S1361841521003923

![image from web](https://ars.els-cdn.com/content/image/1-s2.0-S1361841521003923-ga1_lrg.jpg)

## How to cite

If you use this repository, please cite the associated publication :

```
@article{duran_prostattention-net_2022,
	title = {{ProstAttention}-{Net}: {A} deep attention model for prostate cancer segmentation by aggressiveness in {MRI} scans},
	journal = {Medical Image Analysis},
	pages = {102347},
	month = apr,
	year = {2022},
	volume = {77},
	issn = {1361-8415},
	url = {https://www.sciencedirect.com/science/article/pii/S1361841521003923},
	doi = {10.1016/j.media.2021.102347},
	abstract = {Multiparametric magnetic resonance imaging (mp-MRI) has shown excellent results in the detection of prostate cancer (PCa). However, characterizing prostate lesions aggressiveness in mp-MRI sequences is impossible in clinical practice, and biopsy remains the reference to determine the Gleason score (GS). In this work, we propose a novel end-to-end multi-class network that jointly segments the prostate gland and cancer lesions with GS group grading. After encoding the information on a latent space, the network is separated in two branches: 1) the first branch performs prostate segmentation 2) the second branch uses this zonal prior as an attention gate for the detection and grading of prostate lesions. The model was trained and validated with a 5-fold cross-validation on a heterogeneous series of 219 MRI exams acquired on three different scanners prior prostatectomy. In the free-response receiver operating characteristics (FROC) analysis for clinically significant lesions (defined as GS~{\textgreater}6) detection, our model achieves 69.0\%{\textpm}14.5\% sensitivity at 2.9 false positive per patient on the whole prostate and ~70.8\%{\textpm}14.4\% sensitivity at 1.5 false positive when considering the peripheral zone (PZ) only. Regarding the automatic GS group grading, Cohen{\textquoteright}s quadratic weighted kappa coefficient ($\kappa$) is 0.418{\textpm}0.138, which is the best reported lesion-wise kappa for GS segmentation to our knowledge. The model has encouraging generalization capacities with $\kappa$=0.120{\textpm}0.092 on the PROSTATEx-2 public dataset and achieves state-of-the-art performance for the segmentation of the whole prostate gland with a Dice of 0.875{\textpm}0.013. Finally, we show that ProstAttention-Net improves performance in comparison to reference segmentation models, including U-Net, DeepLabv3+ and E-Net. The proposed attention mechanism is also shown to outperform Attention U-Net.},
	author = {Duran, Audrey and Dussert, Gaspard and Rouvi{\`e}re, Olivier and Jaouen, Tristan and Jodoin, Pierre-Marc and Lartizien, Carole},
	keywords = {Deep learning, Prostate cancer, Magnetic resonance imaging, Semantic segmentation, Computer-aided detection, Attention models},
}
```

## Requirements and installation

Open a terminal and do :

```bash
cd PATH_TO_YOUR_DEVELOPMENT_FOLDER
git clone https://github.com/AudreyDuran/ProstAttention-Net.git
cd ProstAttention-Net
```

Create a conda environment :

```bash
conda env create -f requirements/prostattention.yml
```

Unless you manually edit the first line (name: prostattention) of the file, the environment will be
named `prostattention` by default.

Activate the environment :

```bash
conda activate prostattention
```

Then, install the dependencies and setup the repo with :

```bash
pip install -e .
```

### Windows users

`antspyx` installation via `pip` is not supported yet on Windows. In case you have a Windows machine, you should
remove `antspyx`from `requirements/prostattention.yml` file and use :

```bash
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```

## How to use

### Create a .hdf5 file dataset

First create a .hdf5 file dataset from your data. The original dataset should be composed of .nifti mri volumes and
ground truth, and organized as follow, with one folder for each patient :

```
├── DatasetFolder
│   ├── patient_1
│   │   ├── T2-pat1.nii.gz
│   │   ├── ADC-pat1.nii.gz
│   │   ├── GT-T2-pat1.nii.gz
│   ├── patient_2
│   ├── patient_3
```

The nifti volumes might be named differently but should begin with "T2", "ADC" and "GT" (otherwise modify the source
code). The ground truth nifti volume is expected to contain the following labels :

- 0 : background
- 1 : healthy prostate
- 2 : GS 6
- 3 : GS 3+4
- 4 : GS 4+3
- 5 : GS ≥ 8

#### Cross-validation experiment

In that case, a 2-columns comma-separated values (csv) containing the patient split is expected, with the following
formatting :

```
patient,subfold
patient_1,subfold_0
patient_2,subfold_1
patient_3,subfold_2
patient_4,subfold_0
patient_5,subfold_1
patient_6,subfold_2
```

The patient identifiers should match the patient's folder names. In this table, it is a 3-fold cross-validation, but
note that any number of folds is supported.

Then generate the hdf5 file :

```bash
python ProstAttention/dataset/generate_hdf5.py <path/to/DatasetFolder/> <name/of/output.hdf5> 
--subfold subfold_split.csv
```

#### Train / validation / test experiment

For a classic train / validation / test experiment, the patient split is random, according to the distributions given  
in argument :

```bash
python ProstAttention/dataset/generate_hdf5.py <path/to/DatasetFolder/> <name/of/output.hdf5> 
--set_split 0.6 0.2 0.2
```

Use the `--help` optional argument for detailed parameters explanation.

#### Script description

The script does the following tasks:

- load the T2, ADC and GT data for each patient in the database
- resample images to a the given pixel size (default 1x1mm²)
- align ADC on T2 volumes
- split the patients data into train, validation and test sets given set split or split data in the adequate folds in
  case of a cross-validation dataset (given the specified `--subfold` file)
- save in a hdf5 file

The generated hdf5 file has the following structure :

```
├── <set> (group)
│   ├── patient_1 (group)
│   │   ├── img (dataset)
│   │   ├── gt (dataset)
│   │   ├── gt_prostate (dataset)
│   ├── patient_2
│   ├── patient_3
```

With :

- `<set>` : either 'train', 'val', 'test' or 'subfold_0', 'subfold_1' ... 'subfold_n' in case of a cross-validation
  experiment
- `img` : the input image, of shape (num_slices, height, width, num_channels)
- `gt` : the ground truth, of shape (num_slices, height, width, num_classes)
- `gt_prostate` : the ground truth for the prostate zone, of shape (num_slices, height, width, 2)

### Train the model

Once your dataset is created, you can train ProstAttention-Net model.

#### Cross-validation experiment

```bash
python ProstAttention/prostattentionnet.py dataset_crossval.hdf5 --subfold 0
```

#### Train / validation / test experiment

```bash
python ProstAttention/prostattentionnet.py dataset_trainvaltest.hdf5
```

Use the `--help` optional argument for detailed parameters explanation. Default values were the one used in the paper.

A folder is created for each experiment (named `ExperimentFolder` here), that contains :

```
├── ExperimentFolder
│   ├── run_conf.json 
│   ├── csvlog.csv 
│   ├── RESULTS
│   │   ├── ClassificationReports 
│   │   ├── ConfusionMatrices 
│   │   ├── NIFTI 
│   │   │   ├── train
│   │   │   │   ├── patient_1
│   │   │   │   │   ├── patient_1_image-ch0.nii.gz
│   │   │   │   │   ├── patient_1_image-ch1.nii.gz
│   │   │   │   │   ├── patient_1_groundtruth_prostate.nii.gz
│   │   │   │   │   ├── patient_1_groundtruth.nii.gz
│   │   │   │   │   ├── patient_1_prediction_prostate.nii.gz
│   │   │   │   │   ├── patient_1_prediction.nii.gz
│   │   │   │   ├── patient_2
│   │   │   ├── validation
│   │   │   ├── test (if exists)
│   │   ├── TRAIN_PREDICTION 
│   │   ├── VALID_PREDICTION
│   │   ├── TEST_PREDICTION (if exists)
│   ├── ModelCheckpoint
│   ├── MODEL
│   ├── LOGS
```

With :

- run_conf.json : a json file containing the experiment parameters
- csvlog.csv : a csv file showing the values of the monitored metrics for each epoch
- ClassificationReports folder : contains csv files as computed by sklearn.metrics.classification_report, for each
  output and set
- ConfusionMatrices folder : contains png confusion matrices as returned by sklearn.metrics.confusion_matrix, for each
  output and set
- NIFTI folder : contains a subfolder for each set, and sub-subfolders for each patient. The models input images, ground
  truth and prediction for the prostate and multiclass segmentation tasks
- SET_PREDICTION : hdf5 file containing the predictions for each patient of the given set. Nifti files are generated
  from these prediction files.
- ModelCheckpoint : contains hdf5 files with the weights of the models that had achieved the "best performance" so far
- MODEL : contains a hdf5 file with the final best model weights
- LOGS : training LOGS, used for TensorBoard monitoring (command : `tensorboard --logdir ExperimentFolder/LOGS/`)

## Acknowledgments

Thanks to Pierre-Marc Jodoin, head of VITAL Lab (Sherbrooke, QC), for providing an open-access to
its [VITALab library](https://bitbucket.org/vitalab/vitalabai_public/src/master/). Many thanks to Nathan Painchaud and
Carl Lemaire for their code reviews.

This work was supported by the RHU PERFUSE (ANR-17-RHUS-0006) of Université Claude Bernard Lyon 1 (UCBL), within the
program “Investissements d’Avenir” operated by the French National Research Agency (ANR). It was performed within the
framework of the Discovery grant RGPIN-2018-05401 awarded to P-M Jodoin by the Natural Sciences and Engineering Research
Council of Canada (CA) and the LABEX PRIMES (ANR-11-LABX-0063) of Université de Lyon operated by the French National
Research Agency (ANR).