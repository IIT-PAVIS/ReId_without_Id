# Person Re-Identification without Identification via Event Anonymization (ICCV 2023)
This repository contains the implementation code of the ICCV23 paper "Person Re-Identification without Identification via Event Anonymization."  and the Event-based Person ReId dataset **Event-ReId**.

<!--- <img align="right" src="images/approach.gif" alt="approach" width="400"/>  
<img src="image/Event-ReId.gif" alt="over_view" width="600"/>      --> 

<p align="center">
  <img src="image/Event-ReId.gif" alt="over_view" width="800"/>
</p>

**Abstract**: <p align="justify"> Wide-scale use of visual surveillance in public spaces puts individual privacy at stake while increasing resource consumption (energy, bandwidth, and computation). Neuromorphic vision sensors (event-cameras) have been recently considered a valid solution to the privacy issue because they do not capture detailed RGB visual information of the subjects in the scene. However, recent deep learning architectures have been able to reconstruct images from event cameras with high fidelity, reintroducing a potential threat to privacy for event-based vision applications. In this paper, we aim to anonymize event-streams to protect the identity of human subjects against such image reconstruction attacks. To achieve this, we propose an end-to-end network architecture jointly optimized for the twofold objective of preserving privacy and performing a downstream task such as person ReId. Our network learns to scramble events, enforcing the degradation of images recovered from the privacy attacker. In this work, we also bring to the community the first-ever event-based person ReId dataset gathered to evaluate the performance of our approach.
  
  You can find a pdf of the paper **here**.

Install 
---------------------------------
Dependencies:

**-** Python 3.8

**-** PyTorch >= 1.0

**-** Torchmetrics

**-** NumPy

**-** Pandas

**-** OpenCV

### Install with Anaconda

We recommend a conda environment with Python 3.8 (the code is tested with Python 3.8.13).

Create Anaconda environment with the required dependencies as follows (make sure to adapt the CUDA toolkit version according to your setup):

```bash
conda create -n EvPReId_wo_Id python=3.8
conda activate EvPReId_wo_Id
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pandas
conda install -c conda-forge opencv
conda install -c conda-forge torchmetrics (or use pip install torchmetrics=0.11.0)
pip install tensorboard
pip install setuptools=59.5.0

```


Event-ReId Dataset, Preparation
---------------------------------
Download the Event-ReId dataset from **here** or you can use the [sample dataset](https://github.com/IIT-PAVIS/PReId_wo_Id/tree/main/data/sample_data)

Detail information to prepare the Event-ReId dataset is provided [here](https://github.com/IIT-PAVIS/PReId_wo_Id/blob/main/data/README.md)


Train
---------------------------------
run the following script to train **anonymization and person ReId model** jointly.

```bash
python train.py \
  --represent  ${voxel represent} \
  --batchsize  ${BATCHSIZE} \
  --ReId_loss  ${Id loss or Id+Triplet loss} \
  --AN_loss    ${choice of loss to train anonymization network, SSIM or MSE} \
  --num_Bin    ${number of temporal bins B} \
  --e2vid_path ${path to e2vid weights} \
  --epoch      ${60} \
  --file_name  ${FILE_NAME to save model and log file}
```

Note: before running train.py script, download [Eevnt-to-Image](https://github.com/uzh-rpg/rpg_e2vid) module weights from below, and save it to **e2vid_utils/pretrained/** directory

```bash
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O e2vid_utils/pretrained/E2VID_lightweight.pth.tar
```

Test
---------------------------------
run the following script to evaluate **person-reid w/o id**

```bash
python test.py \
  --model_path ${path to model weights}
```


If you use this project for your research, please cite the following:
```
@InProceedings{Ahmad2023eventreid,
    title     = {Person Re-Identification without Identification via Event Anonymization},
    author    = {Ahmad, Shafiq and Morerio, Pietro, and Del Bue, Alessio},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023}
}

```

## Acknowledgements

The event-to-video generation code borrows from the following open-source projects, whom we would like to thank:

- [E2VID](https://github.com/uzh-rpg/rpg_e2vid)

