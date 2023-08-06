# Person Re-Identification without Identification via Event Anonymization (ICCV 2023)
This is the official implementation of the ICCV23 paper "**Person Re-Identification without Identification via Event Anonymization**" - _Shafiq Ahmad, Pietro Morerio, Alessio Del Bue_. 

Pdf of the paper **here**

<!--- <img align="right" src="images/approach.gif" alt="approach" width="400"/>  
<img src="image/ReId_without_Id.gif" alt="over_view" width="600"/>      --> 

<p align="center">
  <img src="image/ReId_without_Id.gif" alt="over_view" title="My Image Caption" width="800"/>
</p>

> <p align="justify"> Event-to-video conversion can be regarded as a privacy attack on event-camera, which reconstructs a person's appearance from an event stream <code style="color : red">(a)</code>. We propose a learnable Event Anonymization network architecture <code style="color : red">(b)</code>, which deals with such attacks by scrambling the event stream so that reconstruction deteriorates while preserving the performance of an event-based downstream task (e.g., person ReId <code style="color : red">(c)</code>). We also consider a possible Inversion Attack <code style="color : red">(d)</code>, where the attacker tries to reverse the proposed anonymization's effect to attain image reconstruction <code style="color : red">(e)</code>, but our model is resistant to an inversion attack.
  
 
## **Abstract**
<p align="justify"> Wide-scale use of visual surveillance in public spaces puts individual privacy at stake while increasing resource consumption (energy, bandwidth, and computation). Neuromorphic vision sensors (event-cameras) have been recently considered a valid solution to the privacy issue because they do not capture detailed RGB visual information of the subjects in the scene. However, recent deep learning architectures have been able to reconstruct images from event cameras with high fidelity, reintroducing a potential threat to privacy for event-based vision applications. In this paper, we aim to anonymize event-streams to protect the identity of human subjects against such image reconstruction attacks. To achieve this, we propose an end-to-end network architecture jointly optimized for the twofold objective of preserving privacy and performing a downstream task such as person ReId. Our network learns to scramble events, enforcing the degradation of images recovered from the privacy attacker. In this work, we also bring to the community the first-ever event-based person ReId dataset gathered to evaluate the performance of our approach.
  

Installation
---------------------------------
### Dependencies

**-** Python 3.8

**-** PyTorch >= 1.0

**-** Torchmetrics

**-** Tensorboard

**-** NumPy

**-** OpenCV

### Setup Repository 
``` bash
git clone https://github.com/IIT-PAVIS/ReId_without_Id.git
cd ReId_without_Id/
mkdir -p e2vid_utils/pretrained
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O e2vid_utils/pretrained/E2VID_lightweight.pth.tar
```
### Setup Environment 
We recommend a conda environment with Python 3.8 (the code is tested with Python 3.8.13).

```bash
conda create -n EvPReId_wo_Id python=3.8
conda activate EvPReId_wo_Id
pip install torch torchvision  # make sure to adapt the CUDA toolkit version according to your setup
python -c "import torch; print(torch.cuda.is_available())" # check if Pytorch is correctly installed and Cuda is working
pip install torch torchvision --upgrade # if Pytorch is not working, try this
pip install torchmetrics==0.11.0
pip install opencv_python==4.7.0.72 setuptools==59.5.0 filelock pyyaml requests
pip install tensorboard
```

Prepaper Event-ReId Dataset
---------------------------------
Details information of dataset and train/test set preparation is provided [here](https://github.com/IIT-PAVIS/PReId_wo_Id/blob/main/data)

The **Event-ReId** can be downloaded from **here**, or you can use the [sample dataset](https://github.com/IIT-PAVIS/PReId_wo_Id/tree/main/data/sample_data) to check the train and test code. 


Train
---------------------------------
run the following script to train **anonymization and person ReId model** jointly.

```bash
python train.py \
  --represent  voxel \
  --batchsize  8 \
  --ReId_loss  sofmax \
  --AN_loss    SSIM \
  --num_Bin    5 \
  --e2vid_path e2vid_utils/pretrained/E2VID_lightweight.pth.tar \
  --epoch      60 \
  --file_name  training
```

Test
---------------------------------
run the following script to evaluate **person-reid w/o id**

```bash
python test.py --model_path training/net_59.pth  # set path to model weights
```

Cite our Paper
---------------
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

