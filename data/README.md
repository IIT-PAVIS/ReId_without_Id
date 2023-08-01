## Introduction

Event-ReId is the first person re-identification dataset captured from event cameras. It has recordings from four indoor 640x480 pixel event cameras and the files have event streams recorded from 33 persons, each person walking in the four event-camera views. 


### Event-ReId

- Identities:	 33

- Cameras:	 4	

- BBoxes:	 16K	

- Label method:  Hand+Yolo


## Dataset Details


### Format & Structure

Event-ReId provides event streams in text format (.txt). Each row in the events.txt file contains four values (timestamp, x, y, polarity) except the first row, which has two values, width and height (event-camera resolution). The event streams are arranged in the folder Event_ReId/ according to the following structure:

- (ID)/(camera_number)/(file_name).txt  -->  _(e.g. 008/cam03/events.txt)_


### Labels/BBOX

The event camera outputs an asynchronous event stream. However, for bounding box annotation, event timestamps must be synchronized. Therefore, we annotate an event stream of a fixed time duration of 33.3ms (synchronized with RGB camera of 30 FPS). As a result, each bbox coincides with a time window _**T**_ (= 33.3ms) of the event and the total time duration of each event is 3-4 sec. 

You can find each event stream file's corresponding bounding box labels in the directory e.g., _008/cam03/labels/_ in'.json' format. The label.json file contains two points of the box (the top-left _(x1,y1)_ and bottom-right _(x2,y2)_ corners) and frame number, e.g.,  _[[x1, y1, x2, y2], f_num]_. 

- Note: we have cleaned/removed the events lying outside the bounding box; you can use the following two-line code for extracting bbox.
  
  ``` bash
        """ 640x480 => bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1]) # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2]) # y - y_min
  ```    

 
### Train and Test IDs

We split train and test Ids as follows:

``` bash
Train Ids = [001, 003, 004, 006, 007, 009, 010, 012, 013, 015, 016, 018, 019, 021, 022, 024, 025, 027, 028, 030, 031, 033]

Test Ids = [002, 005, 008, 011, 014, 017, 020, 023, 026, 029, 032]
```


### Event-Partition & Generate Voxel-grid:

Use this script to generate an event voxel-grid from the event stream. We can generate voxel-grid either by accumulating events in a fixed time duration _T_ or accumulating a fixed number _N_ of events; we call it a constant time voxel-grid or constant count voxel-grid, respectively. 

Voxel-grid generates in two steps:
 
First, run the [event_partition.py](https://github.com/IIT-PAVIS/PReId_wo_Id/tree/main/data) script to partition the whole event stream (events.txt) into small chunks of event stream and then, during training, each small chunk will be converted into event voxel-grid using function *events_to_voxel_grid(events, num_bins, width, height)* in our dataloader.py script.

  
  ``` bash
  python event_partition.py \
        --input_dir     ${path to dataset} \
        --out_dir       ${path to output dataset} \
        --event_accumc  ${constant time or count voxel-grid} \
        --time_duration ${fixed time duration}\
        --event_count   ${fixed number of events}
  ```
You should get:
	  
``` bash
	   +-- out_dir/Event_ReId/001/ 
	   | cam01/
	   |       +-- 001_c1_001.txt 
	   |       +-- 001_c1_002.txt 
	   |       +-- 001_c1_002.txt
	   | cam02/
	   |       +-- 001_c2_001.txt 
	   |       +-- 001_c2_002.txt 
	   |       +-- 001_c2_002.txt
	   | ...
```
           

   
### Prepare Train and Test/Gallery set   

We utilized complete data for the training set (22 Ids). However, for the test/gallery set (11 Ids), we select every 5th voxel-grid. To construct the query set, we randomly select one voxel-grid per Id per camera from the test set (see [sample dataset](https://github.com/IIT-PAVIS/PReId_wo_Id/tree/main/data/sample_data)). After event partition, use the following script to split train and test sets.

  
   ``` bash
   python split_train_test.py 

   ```



If you use this dataset in your research, please kindly cite our work as,

```bash
@InProceedings{Ahmad2023eventreid,
title     = {Person Re-Identification without Identification via Event Anonymization},
author    = {Ahmad, Shafiq and Morerio, Pietro, and Del Bue, Alessio},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month     = {October},
year      = {2023}
}
```
