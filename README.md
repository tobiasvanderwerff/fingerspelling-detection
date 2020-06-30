# Fingerspelling Detection Using Templates Based on Pose Information

This repository contains the source code that accompanies the bachelor thesis "Detecting Fingerspelling in Sign Language Videos Using Pose Information". 

## Usage

Install dependencies:

``
pip install -r requirements.txt
``
  
Try it on a video, for example one from the Corpus NGT [1], [2]:  

``
python3 main.py examples/CNGT0319_S015_b.mpg --op_out examples/op_out --template_dir template_sets/generic
``

## Processing new videos

For processing your own videos, follow the following steps.
- Process video with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), saving the results in json format:
  - Make a directory for the OpenPose output. Within this folder, create another folder that will contain the OpenPose output for the video you want to process.
    If your video is called 'my_video.mpg', the json output should be stored in a folder called 'my_video', e.g. ``op_out/my_video/``.
  - Process video with OpenPose, e.g. using the Windows precompiled version of OpenPose:   
  
``
bin\OpenPoseDemo.exe --video /path/to/video/ --hand --write_json op_out/my_video/
``  

- Run the system, e.g using the generic template set: 

`` python3 main.py path/to/video/ --op_out op_out/ --template_dir template_sets/generic``

It should be noted that using the OpenPose Windows executable, OpenPose seems to skip some of the initial frames of a video if there is no person appearing
in the frame. This can make the timing estimates for the fingerspelling predictions somewhat less accurate, since these are based on frame numbers. 
Therefore using the OpenPose Python API is preferred, for example by using ``src/process_video_with_openpose.py`` (which requires you to fill in the 
correct OpenPose path in ``src/op1.py``).

## References

[1] Onno Crasborn, Inge Zwitserlood & Johan Ros. 2008. The Corpus NGT. An open access digital corpus of movies with annotations of Sign Language of the Netherlands. Centre for Language Studies, Radboud University Nijmegen. URL: http://hdl.handle.net/hdl:1839/00-0000-0000-0004-DF8E-6. ISLRN: 175-346-174-413-3.  
[2] Onno Crasborn & Inge Zwitserlood (2008) The Corpus NGT: an online corpus for professionals and laymen, In: Construction and Exploitation of Sign Language Corpora. 3rd Workshop on the Representation and Processing of Sign Languages, O. Crasborn, T. Hanke, E. Efthimiou, I. Zwitserlood & E. Thoutenhoofd, eds. ELDA, Paris. pp 44-49.  
