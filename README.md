# local_descriptor_gallery
Comparison of several local descriptors.

There are several different local descriptors in computer vision. This repository explore a few selected popular image 
local descriptors on some categories of image. They are 
* SIFT, 
* SURF, 
* ORB,
* STAR(CenSurE) with BRIEF desciptor,
* FAST
* Harris Corner, and
* Shi-Tomasi Corner.

<table align='center'>
<tr>
<td>
Preview
</td>
</tr>
<tr>
<td>
<img src = 'screenshots/sift.png' height = '400px'>
</td>
</tr>
<tr>
<td>
<img src = 'screenshots/surf.png' height = '400px'>
</td>
</tr>
<tr>
<td>
<img src = 'screenshots/orb.png' height = '400px'>
</td>
</tr>
<tr>
<td>
<img src = 'screenshots/star.png' height = '400px'>
</td>
</tr>
<tr>
<td>
<img src = 'screenshots/fast.png' height = '400px'>
</td>
</tr>
</table>



Any contribution is welcome!

# Requirement
Recommended system: Ubuntu 16.04 LTS

* Python3
* OpenCV3 build with OpenCV-Contrib
```
pip install opencv-contrib-python
```


# Acknowledgement

OpenCV Tutorial: https://docs.opencv.org/3.2.0/db/d27/tutorial_py_table_of_contents_feature2d.html

SIFT, SURF, ORB: https://github.com/fooock/opencv-notebooks

SIFT, Matching, Keypoint: https://github.com/igorrendulic/OpenCVSift/blob/master/sift_example.ipynb

Harris Corner Detection: https://github.com/keho98/python-cv-scratchpad/blob/master/Local%20Image%20Descriptors.ipynb

Matching: http://vgg.fiit.stuba.sk/2015-02/local-descriptors-in-opencv/
