# ShapeCorrespondence

We will explore different ideas


1. Finding matching between two piece-wise bezier shapes. This is achieved by minimizing the distortion when the anchor points of one shape is moved on top of  another. We are choosing the list of vertices for distortion calculation. We are assuming that the two shapes contain the same number of anchor points. Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using only the anchors of the curve. Once can envisage distortion calculation involving the control points as well. 
<b>Distortion\.main()</b> to see an example.

2. We will run a gradient descent on the overlap between two arts, which are normalized and their respective centeroid is translated to origin. Refer to the dcoument <b> Differntiable Overlap </b> for detail.
  Run <b>FuzzyPixel.main()</b> to see an example where two rectangles are placed on top of each other. One has been rotated about the orogin by some degrees. The example plots the oerlap function <img src="https://render.githubusercontent.com/render/math?math=\Omega(\theta)"> with respect to rotated angle <img src="https://render.githubusercontent.com/render/math?math=\theta">. 

