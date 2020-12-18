# ShapeCorrespondence

We will explore different ideas


1. Finding matching between two piece-wise bezier shapes. This is achieved by minimizing the distortion when the anchor points of one shape is moved on top of  another. We are choosing the list of vertices for distortion calculation. We are assuming that the two shapes contain the same number of anchor points. Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using only the anchors of the curve. Once can envisage distortion calculation involving the control points as well. 
<b>Distortion\.main()</b> to see an example.
