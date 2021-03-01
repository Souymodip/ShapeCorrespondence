# ShapeCorrespondence

We will explore different ideas


1. Finding matching between two piece-wise bezier shapes. This is achieved by minimizing the distortion when the anchor points of one shape is moved on top of  another. We are choosing the list of vertices for distortion calculation. We are assuming that the two shapes contain the same number of anchor points. Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using only the anchors of the curve. Once can envisage distortion calculation involving the control points as well. 
<b>Distortion\.main()</b> to see an example.

2. We will run a gradient descent on the overlap between two arts, which are normalized and their respective centeroid is translated to origin. Refer to the dcoument <b> Differntiable Overlap </b> for detail.
  Run <b>FuzzyPixel.main()</b> to see an example where two rectangles are placed on top of each other. One has been rotated about the orogin by some degrees. The example plots the oerlap function <img src="https://render.githubusercontent.com/render/math?math=\Omega(\theta)"> with respect to rotated angle <img src="https://render.githubusercontent.com/render/math?math=\theta">.
  
3. The shape outline curvature is represented as a function of normalised distance. The similarity between two points on their respective shapes is measured as metric of similarity between the curvature vs normalized distance graph. This metric is by construction, rotation and scale invariant. The following graph similarity metrics are considered:</br>

     3.a. <b>Discrete Fourier Transform Descriptor </b>: The graph is transfomed to its frequency domain. The frequencies define a infinite vector space. We approximated to a predefined size. Any vector represents the amplitudes of the various frequences. The similarity metric is the l2-norm of two vector in this space.
  
     3.b. <b>Enclosed Area </b>: The enclosing area between the two graph defines the similarity.
     
     Run <b><i>ShapeSimilarity.py</i></b> to see an example. The following correspondence were established with this method
     <br>
     <img src="https://github.com/Souymodip/ShapeCorrespondence/blob/main/Images/CurveMatching/image3.png" alt="" width="360" height="250">
     

4. Bringing shape similarity with Neural Networks: The task at hand is to measure the degree of similarity of curvature-v-length graph of two shapes. Formally, 

      </t><a href="https://www.codecogs.com/eqnedit.php?latex=f_1&space;\sim&space;f_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1&space;\sim&space;f_2" title="f_1 \sim f_2" /></a> is the measure of similairty between two functions. This is achieved in following steps
      
      4.a. A training set is created containing various perturbation of <a href="https://www.codecogs.com/eqnedit.php?latex=f_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1" title="f_1" /></a>. This can be found in <b><i> Generator.py </b></i>
      
      4.b. A Neural Network is concieved which models the classifier. The neural network is trained in <b><i> Discriminator.py </i></b>.

      4.c. The trained model is used to measure the value of </t><a href="https://www.codecogs.com/eqnedit.php?latex=f_1&space;\sim&space;f_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1&space;\sim&space;f_2" title="f_1 \sim f_2" /></a>
      
      
5. <b> Cut-Match </b>     
