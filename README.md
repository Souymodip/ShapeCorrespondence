# ShapeCorrespondence

We will explore different ideas


1. Finding matching between two piece-wise bezier shapes. This is achieved by minimizing the distortion when the anchor points of one shape is moved on top of  another. We are choosing the list of vertices for distortion calculation. We are assuming that the two shapes contain the same number of anchor points. Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using only the anchors of the curve. Once can envisage distortion calculation involving the control points as well. 
<b>Distortion\.main()</b> to see an example.

2. We will run a gradient descent on the overlap between two arts, which are normalized and their respective centeroid is translated to origin. Refer to the dcoument <b> Differntiable Overlap </b> for detail.
  Run <b>FuzzyPixel.main()</b> to see an example where two rectangles are placed on top of each other. One has been rotated about the orogin by some degrees. The example plots the oerlap function <img src="https://render.githubusercontent.com/render/math?math=\Omega(\theta)"> with respect to rotated angle <img src="https://render.githubusercontent.com/render/math?math=\theta">.
  
3. The shape outline curvature is represented as a function of normalised distance. The similarity between two points on their respective shapes is measured as metric of similarity between the curvature vs normalized distance graph. This metric is by construction, rotation and scale invariant. The following graph similarity metrics are considered:</br>

     * <b>Discrete Fourier Transform Descriptor </b>: The graph is transfomed to its frequency domain. The frequencies define a infinite vector space. We approximated to a predefined size. Any vector represents the amplitudes of the various frequences. The similarity metric is the l2-norm of two vector in this space.
  
     * <b>Enclosed Area </b>: The enclosing area between the two graph defines the similarity.
     
     Run <b><i>ShapeSimilarity.py</i></b> to see an example. The following correspondence were established with this method
     
     <br>
     <img src="https://github.com/Souymodip/ShapeCorrespondence/blob/main/Images/CurveMatching/image3.png" alt="" width="360" height="250">
     <br>

4. Bringing shape similarity with Neural Networks: The task at hand is to measure the degree of similarity of curvature-v-length graph of two shapes. Formally, 

      </t><a href="https://www.codecogs.com/eqnedit.php?latex=f_1&space;\sim&space;f_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1&space;\sim&space;f_2" title="f_1 \sim f_2" /></a> is the measure of similairty between two functions. This is achieved in following steps
      
      * A training set is created containing various perturbation of <a href="https://www.codecogs.com/eqnedit.php?latex=f_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1" title="f_1" /></a>. This can be found in <b><i> Generator.py </b></i>
      
      * A Neural Network is concieved which models the classifier. The neural network is trained in <b><i> Discriminator.py </i></b>.

      * The trained model is used to measure the value of </t><a href="https://www.codecogs.com/eqnedit.php?latex=f_1&space;\sim&space;f_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_1&space;\sim&space;f_2" title="f_1 \sim f_2" /></a>
      
      <img src="https://github.com/Souymodip/ShapeCorrespondence/blob/main/image2.png" alt="" width="360" height="250">
      
      <br>
      
      <img src="https://github.com/Souymodip/ShapeCorrespondence/blob/main/image3.png" alt="" width="360" height="250">
      
      <br>
      
      <img src="https://github.com/Souymodip/ShapeCorrespondence/blob/main/image5.png" alt="" width="400" height="250">
     
      <br>
      
      
      
5.<b> Cut-Match </b> The art is broken down into sequence of cuts. The objective is to match cuts from one art to the other. The algorithm can be viewed as 2-player game (without objective function). First player (P1) chooses a cut to from art1 and second player (P2) finds a cut from art2 which is most similar to the given cut. This process is repeated till the entire art is covered. Below we formaly define what is a cut.
      <br>
      An art is defined as a sequence of points  <a href="https://www.codecogs.com/eqnedit.php?latex=\{p_0,&space;p_1,&space;\dots,&space;p_n\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{p_0,&space;p_1,&space;\dots,&space;p_n\}" title="\{p_0, p_1, \dots, p_n\}" /></a>.
      Define a distance function  <a href="https://www.codecogs.com/eqnedit.php?latex=X&space;:&space;P&space;\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X&space;:&space;P&space;\rightarrow&space;\mathbb{R}" title="X : P \rightarrow \mathbb{R}" /></a> such that,
      <br>
      <a href="https://www.codecogs.com/eqnedit.php?latex=X(p_0)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X(p_0)&space;=&space;0" title="X(p_0) = 0" /></a>  
       <a href="https://www.codecogs.com/eqnedit.php?latex=X(p_i)&space;=&space;X(p_{i-1})&space;&plus;&space;\|&space;p_i&space;-&space;p_{i-1}\|_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X(p_i)&space;=&space;X(p_{i-1})&space;&plus;&space;\|&space;p_i&space;-&space;p_{i-1}\|_2" title="X(p_i) = X(p_{i-1}) + \| p_i - p_{i-1}\|_2" /></a>
      <br> <br>
      A **cut** at _x_ of length _l_ is defined as a sequence of points  <a href="https://www.codecogs.com/eqnedit.php?latex=\{s,&space;p_i,&space;\dots,&space;p_{i&plus;m},&space;e\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{s,&space;p_i,&space;\dots,&space;p_{i&plus;m},&space;e\}" title="\{s, p_i, \dots, p_{i+m}, e\}" /></a>  
      such that <br>
       <a href="https://www.codecogs.com/eqnedit.php?latex=x\leq&space;X(q_i)&space;\leq&space;x&space;&plus;&space;l,&space;\forall&space;i&space;\in&space;[1,&space;m&space;-1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x\leq&space;X(q_i)&space;\leq&space;x&space;&plus;&space;l,&space;\forall&space;i&space;\in&space;[1,&space;m&space;-1]" title="x\leq X(q_i) \leq x + l, \forall i \in [1, m -1]" /></a>
      <br> and <br>
      <a href="https://www.codecogs.com/eqnedit.php?latex=X(p_{i-1})&space;&plus;&space;\|p_{i-1}&space;-&space;s&space;\|_2&space;=&space;x,&space;\&space;\&space;\&space;X(p_{i&plus;m})&space;&plus;&space;\|p_{i&plus;m}&space;-&space;e\|_2&space;=&space;x&space;&plus;&space;l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X(p_{i-1})&space;&plus;&space;\|p_{i-1}&space;-&space;s&space;\|_2&space;=&space;x,&space;\&space;\&space;\&space;X(p_{i&plus;m})&space;&plus;&space;\|p_{i&plus;m}&space;-&space;e\|_2&space;=&space;x&space;&plus;&space;l" title="X(p_{i-1}) + \|p_{i-1} - s \|_2 = x, \ \ \ X(p_{i+m}) + \|p_{i+m} - e\|_2 = x + l" /></a>
      
      
      
