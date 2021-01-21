import Art
import numpy as np

'''
.---------------------------------------------------------------.
|   Level 1 Test cases                                          |
|   Contains set of pair of arts and their non-rigid transform  |
|   The two arts have the same number of anchor points          |    
'---------------------------------------------------------------'
'''

def get_test(index):
    if index == 0:
        test0 = Art.PieceWiseBezier(np.array([
        [[0.667208, -1.7588], [0.691714, -1.79214], [0.665447, -1.7674]] , [[0.663044, -1.7795], [0.663721, -1.77339], [0.655354, -1.84885]] , [[0.640231, -1.98758], [0.647877, -1.91822], [0.636594, -2.02056]] , [[0.630749, -2.08618], [0.625506, -2.05472], [0.637512, -2.12677]] , [[0.609297, -2.19295], [0.624633, -2.15879], [0.596713, -2.22098]] , [[0.553874, -2.20925], [0.575541, -2.208], [0.562493, -2.19464]] , [[0.576449, -2.17097], [0.569515, -2.18283], [0.594812, -2.13956]] , [[0.590796, -2.07111], [0.595506, -2.10766], [0.584601, -2.02303]] , [[0.591328, -1.9244], [0.590881, -1.97338], [0.591708, -1.88273]] , [[0.591083, -1.79937], [0.592071, -1.84103], [0.590734, -1.78468]] , [[0.584201, -1.75542], [0.584913, -1.77015], [0.583351, -1.73784]] , [[0.587207, -1.70324], [0.578848, -1.71584], [0.61665, -1.6589]] , [[0.601809, -1.56596], [0.609474, -1.61294], [0.600603, -1.55857]] , [[0.59735, -1.54392], [0.598572, -1.55131], [0.593489, -1.52057]] , [[0.556218, -1.50827], [0.579354, -1.50933], [0.531319, -1.50713]] , [[0.48153, -1.50492], [0.506428, -1.50497], [0.45698, -1.50487]] , [[0.432335, -1.47161], [0.439195, -1.49622], [0.422751, -1.43721]] , [[0.405846, -1.36796], [0.409999, -1.40307], [0.401086, -1.32772]] , [[0.345375, -1.29086], [0.380642, -1.30351], [0.320169, -1.28182]] , [[0.308058, -1.23898], [0.312104, -1.26386], [0.331899, -1.2444]] , [[0.375882, -1.25502], [0.354315, -1.24833], [0.383511, -1.25738]] , [[0.39528, -1.27261], [0.390883, -1.26534], [0.425843, -1.32317]] , [[0.471231, -1.43196], [0.457914, -1.37302], [0.475354, -1.45021]] , [[0.507576, -1.45913], [0.487501, -1.45871], [0.553342, -1.46011]] , [[0.644806, -1.46463], [0.599058, -1.46442], [0.656458, -1.46468]] , [[0.679835, -1.44898], [0.668158, -1.45447], [0.673847, -1.43751]] , [[0.661296, -1.41524], [0.670425, -1.42302], [0.645963, -1.40216]] , [[0.610735, -1.37955], [0.627, -1.39334], [0.648921, -1.36729]] , [[0.720756, -1.34465], [0.677824, -1.33257], [0.728985, -1.34696]] , [[0.744429, -1.35462], [0.737836, -1.34957], [0.754864, -1.36261]] , [[0.776688, -1.38552], [0.763634, -1.37278], [0.776094, -1.38679]] , [[0.770031, -1.39866], [0.774361, -1.39748], [0.745911, -1.40526]] , [[0.731376, -1.44477], [0.743605, -1.42913], [0.718623, -1.46108]] , [[0.748533, -1.47889], [0.727535, -1.47773], [0.756813, -1.47935]] , [[0.773484, -1.47792], [0.765235, -1.47889], [0.790983, -1.47588]] , [[0.804458, -1.50039], [0.802588, -1.48579], [0.809467, -1.53949]] , [[0.865955, -1.58467], [0.844782, -1.55279], [0.872993, -1.57212]] , [[0.88125, -1.55708], [0.87807, -1.56498], [0.896093, -1.52019]] , [[0.925625, -1.44638], [0.90925, -1.48256], [0.930543, -1.43551]] , [[0.95369, -1.42033], [0.942509, -1.42539], [0.975409, -1.41051]] , [[1.02648, -1.39532], [0.998857, -1.40452], [1.026, -1.40406]] , [[1.0246, -1.41903], [1.02844, -1.4139], [1.0139, -1.43331]] , [[0.987856, -1.45774], [1.00295, -1.45059], [0.953197, -1.47415]] , [[0.93344, -1.53616], [0.943071, -1.50289], [0.923408, -1.57081]] , [[0.892848, -1.63703], [0.907577, -1.6039], [0.887293, -1.64953]] , [[0.864143, -1.64314], [0.875812, -1.65333], [0.84049, -1.6225]] , [[0.794341, -1.57997], [0.815722, -1.6028], [0.771113, -1.55517]] , [[0.752596, -1.58519], [0.76601, -1.55484], [0.745886, -1.60037]] , [[0.729374, -1.62935], [0.737573, -1.61489], [0.709669, -1.66413]] , [[0.734626, -1.73], [0.711243, -1.69796], [0.753261, -1.75554]] , [[0.792859, -1.8049], [0.77277, -1.78049], [0.815511, -1.83243]] , [[0.798646, -1.8905], [0.814706, -1.86013], [0.77264, -1.93968]] , [[0.723231, -2.03941], [0.746854, -1.98906], [0.700089, -2.08874]] , [[0.757253, -2.15486], [0.712086, -2.12461], [0.763169, -2.15882]] , [[0.772347, -2.17278], [0.766862, -2.1661], [0.742803, -2.18455]] , [[0.702001, -2.14578], [0.712273, -2.17286], [0.695907, -2.12972]] , [[0.670602, -2.11875], [0.687645, -2.12087], [0.652803, -2.11653]] , [[0.661933, -2.09409], [0.649501, -2.10884], [0.684246, -2.06761]] , [[0.692037, -2.00176], [0.693172, -2.03725], [0.690547, -1.9552]] , [[0.731602, -1.87312], [0.701564, -1.91098], [0.735486, -1.86822]] , [[0.735935, -1.85347], [0.738601, -1.85736], [0.714563, -1.8223]]
            ]), is_closed=True)
        test1 = Art.PieceWiseBezier(np.array([
        [[5.22028, -1.8226], [5.20595, -1.86521], [5.20052, -1.81623]] , [[5.16448, -1.80454], [5.18239, -1.8107], [5.14585, -1.79814]] , [[5.10899, -1.78425], [5.12722, -1.79168], [5.09479, -1.77846]] , [[5.08299, -1.79693], [5.08677, -1.77792], [5.07194, -1.8524]] , [[5.04695, -1.96277], [5.05705, -1.90714], [5.03839, -2.00994]] , [[5.0311, -2.10525], [5.02572, -2.05895], [5.03545, -2.14273]] , [[5.00743, -2.19654], [5.01763, -2.16712], [5.00223, -2.21157]] , [[4.95631, -2.209], [4.98018, -2.21568], [4.96729, -2.188]] , [[4.9886, -2.14605], [4.97944, -2.16766], [4.9935, -2.13448]] , [[4.99489, -2.10796], [4.9953, -2.12065], [4.99374, -2.07167]] , [[4.98883, -1.99919], [4.98962, -2.03547], [4.98827, -1.97348]] , [[4.99277, -1.92196], [4.99046, -1.94763], [4.99588, -1.88736]] , [[5.00376, -1.81832], [5.00118, -1.85295], [5.00508, -1.80054]] , [[5.00109, -1.7647], [5.00557, -1.78166], [4.99084, -1.72592]] , [[5.03109, -1.68395], [4.99695, -1.70802], [5.06963, -1.65678]] , [[5.14499, -1.6], [5.10732, -1.62838], [5.15139, -1.59518]] , [[5.16098, -1.58145], [5.15675, -1.58834], [5.1712, -1.56483]] , [[5.1474, -1.5484], [5.16747, -1.55297], [5.11657, -1.54139]] , [[5.05512, -1.52673], [5.08531, -1.53582], [5.04521, -1.52374]] , [[5.03045, -1.50356], [5.03332, -1.51312], [5.02, -1.46862]] , [[5.00542, -1.39698], [5.00997, -1.43304], [5.00008, -1.35463]] , [[4.9399, -1.31532], [4.97938, -1.32909], [4.91899, -1.30803]] , [[4.91193, -1.26528], [4.90878, -1.28779], [4.92656, -1.27035]] , [[4.95538, -1.27969], [4.94069, -1.27853], [4.99526, -1.28283]] , [[5.01914, -1.34276], [5.00345, -1.31765], [5.03647, -1.37051]] , [[5.05997, -1.43259], [5.05184, -1.40113], [5.07185, -1.4786]] , [[5.12394, -1.49382], [5.07613, -1.48406], [5.16252, -1.50169]] , [[5.23799, -1.52586], [5.19978, -1.51586], [5.2475, -1.52834]] , [[5.26755, -1.52434], [5.26434, -1.52931], [5.27673, -1.51016]] , [[5.25387, -1.49027], [5.26851, -1.4982], [5.23872, -1.48208]] , [[5.21039, -1.45983], [5.22423, -1.47265], [5.24699, -1.44604]] , [[5.31712, -1.42396], [5.27453, -1.41224], [5.32332, -1.42566]] , [[5.33505, -1.43086], [5.33065, -1.42683], [5.3473, -1.44207]] , [[5.37048, -1.46747], [5.35828, -1.45466], [5.35557, -1.48475]] , [[5.32172, -1.52394], [5.33869, -1.50439], [5.30191, -1.54676]] , [[5.33662, -1.57273], [5.30731, -1.5658], [5.35718, -1.57759]] , [[5.35819, -1.60879], [5.36616, -1.58917], [5.34518, -1.64078]] , [[5.38188, -1.68342], [5.36979, -1.66006], [5.3826, -1.68481]] , [[5.39998, -1.6787], [5.39828, -1.68247], [5.41673, -1.64169]] , [[5.44751, -1.56643], [5.43246, -1.60418], [5.45674, -1.54329]] , [[5.49742, -1.52849], [5.47152, -1.53029], [5.50784, -1.52777]] , [[5.52799, -1.51565], [5.51753, -1.51933], [5.5352, -1.51311]] , [[5.55078, -1.51137], [5.54316, -1.51272], [5.54979, -1.51862]] , [[5.54733, -1.53267], [5.55139, -1.52825], [5.53525, -1.54581]] , [[5.50728, -1.56849], [5.52191, -1.55831], [5.48147, -1.58642]] , [[5.46022, -1.64068], [5.46269, -1.607], [5.45711, -1.6832]] , [[5.40563, -1.74683], [5.43495, -1.71752], [5.40112, -1.75134]] , [[5.38088, -1.74782], [5.3849, -1.75197], [5.35858, -1.72478]] , [[5.31801, -1.67493], [5.33641, -1.70109], [5.30067, -1.65028]] , [[5.27048, -1.66227], [5.29691, -1.64847], [5.23677, -1.67987]] , [[5.16731, -1.71119], [5.20146, -1.69439], [5.15523, -1.71713]] , [[5.1332, -1.73337], [5.14453, -1.7259], [5.13384, -1.73566]] , [[5.13512, -1.74024], [5.13448, -1.73795], [5.15596, -1.74259]] , [[5.19763, -1.74742], [5.17693, -1.74417], [5.27075, -1.75887]] , [[5.28076, -1.87021], [5.29747, -1.79506], [5.27124, -1.91305]] , [[5.24234, -1.99652], [5.25626, -1.95477], [5.22549, -2.04706]] , [[5.27139, -2.11015], [5.23214, -2.07458], [5.27609, -2.11441]] , [[5.2814, -2.12745], [5.27813, -2.12161], [5.27464, -2.12866]] , [[5.26137, -2.13051], [5.26669, -2.13274], [5.2496, -2.12557]] , [[5.22886, -2.11013], [5.23618, -2.11973], [5.21776, -2.09557]] , [[5.1902, -2.07364], [5.21035, -2.07923], [5.18688, -2.07272]] , [[5.18677, -2.04993], [5.18364, -2.05604], [5.20225, -2.01977]] , [[5.20845, -1.9546], [5.21244, -1.98953], [5.20204, -1.89858]]
            ]), is_closed=True)

        test0.apply(Art.Scale(15))
        test0.apply(Art.Translate([-15, 30]))
        test1.apply(Art.Scale(15))
        test1.apply(Art.Translate([-70, 30]))
        return test0, test1

    if index == 1:
        test0 = Art.PieceWiseBezier(np.array([
        [[0.667208, -1.7588], [0.691714, -1.79214], [0.665447, -1.7674]] , [[0.663044, -1.7795], [0.663721, -1.77339], [0.655354, -1.84885]] , [[0.640231, -1.98758], [0.647877, -1.91822], [0.636594, -2.02056]] , [[0.630749, -2.08618], [0.625506, -2.05472], [0.637512, -2.12677]] , [[0.609297, -2.19295], [0.624633, -2.15879], [0.596713, -2.22098]] , [[0.553874, -2.20925], [0.575541, -2.208], [0.562493, -2.19464]] , [[0.576449, -2.17097], [0.569515, -2.18283], [0.594812, -2.13956]] , [[0.590796, -2.07111], [0.595506, -2.10766], [0.584601, -2.02303]] , [[0.591328, -1.9244], [0.590881, -1.97338], [0.591708, -1.88273]] , [[0.591083, -1.79937], [0.592071, -1.84103], [0.590734, -1.78468]] , [[0.584201, -1.75542], [0.584913, -1.77015], [0.583351, -1.73784]] , [[0.587207, -1.70324], [0.578848, -1.71584], [0.61665, -1.6589]] , [[0.601809, -1.56596], [0.609474, -1.61294], [0.600603, -1.55857]] , [[0.59735, -1.54392], [0.598572, -1.55131], [0.593489, -1.52057]] , [[0.556218, -1.50827], [0.579354, -1.50933], [0.531319, -1.50713]] , [[0.48153, -1.50492], [0.506428, -1.50497], [0.45698, -1.50487]] , [[0.432335, -1.47161], [0.439195, -1.49622], [0.422751, -1.43721]] , [[0.405846, -1.36796], [0.409999, -1.40307], [0.401086, -1.32772]] , [[0.345375, -1.29086], [0.380642, -1.30351], [0.320169, -1.28182]] , [[0.308058, -1.23898], [0.312104, -1.26386], [0.331899, -1.2444]] , [[0.375882, -1.25502], [0.354315, -1.24833], [0.383511, -1.25738]] , [[0.39528, -1.27261], [0.390883, -1.26534], [0.425843, -1.32317]] , [[0.471231, -1.43196], [0.457914, -1.37302], [0.475354, -1.45021]] , [[0.507576, -1.45913], [0.487501, -1.45871], [0.553342, -1.46011]] , [[0.644806, -1.46463], [0.599058, -1.46442], [0.656458, -1.46468]] , [[0.679835, -1.44898], [0.668158, -1.45447], [0.673847, -1.43751]] , [[0.661296, -1.41524], [0.670425, -1.42302], [0.645963, -1.40216]] , [[0.610735, -1.37955], [0.627, -1.39334], [0.648921, -1.36729]] , [[0.720756, -1.34465], [0.677824, -1.33257], [0.728985, -1.34696]] , [[0.744429, -1.35462], [0.737836, -1.34957], [0.754864, -1.36261]] , [[0.776688, -1.38552], [0.763634, -1.37278], [0.776094, -1.38679]] , [[0.770031, -1.39866], [0.774361, -1.39748], [0.745911, -1.40526]] , [[0.731376, -1.44477], [0.743605, -1.42913], [0.718623, -1.46108]] , [[0.748533, -1.47889], [0.727535, -1.47773], [0.756813, -1.47935]] , [[0.773484, -1.47792], [0.765235, -1.47889], [0.790983, -1.47588]] , [[0.804458, -1.50039], [0.802588, -1.48579], [0.809467, -1.53949]] , [[0.865955, -1.58467], [0.844782, -1.55279], [0.872993, -1.57212]] , [[0.88125, -1.55708], [0.87807, -1.56498], [0.896093, -1.52019]] , [[0.925625, -1.44638], [0.90925, -1.48256], [0.930543, -1.43551]] , [[0.95369, -1.42033], [0.942509, -1.42539], [0.975409, -1.41051]] , [[1.02648, -1.39532], [0.998857, -1.40452], [1.026, -1.40406]] , [[1.0246, -1.41903], [1.02844, -1.4139], [1.0139, -1.43331]] , [[0.987856, -1.45774], [1.00295, -1.45059], [0.953197, -1.47415]] , [[0.93344, -1.53616], [0.943071, -1.50289], [0.923408, -1.57081]] , [[0.892848, -1.63703], [0.907577, -1.6039], [0.887293, -1.64953]] , [[0.864143, -1.64314], [0.875812, -1.65333], [0.84049, -1.6225]] , [[0.794341, -1.57997], [0.815722, -1.6028], [0.771113, -1.55517]] , [[0.752596, -1.58519], [0.76601, -1.55484], [0.745886, -1.60037]] , [[0.729374, -1.62935], [0.737573, -1.61489], [0.709669, -1.66413]] , [[0.734626, -1.73], [0.711243, -1.69796], [0.753261, -1.75554]] , [[0.792859, -1.8049], [0.77277, -1.78049], [0.815511, -1.83243]] , [[0.798646, -1.8905], [0.814706, -1.86013], [0.77264, -1.93968]] , [[0.723231, -2.03941], [0.746854, -1.98906], [0.700089, -2.08874]] , [[0.757253, -2.15486], [0.712086, -2.12461], [0.763169, -2.15882]] , [[0.772347, -2.17278], [0.766862, -2.1661], [0.742803, -2.18455]] , [[0.702001, -2.14578], [0.712273, -2.17286], [0.695907, -2.12972]] , [[0.670602, -2.11875], [0.687645, -2.12087], [0.652803, -2.11653]] , [[0.661933, -2.09409], [0.649501, -2.10884], [0.684246, -2.06761]] , [[0.692037, -2.00176], [0.693172, -2.03725], [0.690547, -1.9552]] , [[0.731602, -1.87312], [0.701564, -1.91098], [0.735486, -1.86822]] , [[0.735935, -1.85347], [0.738601, -1.85736], [0.714563, -1.8223]]
            ]), is_closed=True, show_control=True)
        test1 = Art.PieceWiseBezier(np.array([
        [[5.54642, -3.31189], [5.55044, -3.32374], [5.59194, -3.31395]] , [[5.66433, -3.33561], [5.63238, -3.3052], [5.66933, -3.34036]] , [[5.67667, -3.35245], [5.67436, -3.34615], [5.68147, -3.36556]] , [[5.68934, -3.39701], [5.68446, -3.37934], [5.68544, -3.39866]] , [[5.66993, -3.40512], [5.67786, -3.40242], [5.64732, -3.41282]] , [[5.60204, -3.42793], [5.62401, -3.41874], [5.58247, -3.43611]] , [[5.5952, -3.47002], [5.58048, -3.45424], [5.61717, -3.49358]] , [[5.59329, -3.52012], [5.61706, -3.49669], [5.5798, -3.53341]] , [[5.57883, -3.59228], [5.56987, -3.57777], [5.58091, -3.59566]] , [[5.59647, -3.59472], [5.5928, -3.59737], [5.63264, -3.5686]] , [[5.70337, -3.51415], [5.66871, -3.54223], [5.72159, -3.49939]] , [[5.75919, -3.507], [5.73909, -3.50124], [5.76998, -3.51009]] , [[5.79338, -3.50997], [5.78209, -3.50821], [5.8021, -3.51133]] , [[5.81902, -3.51745], [5.81049, -3.51488], [5.814, -3.52503]] , [[5.80367, -3.53927], [5.81039, -3.5372], [5.78648, -3.54459]] , [[5.74986, -3.54762], [5.76794, -3.54719], [5.71851, -3.54835]] , [[5.67502, -3.585], [5.69363, -3.55919], [5.64765, -3.62295]] , [[5.56529, -3.65857], [5.60652, -3.64137], [5.54721, -3.66611]] , [[5.53588, -3.63692], [5.53847, -3.65134], [5.5309, -3.60927]] , [[5.52576, -3.5532], [5.52649, -3.5812], [5.52499, -3.52421]] , [[5.49583, -3.52182], [5.52391, -3.52245], [5.45272, -3.52086]] , [[5.3666, -3.51538], [5.40969, -3.51722], [5.35479, -3.51488]] , [[5.33101, -3.51755], [5.34287, -3.51676], [5.33043, -3.51991]] , [[5.32928, -3.52464], [5.32985, -3.52228], [5.33593, -3.53003]] , [[5.34925, -3.54077], [5.34243, -3.53561], [5.36572, -3.55322]] , [[5.39862, -3.57809], [5.38357, -3.56414], [5.43237, -3.60937]] , [[5.40226, -3.68898], [5.43498, -3.65612], [5.35782, -3.73363]] , [[5.26418, -3.81817], [5.31132, -3.77634], [5.23447, -3.84453]] , [[5.24411, -3.90882], [5.2341, -3.87456], [5.24645, -3.91682]] , [[5.24299, -3.93521], [5.24351, -3.92637], [5.23501, -3.93079]] , [[5.21961, -3.9215], [5.22486, -3.92809], [5.21323, -3.91349]] , [[5.20709, -3.89228], [5.21041, -3.90239], [5.20188, -3.87645]] , [[5.19345, -3.84423], [5.19613, -3.86056], [5.19222, -3.83674]] , [[5.20063, -3.82079], [5.19534, -3.82362], [5.23804, -3.80074]] , [[5.28288, -3.73369], [5.26281, -3.77087], [5.30147, -3.69924]] , [[5.35945, -3.63918], [5.31704, -3.66078], [5.34607, -3.62553]] , [[5.32274, -3.60151], [5.33386, -3.614], [5.30904, -3.58612]] , [[5.28344, -3.55381], [5.29494, -3.57081], [5.27279, -3.53809]] , [[5.25204, -3.55117], [5.26418, -3.53849], [5.20834, -3.59679]] , [[5.12166, -3.6885], [5.16006, -3.63881], [5.0871, -3.7332]] , [[5.02101, -3.82598], [5.03774, -3.76783], [5.01938, -3.83167]] , [[5.00516, -3.83983], [5.01103, -3.83588], [4.99358, -3.84762]] , [[4.96986, -3.86233], [4.98096, -3.85395], [4.95008, -3.87728]] , [[4.91137, -3.8506], [4.92679, -3.87305], [4.93326, -3.83797]] , [[4.97661, -3.81211], [4.95621, -3.82682], [4.98785, -3.80401]] , [[5.00341, -3.77839], [4.99605, -3.79068], [5.0216, -3.74802]] , [[5.05619, -3.68623], [5.03779, -3.71646], [5.06828, -3.66637]] , [[5.09634, -3.62904], [5.08234, -3.64765], [5.12119, -3.59601]] , [[5.17157, -3.53046], [5.14737, -3.56396], [5.182, -3.51602]] , [[5.1968, -3.48298], [5.19156, -3.49984], [5.20715, -3.44964]] , [[5.26294, -3.42843], [5.22713, -3.43219], [5.30841, -3.42366]] , [[5.39916, -3.41223], [5.35376, -3.41769], [5.40164, -3.41193]] , [[5.40652, -3.41107], [5.40454, -3.41225], [5.4223, -3.40163]] , [[5.45358, -3.38224], [5.43791, -3.39189], [5.44366, -3.36989]] , [[5.42355, -3.34547], [5.43493, -3.35629], [5.40737, -3.3301]] , [[5.37246, -3.30205], [5.38828, -3.31774], [5.36688, -3.29652]] , [[5.36557, -3.27639], [5.3633, -3.28377], [5.37798, -3.236]] , [[5.40686, -3.15654], [5.39071, -3.19552], [5.41922, -3.12671]] , [[5.39939, -3.07494], [5.41896, -3.10178], [5.38983, -3.06183]] , [[5.38278, -3.02707], [5.38695, -3.04348], [5.38175, -3.02304]] , [[5.39051, -3.01306], [5.38643, -3.01502], [5.39357, -3.01159]] , [[5.40474, -3.01916], [5.39984, -3.01697], [5.42239, -3.02703]] , [[5.44321, -3.05827], [5.42031, -3.0542], [5.44628, -3.05881]] , [[5.44834, -3.08141], [5.44828, -3.07333], [5.44855, -3.10462]] , [[5.44643, -3.15099], [5.44951, -3.12811], [5.44281, -3.17779]] , [[5.42884, -3.23034], [5.43762, -3.20486], [5.41617, -3.2671]] , [[5.44691, -3.29552], [5.41635, -3.2716], [5.46965, -3.3133]] , [[5.5132, -3.35123], [5.49221, -3.33148], [5.52252, -3.36]] , [[5.53505, -3.38408], [5.5272, -3.37351], [5.54172, -3.39305]] , [[5.56254, -3.39942], [5.54564, -3.41071], [5.57788, -3.38919]] , [[5.5705, -3.3583], [5.58133, -3.37442], [5.56635, -3.35211]] , [[5.55657, -3.34078], [5.55993, -3.34728], [5.55245, -3.33281]]
            ]), is_closed=True,  show_control=True)

        test0.apply(Art.Scale(15))
        test0.apply(Art.Translate([-15, 30]))
        test1.apply(Art.Scale(15))
        test1.apply(Art.Translate([-70, 55]))

        return test0, test1

    if index == 2:
        tree1 = Art.PieceWiseBezier(np.array([
            [[0.627062, -0.462362], [0.834525, -0.435596], [0.538872, -0.47374]],
            [[0.390052, -0.572611], [0.450551, -0.507443], [0.329553, -0.637778]],
            [[0.339157, -0.819163], [0.302703, -0.738058], [0.375612, -0.900267]],
            [[0.560963, -0.904284], [0.482875, -0.946818], [0.47604, -0.992053]],
            [[0.377534, -1.21595], [0.403403, -1.09659], [0.351666, -1.3353]],
            [[0.46967, -1.55357], [0.379638, -1.47105], [0.559702, -1.63609]],
            [[0.801273, -1.56006], [0.715077, -1.64658], [0.800128, -1.66212]],
            [[0.706292, -1.84866], [0.747106, -1.75511], [0.665478, -1.94222]],
            [[0.682308, -2.14388], [0.636561, -2.05263], [0.725808, -2.23064]],
            [[0.923116, -2.28445], [0.82651, -2.27508], [1.01972, -2.29382]],
            [[1.21353, -2.27015], [1.1166, -2.27515], [1.31046, -2.26515]],
            [[1.49078, -2.33578], [1.41417, -2.27618], [1.59482, -2.41673]],
            [[1.62944, -2.69329], [1.62003, -2.5618], [1.64913, -2.96817]],
            [[1.58204, -3.5164], [1.63316, -3.24559], [1.57014, -3.57944]],
            [[1.52221, -3.69816], [1.55581, -3.6435], [1.43179, -3.84526]],
            [[1.06457, -3.9093], [1.23693, -3.88007], [1.0633, -3.9093]],
            [[1.06196, -3.91959], [1.06203, -3.9093], [1.55606, -3.86208]],
            [[2.5509, -3.92019], [2.05684, -3.86229], [2.46474, -3.92344]],
            [[2.32805, -3.80767], [2.38306, -3.87406], [2.27304, -3.74128]],
            [[2.21265, -3.5778], [2.24071, -3.65933], [2.07515, -3.17841]],
            [[2.04914, -2.32982], [2.01917, -2.75115], [2.28377, -2.34714]],
            [[2.59007, -1.96213], [2.51984, -2.18667], [2.66031, -1.73759]],
            [[2.35507, -1.35174], [2.55775, -1.47121], [2.54253, -1.34132]],
            [[2.72284, -0.988809], [2.70622, -1.17583], [2.73946, -0.80179]],
            [[2.43711, -0.549872], [2.61362, -0.613861], [2.26059, -0.485884]],
            [[1.92474, -0.686397], [2.0488, -0.545469], [1.9595, -0.615909]],
            [[1.88061, -0.469225], [1.93428, -0.526644], [1.82694, -0.411806]],
            [[1.67402, -0.362737], [1.7504, -0.381276], [1.55532, -0.333925]],
            [[1.31523, -0.383199], [1.42569, -0.33106], [1.20476, -0.435338]],
            [[1.12913, -0.673089], [1.1195, -0.551319], [1.04829, -0.509503]]
        ]), is_closed=True)

        tree2 = Art.PieceWiseBezier(np.array([
            [[1.74706, -0.742362], [1.95452, -0.715596], [1.65887, -0.75374]],
            [[1.17005, -0.852611], [1.23055, -0.787443], [1.10955, -0.917778]],
            [[1.60916, -1.05916], [1.5727, -0.978058], [1.64561, -1.14027]],
            [[0.960963, -1.20428], [0.882875, -1.24682], [0.87604, -1.29205]],
            [[1.47753, -1.45595], [1.5034, -1.33659], [1.45167, -1.5753]],
            [[0.80967, -1.58357], [0.719638, -1.50105], [0.899702, -1.66609]],
            [[1.47127, -1.76006], [1.38508, -1.84658], [1.47013, -1.86212]],
            [[0.746292, -1.95866], [0.787106, -1.86511], [0.705478, -2.05222]],
            [[1.53231, -2.06388], [1.48656, -1.97263], [1.57581, -2.15064]],
            [[0.923116, -2.28445], [0.82651, -2.27508], [1.01972, -2.29382]],
            [[1.21353, -2.27015], [1.1166, -2.27515], [1.31046, -2.26515]],
            [[1.49078, -2.33578], [1.41417, -2.27618], [1.59482, -2.41673]],
            [[1.62944, -2.69329], [1.62003, -2.5618], [1.64913, -2.96817]],
            [[1.58204, -3.5164], [1.63316, -3.24559], [1.57014, -3.57944]],
            [[1.52221, -3.69816], [1.55581, -3.6435], [1.43179, -3.84526]],
            [[1.06457, -3.9093], [1.23693, -3.88007], [1.0633, -3.9093]],
            [[1.06196, -3.91959], [1.06203, -3.9093], [1.55606, -3.86208]],
            [[2.5509, -3.92019], [2.05684, -3.86229], [2.46474, -3.92344]],
            [[2.32805, -3.80767], [2.38306, -3.87406], [2.27304, -3.74128]],
            [[2.21265, -3.5778], [2.24071, -3.65933], [2.07515, -3.17841]],
            [[2.04914, -2.32982], [2.01917, -2.75115], [2.28377, -2.34714]],
            [[2.90007, -2.17213], [2.82984, -2.39667], [2.97031, -1.94759]],
            [[2.09507, -1.84174], [2.29775, -1.96121], [2.28253, -1.83132]],
            [[2.88, -1.58881], [2.86338, -1.77583], [2.89662, -1.40179]],
            [[1.96711, -1.53987], [2.14362, -1.60386], [1.79059, -1.47588]],
            [[2.79474, -1.3064], [2.29214, -1.39674], [2.8295, -1.23591]],
            [[2.16061, -0.959225], [2.21428, -1.01664], [2.10694, -0.901806]],
            [[2.57402, -0.592737], [2.6504, -0.611276], [2.45532, -0.563925]],
            [[1.81523, -0.05], [1.92569, 0.00213888], [1.70476, -0.102139]],
            [[1.22913, -0.533089], [1.2195, -0.411319], [1.55577, -0.708871]]
        ]), is_closed=True)
        tree1.apply(Art.Translate([-3.5, 1]))
        tree2.apply(Art.Translate([0, 3]))
        tree1.apply(Art.Scale(5))
        tree2.apply(Art.Scale(5))
        return tree1, tree2

    if index == 3:
        dolphine = Art.PieceWiseBezier(np.array([
            [[6.69803, -1.7079], [6.03033, -2.34665], [7.28402, -1.1473]],
            [[9.24782, -0.285665], [8.01161, -0.373634], [10.7228, -0.180703]],
            [[11.6864, -0.458731], [10.8592, -0.692689], [12.5136, -0.224774]],
            [[13.5916, 0.62045], [12.0475, 0.508976], [15.1357, 0.731925]],
            [[13.3276, -0.542757], [13.1778, 0.0190007], [13.4775, -1.10452]],
            [[14.5515, -1.35132], [13.6211, -1.11129], [16.0777, -1.7451]],
            [[16.1475, -3.19106], [16.1475, -3.19106], [16.1475, -3.19106]],
            [[17.0608, -2.94747], [16.1988, -2.58347], [17.7072, -3.22047]],
            [[17.7254, -3.68425], [17.7254, -3.68425], [17.7254, -3.68425]],
            [[16.816, -3.61757], [16.8543, -3.59855], [16.7778, -3.63658]],
            [[16.9113, -4.49283], [16.9113, -4.49283], [16.9113, -4.49283]],
            [[15.8806, -3.5789], [16.2507, -4.23361], [15.5105, -2.92418]],
            [[14.3937, -2.50585], [15.1803, -2.57651], [13.6071, -2.4352]],
            [[11.0873, -2.87188], [11.3945, -2.72242], [10.78, -3.02134]],
            [[11.94, -4.23058], [11.2025, -3.8672], [12.6776, -4.59395]],
            [[12.1241, -4.84441], [12.7143, -4.78517], [11.5338, -4.90364]],
            [[9.14978, -3.12275], [9.51345, -3.85899], [9.14978, -3.12275]],
            [[6.99059, -2.89243], [7.87954, -3.01054], [6.07123, -2.77028]],
            [[5.57172, -2.786], [6.14597, -2.81617], [4.89868, -2.75066]],
            [[5.07357, -2.34911], [4.78022, -2.50921], [5.65771, -2.0303]]
        ]), is_closed=True, show_control=False)

        shark = Art.PieceWiseBezier(np.array([
            [[3.18653, -7.7979], [2.51663, -8.14829], [3.55991, -7.60261]],
            [[4.97359, -7.26631], [4.20085, -7.3226], [5.34176, -7.2395]],
            [[6.14438, -7.02566], [6.0331, -7.22105], [6.55277, -6.3086]],
            [[7.86285, -5.88481], [7.86285, -5.88481], [7.86285, -5.88481]],
            [[7.68371, -7.22039], [7.30413, -6.7215], [7.98068, -7.6107]],
            [[9.67019, -7.8402], [8.84569, -7.6699], [10.144, -7.93806]],
            [[11.3241, -7.94505], [11.1132, -8.22488], [11.902, -7.17828]],
            [[12.6761, -7.04223], [12.6761, -7.04223], [12.6761, -7.04223]],
            [[12.0003, -8.16643], [12.0003, -8.16643], [12.0003, -8.16643]],
            [[12.6919, -9.35143], [12.4774, -8.80204], [12.8302, -9.70582]],
            [[11.4742, -8.6448], [11.4742, -8.6448], [11.4742, -8.6448]],
            [[9.91189, -8.93885], [9.91189, -8.93885], [9.91189, -8.93885]],
            [[10.3736, -9.75698], [10.3736, -9.75698], [10.3736, -9.75698]],
            [[9.05765, -8.93606], [9.42766, -9.39996], [8.87717, -8.70979]],
            [[6.16403, -9.25355], [6.01899, -8.55394], [6.23488, -9.59529]],
            [[6.96231, -10.4106], [6.96231, -10.4106], [6.96231, -10.4106]],
            [[4.73407, -9.2391], [5.29222, -9.96773], [4.73407, -9.2391]],
            [[3.89514, -9.23964], [4.68024, -9.28627], [3.5161, -9.21713]],
            [[2.6493, -9.08317], [3.14582, -9.16858], [2.31766, -9.02613]],
            [[1.92412, -8.69059], [1.88786, -8.96997], [1.9408, -8.56206]]
        ]), is_closed=True, show_control=False)
        shark.apply(Art.Translate([0,10]))
        dolphine.apply(Art.Translate([-20, 5]))

        return shark, dolphine
    else:
        print("Level 0 test case of index {} is not available. Choose from : {}".format(index, ', '.join(
            [str(i) for i in range(4)])))
        return None


def main_test(test):
    d = Art.Draw()
    d.add_art(test[0])
    d.add_art(test[1])
    d.draw()


# main_test(get_test(3))