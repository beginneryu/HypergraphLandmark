# Paper
the  implement of paper "Hypergraph-driven Landmark Detection Foundation Model on Echocardiography for Cardiac Function Quantification"
![image](https://github.com/beginneryu/HypergraphLandmark/blob/main/method.jpg)

# Innovative
1.We propose an innovative hypergraph-based landmark detection foundation model for echocardiograms which was trained on large scale echocardiography datasets. This method directly quantifies cardiac function using extracted landmark infor- mation while providing interpretability.
2.We developed a hypergraph-based feature extraction backbone that integrates a hypergraph dynamic system with an adaptive hypergraph structure to capture higher-order relationships within images.  By leveraging hypergraph dynamic systems, it offers enhanced controllability and stability while enabling the cap- ture of long-range correlations between vertices.
3.We introduce a Bidirectional Hypergraph Spatio-Temporal (BHST) decoding module.  The hypergraph-based spatial awareness captures spatial relationships between landmarks. Additionally, bidirectional temporal perception is incorpo- rated, enabling landmark predictions in the current frame to be informed by both previous and subsequent frames, thus improving accuracy and consistency.
4.Our method has been trained and evaluated on large scale datasets (more than 10000 patients). The results demonstrate that our method not only outperforms other approaches in landmark detection but also achieves accurate cardiac func- tion quantification, closely aligning with the results from human experts and commercial tools.

# Dataset

EchoNet-Dynamic dataset https://echonet.github.io/dynamic/
CAMUS dataset https://www.creatis.insa-lyon.fr/Challenge/camus/



# Data organization
The dataset we provide is read from data stored in the following way.

Images: Data/camus_4ch/Image/patient0001/000.png

Here, "Data" is an arbitrary root directory, "camus_4ch" is the name of the dataset, "Image" indicates that images are stored in this folder, "patient000X" represents the X - th sample, and each sample stores 10 images, numbered from 000 to 009.

Points: Data/camus_4ch/Points/46pts/adjmatrix_top5.txt

Here, "Data" is an arbitrary root directory, "camus_4ch" is the name of the dataset, "Points" indicates that points and adjacency matrices are stored in this folder, including: "adjmatrix_topk.txt", where k is 3, 5, 7, 13, 21, 33, 46. In addition, there is also "point.txt", which stores 46 densely - sampled points. 

# run
bash script/run.sh
Some parameters can be set in run.sh, such as selecting the backbone network and the predictor, setting the GPU number for the run, and so on. 
