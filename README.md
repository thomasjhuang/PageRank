# PageRank
The intention of this repo is for educational purposes, and as such I provide detailed explanation of the data passed into the algorithm and how to run the code in this readme. This pagerank algorithm effectively runs on medium size datasets (several thousand links), and follows an interative implementation without random-walking. 

## Basic Information
The pagerank algorithm runs directly from run_pagerank.py using command line arguments. The basic format that this algorithm will process is a two dimentional numpy array. In my test example, I parse data from adj_list.json, which is a dictionary representation of an adjacency list of outgoing links. For example, adj_list['34'] will output a list of ['4','5','12'], meaning node 34 links to nodes 4, 5, and 12. I preformat this data from adj_list.json, so that when the pagerank() function is called, I pass in a square matrix which is ordered such that the index of each row in the 2d matrix corresponds to a node number. Thus, row 34 will correspond to node 34, and index 4, 5, and 12 should have values of 1 in that row (since those are outgoing links). The execution of the algorithm is straightforward, and will iteratively compute values until a convergence according to a preset epsilon. 
