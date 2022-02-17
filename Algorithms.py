import inspect
import networkx as nx
import matplotlib.pyplot as plt
import sys
from networkx.algorithms import tree
import tkinter
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from networkx.readwrite.json_graph import adjacency
from matplotlib import animation, rc
import inspect
import networkx as nx
import matplotlib.pyplot as plt
import sys
from networkx.algorithms import tree
import tkinter
from queue import PriorityQueue
from random import randint, uniform
from networkx.algorithms.assortativity.pairs import node_degree_xy
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.graph import Graph
from networkx.readwrite.graphml import GraphMLWriterLxml
rc('animation', html='html5')
from tkinter import Tk    
from tkinter.filedialog import askopenfilename

def minimumDistance(dist, mstSet, V):
    minnum = sys.maxsize
    for i in range(V):
        if dist[i] < minnum and mstSet[i] == False:
            minnum = dist[i]
            min_index = i
    return min_index

##########################Prims ###########################################
def findMaxVertex(visited, weights,V):
    index = -1;
    maxW = sys.maxsize;
    for i in range(V):
        if (visited[i] == False and weights[i] < maxW):
            maxW = weights[i];
            index = i;
    return index;

def prims(graph, V,S):

    visited = [True] * V;
    weights = [0] * V;
    parent = [0] * V;

    for i in range(V):
        visited[i] = False;
        weights[i] = sys.maxsize;

    weights[S] = 0;
    parent[S] = -1;

    for i in range(V - 1):
        maxVertexIndex = findMaxVertex(visited, weights,V);
        visited[maxVertexIndex] = True;
        for j in range(V):
            if (graph[j][maxVertexIndex] != 0 and visited[j] == False):
                if (graph[j][maxVertexIndex] < weights[j]):
                    weights[j] = graph[j][maxVertexIndex];
                    parent[j] = maxVertexIndex;
    mst=0
    for i in range(V):
            for j in range(V):
                graph[i][j]=0
                if(parent[j]==i):
                    graph[i][j]=weights[j]
                    mst=mst+weights[j]
    mst=mst/10000000
    print("MST Cost",mst)
    return graph
##########################Prims End###########################################



##########################Kruaskal###########################################

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# def find(parent, i):
#     if parent[i] == i:
#         return i
#     # print(parent,parent[i])    
#     return find(parent, parent[i])



#             # Step 2: Pick the smallest edge and increment
#             # the index for next iteration
#         u, v, w = graph[i]
#         i = i + 1
#         print(u,v)
#         x = find(parent, u)
#         y = find(parent, v)
 
#             # If including this edge does't
#             #  cause cycle, include it in result
#             #  and increment the indexof result
#             # for next edge
#         if x != y:
#             e = e + 1
#             mst.add_edge(u,v,weight=float(w/1000000))
#             print("y")
#             union_kruskal(parent, rank, x, y)
#         # Else discard the edge
 
#         minimumCost = 0
#         print ("Edges in the constructed MST")
#         # for u, v, weight in result:
#         #     minimumCost += weight
#         #     print("%d -- %d == %d" % (u, v, weight))
#         # print("Minimum Spanning Tree" , minimumCost)
#     plot(mst)


# print("kruskal")
# Kruskal()


# def Kruskal():
#     print("-----------------")
#     print(nodes)
#     print(graph)

#     mincost = 0  # Cost of min MST
#     parent = [0]*nodes
#     G = [[0] * nodes for _ in range(nodes)]
#     # Initialize sets of disjoint sets
#     for i in range(nodes):
#         parent[i] = i
#         # print(parent[i])
#     # Include minimum weight edges one by one
#     edge_count = 0
#     while edge_count < nodes - 1:
#         min = sys.maxsize
#         a = -1
#         b = -1
#         for i in range(nodes):
#             for j in range(nodes):
#                 if find(parent, i) != find(parent, j) and graph[i][j] < min and graph[i][j]!=0:
#                     min = graph[i][j]
#                     a = i
#                     b = j
#         union_kruskal(parent,a, b)
#         print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, min))
#         G[a][b] = min
#         edge_count += 1
#         mincost += min

#     print("Minimum cost= {}".format(mincost))
#     return G 

# Kruskal()



# Kruskal();  
# # def update(mst_edges):
# #     ax.clear()
# #     nx.draw_networkx_nodes(graph, pos,  node_size=200, ax=ax,node_color="tab:Red")
# #     nx.draw_networkx_edge_labels(graph,pos,edge_labels=label)
# #     nx.draw_networkx_edges(
# #         graph, pos, edgelist=all_edges-mst_edges, alpha=0.1,
# #         edge_color='#000080', width=2, ax=ax
# #     )
# #     nx.draw_networkx_edges(
# #         graph, pos, edgelist=mst_edges, alpha=1.0,
# #         edge_color='#FFBF00', width=2, ax=ax
# #     )
# # def do_nothing():
# #     # FuncAnimation requires an initialization function. We don't
# #     # do any initialization, so we provide a no-op function.
# #     pass

# # ani = animation.FuncAnimation(
# #     fig,
# #     update,
# #     init_func=do_nothing,
# #     frames=Kruskal,
# #     interval=500,
# # )

# # ani
# # plt.show()   

def union_kruskal(parent,i, j):
    a = find(parent, i)
    b = find(parent, j)
    parent[a] = b


def Kruskal(graph, V):

    mincost = 0  # Cost of min MST
    parent = [0]*V
    G = [[0] * V for _ in range(V)]
    # Initialize sets of disjoint sets
    for i in range(V):
        parent[i] = i

    # Include minimum weight edges one by one
    edge_count = 0
    while edge_count < V - 1:
        min = sys.maxsize
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(parent, i) != find(parent, j) and graph[i][j] < min and graph[i][j]!=0:
                    min = graph[i][j]
                    a = i
                    b = j
        union_kruskal(parent,a, b)
        print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, min))
        G[a][b] = min
        edge_count += 1
        mincost += min

    print("Minimum cost= {}".format(mincost))
    return G



##############################Kruskal End##############################33


def plot(g):
    pos = nx.get_node_attributes(g, 'pos')
    weight = nx.get_edge_attributes(g, 'weight')
    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="b")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight, font_size=10, font_family="sans-serif")
    plt.show()

def plot_AdjacentcyMatrix(Matrix,V,positions):
    
    G = nx.Graph()
    


    for i in range(len(positions)):
        G.add_node(i,pos=positions[i])

    for i in range(V):
        for j in range(V):
            if Matrix[i][j] == 0:
                continue

            G.add_edge(i,j,weight=Matrix[i][j])

    return G


# ------------------------------------------------Dijkstra Algorithm--------------------------------------------
def minDistance(V, dist, sptSet):
    min = sys.maxsize
    min_index = -1

    for u in range(V):
        if dist[u] < min and sptSet[u] == False:
            min = dist[u]
            min_index = u

    return min_index

def dijkstra(G, V, src):
    dist = [sys.maxsize] * V
    sptSet = [False] * V
    parent = [-1] * V
    dist[src] = 0
    parent[src] = src

    for _ in range(V):

        x = minDistance(V, dist, sptSet)
        sptSet[x] = True

        for y in range(0, V):

            if (G[x][y] > 0 and sptSet[y] == False) and (dist[y] > dist[x] + G[x][y]):
                dist[y] = dist[x] + G[x][y]
                parent[y] = x

    for i in range(V):
        for j in range(V):
            G[i][j] = 0

    for i in range(V):
        if parent[i]!=-1:
            G[parent[i]][i] = dist[i]
    # print(G)
    return G
# ------------------------------------------Dijkstra Algorithm End--------------------------------------------


# --------------------------------------------BellmanFord---------------------------------------------------------

def BellmanFord(graph, V, src):

    dist = [sys.maxsize] * V
    dist[src] = 0
    parent = [-1] * V
    G = [[0] * V for _ in range(V)]

    for q in range(V - 1):
        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue

                w = graph[i][j]
                if dist[i] + w < dist[j]:
                    dist[j] = dist[i] + w
                    parent[j] = i


    for i in range(V):
        for j in range(V):
            if graph[i][j] == 0:
                continue
            w = graph[i][j]
            if dist[i] != sys.maxsize and dist[i] + w < dist[j]:
                print("Graph contains negative weight cycle")
                return None
    t=0
    print("Vertex Distance from Source")
    for i in range(V):
        print("{0}\t\t{1}".format(i, dist[i]))
        if parent[i]!=-1:
            G[parent[i]][i] = dist[i]
            t=t+dist[i]
    print("Total Distance",t)
    return G
# --------------------------------------------BellmanFord END--------------------------------------------------------


# --------------------------------------------FLOYD WARSHALL---------------------------------------------------------


def floyd_warshall(graph, V):

    G = graph
    INF = sys.maxsize
    for i in range(V):
        for j in range(V):
            if G[i][j] == 0 and i!=j:
                G[i][j] = INF

    # Adding vertices individually
    for k in range(V):
        for i in range(V):
            for j in range(V):
                G[i][j] = min(G[i][j], G[i][k] + G[k][j])
    for i in range(V):
        for j in range(V):
            if G[i][j] == INF:
                G[i][j] = 0

    return G

# --------------------------------------------FLOYD WARSHALL END---------------------------------------------------------


# -----------------------------------------------Boruvka---------------------------------------------------------


# A function that does union of two sets of x and y
# (uses union by rank)
def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    # Attach smaller rank tree under root of high rank tree
    # (Union by Rank)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    # If ranks are same, then make one as root and increment
    # its rank by one
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def boruvka(graph, V):
    parent = []
    rank = []
    G  = [[0] * V for _ in range(V)]
    # An array to store index of the cheapest edge of
    # subset. It store [u,v,w] for each component
    cheapest = []

    # Initially there are V different trees.
    # Finally there will be one tree that will be MST
    numTrees = V
    MSTweight = 0

    # Create V subsets with single elements
    for node in range(V):
        parent.append(node)
        rank.append(0)
        cheapest = [-1] * V

    # Keep combining components (or sets) until all
    # compnentes are not combined into single MST

    while numTrees > 1:

        # Traverse through all edges and update
        # cheapest of every component
        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                # Find components (or sets) of two corners
                # of current edge
                w = graph[i][j]
                set1 = find(parent, i)
                set2 = find(parent, j)

                # If two corners of current edge belong to
                # same set, ignore current edge. Else check if
                # current edge is closer to previous
                # cheapest edges of set1 and set2
                if set1 != set2:

                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [i, j, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [i, j, w]

        # Consider the above picked cheapest edges and add them
        # to MST
        for node in range(V):

            # Check if cheapest for current set exists
            if cheapest[node] != -1:
                u, v, w = cheapest[node]
                set1 = find(parent, u)
                set2 = find(parent, v)

                if set1 != set2:
                    MSTweight += w
                    union(parent, rank, set1, set2)
                    G[u][v] = w
                    print("Edge %d-%d with weight %d included in MST" % (u, v, w))
                    numTrees = numTrees - 1

        # reset cheapest array
        cheapest = [-1] * V

    print("Weight of MST is %d" % MSTweight)
    return G



# -----------------------------------------------Boruvka End---------------------------------------------------------


# -----------------------------------------------Clustering---------------------------------------------------------
def Clustering(G):

    Average_Clustering = nx.average_clustering(G)
    print(Average_Clustering)

# -----------------------------------------------Clustering End---------------------------------------------------------


def filing(Input):
    if Input == "Input 10":
        filename = 'input10.txt'
    elif Input == "Input 20":
        filename = 'input20.txt'
    elif Input == "Input 30":
        filename = 'input30.txt'
    elif Input == "Input 40":
        filename = 'input40.txt'
    elif Input == "Input 50":
        filename = 'input50.txt'
    elif Input == "Input 60":
        filename = 'input60.txt'
    elif Input == "Input 70":
        filename = 'input70.txt'
    elif Input == "Input 80":
        filename = 'input80.txt'
    elif Input == "Input 90":
        filename = 'input90.txt'
    elif Input == "Input 100":
        filename = 'input100.txt'

    with open(filename) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)

    count = 0
    list1 = []
    Node = []

    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list1.append(listli)

    v = int(list1[1][0])

    adjacent = [[0] * v for _ in range(v)]


    for i in range(0, v):
        ps = (float(list1[2 + i][1]), float(list1[2 + i][2]))
        Node.append(ps)
        l=0

    for i in range(v + 2, len(list1) - 1):
        noe = int(list1[i][0])
        for j in range(1,len(list1[i]),4):
            t = int(list1[i][j])
            w = round((float(list1[i][j + 2])/10000000),2)
            # we = round(w, 2)
            adjacent[l][t]=w
        l+=1
    source = int(list1[len(list1)-1][0])
    return adjacent,v,Node,source



def Show_Input_File():

    file_name = comboInput.get()
    algorithm_name = comboAlgorithm.get()
    FileNames = ["Input 10", "Input 20", "Input 30", "Input 40", "Input 50", "Input 60", "Input 70", "Input 80",
                 "Input 90", "Input 100"]

    if file_name not in FileNames:
        return

    adjacency_matrix, V, positions, src = filing(file_name)

    G = plot_AdjacentcyMatrix(adjacency_matrix, V, positions)
    plot(G)

def Implement_Algorithm():
    file_name = comboInput.get()
    algorithm_name = comboAlgorithm.get()
    FileNames = ["Input 10", "Input 20", "Input 30", "Input 40", "Input 50", "Input 60", "Input 70", "Input 80",
                 "Input 90", "Input 100"]
    AlgorithmNames = ["Prims Algorithm", "Kruskal Algorithm", "Dijkstra Algorithm", "Bellman Ford Algorithm", "Floyd Warshall Algorithm",
                      "Local Clustering Coefficient", "Boruvka's Algorithm"]

    if (file_name not in FileNames) or (algorithm_name not in AlgorithmNames):
        return

    

    adjacency_matrix, V, positions, src = filing(file_name)

    if algorithm_name =="Prims Algorithm":
        G = prims(adjacency_matrix,V,src)
        g = plot_AdjacentcyMatrix(G, V, positions)
        plot(g)

    elif algorithm_name == "Kruskal Algorithm":
        G = Kruskal(adjacency_matrix,V)
        g = plot_AdjacentcyMatrix(G, V, positions)
        plot(g)

    elif algorithm_name == "Dijkstra Algorithm":
        G = dijkstra(adjacency_matrix,V,3)
        g = plot_AdjacentcyMatrix(G,V,positions)
        plot(g)

    elif algorithm_name == "Bellman Ford Algorithm":
        G = BellmanFord(adjacency_matrix,V,src)
        g = plot_AdjacentcyMatrix(G, V, positions)
        plot(g)

    elif algorithm_name == "Floyd Warshall Algorithm":
        G = floyd_warshall(adjacency_matrix,V)
        g = plot_AdjacentcyMatrix(G, V, positions)
        plot(g)

    elif algorithm_name == "Local Clustering Coefficient":
        G = plot_AdjacentcyMatrix(adjacency_matrix,V,positions)
        Clustering(G)

    elif algorithm_name == "Boruvka's Algorithm":
        G = boruvka(adjacency_matrix, V)
        g = plot_AdjacentcyMatrix(G, V, positions)
        plot(g)



#------------------------------------Start GUI---------------------------------------------------------
window = Tk()
window.title("Project")
window.configure(bg="#000080")


comboInput = Combobox(window, state="readonly", width=30, height=10)
comboInput['values'] = (
"Input 10", "Input 20", "Input 30", "Input 40", "Input 50", "Input 60", "Input 70", "Input 80", "Input 90", "Input 100")
comboInput.grid(row=1, column=0)
comboInput.place(x=200, y=200)
comboInput.set('Select Input File')

comboAlgorithm = Combobox(window, state="readonly", width=30, height=10)
comboAlgorithm['values'] = (
"Prims Algorithm", "Kruskal Algorithm", "Dijkstra Algorithm", "Bellman Ford Algorithm", "Floyd Warshall Algorithm","Boruvka's Algorithm", "Local Clustering Coefficient")
comboAlgorithm.place(x=500, y=200)
comboAlgorithm.set('Select Algorithm')

bt_showInputfile = Button(window, text="Show Orignal Graph", command=Show_Input_File, width=20)
bt_showInputfile.place(x=300, y=250)

bt_Implement = Button(window, text="Implement Algorithm and Show MST Graph", command=Implement_Algorithm, width=20)
bt_Implement.place(x=600, y=250)

window.geometry('1000x500')
window.mainloop()