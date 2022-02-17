from queue import PriorityQueue
from random import randint, uniform
from networkx.algorithms.assortativity.pairs import node_degree_xy
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.graph import Graph
from networkx.readwrite.graphml import GraphMLWriterLxml
import numpy as np
import sys
import networkx as nx
from matplotlib import animation, rc
import matplotlib.pyplot as plt
rc('animation', html='html5')
from tkinter import Tk    
from tkinter.filedialog import askopenfilename

# Prim's Algorithm in Python


# INF = 9999999
# # number of vertices in graph
# V = 5
# # create a 2d array of size 5x5
# # for adjacency matrix to represent graph
# G = [[0, 9, 75, 0, 0],
#      [9, 0, 95, 19, 42],
#      [75, 95, 0, 51, 66],
#      [0, 19, 51, 0, 31],
#      [0, 42, 66, 31, 0]] 
# # create a array to track selected vertex
# # selected will become true otherwise false
# selected = [0, 0, 0, 0, 0]
# # set number of edge to 0
# no_edge = 0
# # the number of egde in minimum spanning tree will be
# # always less than(V - 1), where V is number of vertices in
# # graph
# # choose 0th vertex and make it true
# selected[0] = True
# # print for edge and weight
# print("Edge : Weight\n")
# while (no_edge < V - 1):
#     # For every vertex in the set S, find the all adjacent vertices
#     #, calculate the distance from the vertex selected at step 1.
#     # if the vertex is already in the set S, discard it otherwise
#     # choose another vertex nearest to selected vertex  at step 1.
#     # minimum = INF
#     x = 0
#     y = 0
#     for i in range(V):
#         if selected[i]:
#             for j in range(V):
#                 if ((not selected[j]) and G[i][j]):  
#                     # not in selected and there is an edge
#                     if minimum > G[i][j]:
#                         minimum = G[i][j]
#                         x = i
#                         y = j
#     print(str(x) + "-" + str(y) + ":" + str(G[x][y]))
#     selected[y] = True
#     no_edge += 1

Tk().withdraw() 
filename = askopenfilename() 
# print(filename)

nodes_data = []
edges_data = []
nodes=0
start=0
temp_1=[]
temp_2=[]
f_read = open(filename, "r") #to change file 
str = f_read.read()
f_read.close()
str = str.replace("NETSIM", "")
line_sep = str.split("\n")
no_line= [line.strip() for line in line_sep if line.strip() != ""]
str_no_line = ""
for line in no_line:
    str_no_line += line + "\n"
str = str_no_line
temp = str.splitlines()
temp = [i.split("\t") for i in temp]  
nodes = int(temp[0][0])
start = int(temp[-1][0])
temp.pop(0)
temp.pop()
for j in range(nodes):
    temp_1 = [float(i) for i in temp[0]]
    nodes_data.append(tuple(temp_1[0:3]))
    temp.pop(0)
temp_1.clear()
for i in range(len(temp)):  
    for j in range(2, len(temp[i]), 2):
        temp_1.append(temp[i][j])
    for j in range(len(temp_1)):
        for k in range(1, len(temp[i])):
            if temp[i][k] == temp_1[j]:
                del temp[i][k]
                break
    temp_1.clear()
l=0
for i in temp:
    length=int(i.pop(0))
    for j in range(length):
        edges_data.append((int(nodes_data[l][0]),int(i[0]),float(i[1])))
        # matrix[node]
        i.pop(0)
        i.pop(0)
    l=l+1
for i in range(len(edges_data)):
    if edges_data[i][0]==edges_data[i][1]:
        continue
    temp_2.append((edges_data[i][0],edges_data[i][1],int(edges_data[i][2])))
edges_data.clear()
for i in temp_2:
    edges_data.append(i)
temp_2.clear()

print(start)
print(nodes)
print(nodes_data)
print(edges_data)
print(len(edges_data))

i =0
# print(edges_data[1][0],edges_data[i][1],edges_data[i][2])
#############3
def plot(g):
    pos = nx.get_node_attributes(g, 'pos')
    weight = nx.get_edge_attributes(g, 'weight')
    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="b")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight, font_size=10, font_family="sans-serif")
    plt.show()
##################33




graph = nx.Graph()
for i in range(nodes):
    graph.add_node(int(nodes_data[i][0]),pos=(nodes_data[i][1],nodes_data[i][2]))  

for i in range(len(edges_data)):
    # graph.add_weighted_edges_from(edges_data)
    graph.add_edge(int(edges_data[i][0]),int(edges_data[i][1]),weight=(int((edges_data[i][2])/10000000)))
label = nx.get_edge_attributes(graph,'weight')   

pos = nx.get_node_attributes(graph,'pos')   
all_edges = set(
    tuple(sorted((n1, n2))) for n1, n2 in graph.edges()
)
# plot(graph)

plt.show()
print(all_edges)
edges_in_mst = set()
nodes_on_mst = set()
total = 0
fig, ax = plt.subplots(figsize=(10,8))

def prims():
    pqueue = PriorityQueue()
    # Start at any random node and add all edges connected to this
    # node to the priority queue.
    # start_node = start
    x=0
    for neighbor in graph.neighbors(start):
        edge_data = graph.get_edge_data(start, neighbor)
        edge_weight = edge_data["weight"]

        # total = total + edge_data["weight"]
        pqueue.put((edge_weight, (start, neighbor)))
   
    # Loop until all nodes are in the MST
    while len(nodes_on_mst) < nodes:
        # Get the edge with smallest weight from the priority queue
        _, edge = pqueue.get(pqueue)

        if edge[0] not in nodes_on_mst:
            new_node = edge[0]
        elif edge[1] not in nodes_on_mst:
            new_node = edge[1]
        else:
            # If this edge connects two nodes that are already in the
            # MST, then skip this and continue to the next edge in
            # the priority queue.
            continue

        # Every time a new node is added to the priority queue, add
        # all edges that it sits on to the priority queue.
        for neighbor in graph.neighbors(new_node):
            edge_data = graph.get_edge_data(new_node, neighbor)
            edge_weight = edge_data["weight"]
            pqueue.put((edge_weight, (new_node, neighbor)))

        # Add this edge to the MST.
        edges_in_mst.add(tuple(sorted(edge)))
        nodes_on_mst.add(new_node)
        # Yield edges in the MST to plot.
        yield edges_in_mst
        # return edges_in_mst
        # plt.show(nx.draw_networkx_nodes(edges_in_mst))  


def update(mst_edges):
    ax.clear()
    nx.draw_networkx_nodes(graph, pos,  node_size=200, ax=ax,node_color="tab:Red")
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=label)
    nx.draw_networkx_edges(
        graph, pos, edgelist=all_edges-mst_edges, alpha=0.1,
        edge_color='#000080', width=2, ax=ax
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=mst_edges, alpha=1.0,
        edge_color='#FFBF00', width=2, ax=ax
    )
def do_nothing():
    # FuncAnimation requires an initialization function. We don't
    # do any initialization, so we provide a no-op function.
    pass

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=do_nothing,
    frames=prims,
    interval=500,
)

ani
# plt.show()
plt.show()
# def find(parent, i):
#     if parent[i] == i:
#         return i
#     # print(parent,parent[i])    
#     return find(parent, parent[i])


# def union_kruskal(parent,rank,x, y):
#     xroot = find(parent, x)
#     yroot = find(parent, y)
 
#         # Attach smaller rank tree under root of
#         # high rank tree (Union by Rank)
#     if rank[xroot] < rank[yroot]:
#         parent[xroot] = yroot
#     elif rank[xroot] > rank[yroot]:
#         parent[yroot] = xroot
 
#     # If ranks are same, then make one as root
#     # and increment its rank by one
#     else:
#         parent[yroot] = xroot
#         rank[xroot] += 1
 

# # def Kruskal():
    
# #     mincost = 0  # Cost of min MST
# #     graph = sorted(edges_data,
# #                             key=lambda item: item[2])
# #     parent = []
# #     rank = []
#     mst = nx.Graph()
#     for i in range(nodes):
#         mst.add_node(i,pos=pos[i])
    


#     # Initialize sets of disjoint sets
#     for node in range(nodes):
#             parent.append(node)
#             rank.append(0)

    # Include minimum weight edges one by one
#     edge_count = 0
#     i = 0
#     e = 0
#     while e < nodes - 1:
     
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
