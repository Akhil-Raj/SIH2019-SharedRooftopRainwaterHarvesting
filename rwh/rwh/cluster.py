import random as rand
from clustering import clustering
from point import Point
import csv
from opencvPrac import canny2
import cv2

geo_locs = []
#loc_ = Point(0.0, 0.0)  #tuples for location
#geo_locs.append(loc_)
#read the fountains location from the csv input file and store each fountain location as a Point(latit,longit) object
#f = open('/home/kazem/Downloads/Hackathon/drinkingFountains.csv', 'r')
#reader = csv.reader(f, delimiter=",")
reader = canny2()

for line in reader:
    #print(line)
    loc_ = Point(float(line[0]), float(line[1]))  #tuples for location
    geo_locs.append(loc_)
#print len(geo_locs)
#for p in geo_locs:
#    print "%f %f" % (p.latit, p.longit)
#let's run k_means clustering. the second parameter is the no of clusters
cluster = clustering(geo_locs, 6)
flag = cluster.k_means(False)
if flag == -1:
    print("Error in arguments!")
else:
    #the clustering results is a list of lists where each list represents one cluster
    print("clustering results:")
    #cluster.print_clusters(cluster.clusters)
print(cluster.clusters)
means = []
for i in cluster.clusters:
    num = 0
    tempx = 0
    tempy = 0
    for j in cluster.clusters[i]:
        num += 1
        tempx += j.latit
        tempy += j.longit
    means += [[int(tempx / num), int(tempy / num)]]

for i in range(6):
    print("mean " + str(i) + ": " + str(means[i]))

img = cv2.imread("sat.png")
for i in range(6):
    cv2.circle(img, (means[i][0], means[i][1]), 5, (255, 0, 0), -1)
cv2.imwrite("imageToShow2.png", img)
#cv2.imshow("img", img)
#cv2.waitKey(0)