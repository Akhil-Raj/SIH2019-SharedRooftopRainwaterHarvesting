from django.http import HttpResponse
from django.shortcuts import render
from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests
import random as rand
from .clustering import Clustering
from .point import Point
import csv
from .opencvPrac import canny2
import cv2
from django.views.decorators.csrf import csrf_exempt

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = 'jsons'
api_key='AIzaSyD4ll-YMfrc8zWti_WR1V9pT89DohYBTMQ'
def index(request):
	return render(request,"index.html")

def make_image_data_list(imgname):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    makedirs(RESULTS_DIR, exist_ok=True)
    img_requests = []
    ctxt = b64encode(imgname.read()).decode()
    img_requests.append({
        'image': {'content': ctxt},
        'features': [{
            'type': 'LABEL_DETECTION',
        }]
    })
    return img_requests

def make_image_data(image_filename):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filename)
    return json.dumps({"requests": imgdict }).encode()


def request_img_recog(image_filename):
    response = requests.post(ENDPOINT_URL,
                             data=make_image_data(image_filename),
                             params={'key': api_key},
                             headers={'Content-Type': 'application/json'})
    print (response)
    return response


def catchmentArea(request):
	image_filename=request.FILES['file_submit']
	response = request_img_recog(image_filename)
	if response.status_code != 200 or response.json().get('error'):
   	    return HttpResponse()
	else:
		for idx, resp in enumerate(response.json()['responses']):
            # save to JSON file
			imgname = image_filename.name
			jpath = join(RESULTS_DIR, basename(imgname) + '.json')
			datatxt = json.dumps(resp, indent=2)
			print (datatxt)
		return HttpResponse(json.dumps(datatxt))

@csrf_exempt
def canny(request):
	filename= request.GET.get('filename')	
	geo_locs = []
	#loc_ = Point(0.0, 0.0)  #tuples for location
	#geo_locs.append(loc_)
	#read the fountains location from the csv input file and store each fountain location as a Point(latit,longit) object
	#f = open('/home/kazem/Downloads/Hackathon/drinkingFountains.csv', 'r')
	#reader = csv.reader(f, delimiter=",")
	reader = canny2(filename)

	for line in reader:
	    #print(line)
	    loc_ = Point(float(line[0]), float(line[1]))  #tuples for location
	    geo_locs.append(loc_)
	#print len(geo_locs)
	#for p in geo_locs:
	#    print "%f %f" % (p.latit, p.longit)
	#let's run k_means clustering. the second parameter is the no of clusters
	cluster = Clustering(geo_locs, 6)
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

	img = cv2.imread(filename)
	for i in range(6):
	    cv2.circle(img, (means[i][0], means[i][1]), 5, (255, 0, 0), -1)
	cv2.imwrite("static/imageToShow2.png", img)
	return HttpResponse()

