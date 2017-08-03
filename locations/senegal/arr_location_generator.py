import csv

headers=["site_id","arr_id","lon","lat"]
dictLocation = {}
with open('SITE_ARR_LONLAT.CSV', 'rb') as csvfile:
        mreader = csv.reader(csvfile, delimiter=',')
	for row in mreader:
	    arr_id = int(row[headers.index("arr_id")])
	    lon = float(row[headers.index("lon")])
	    lat = float(row[headers.index("lat")])
	    if (arr_id  in  dictLocation.keys()):
		dictLocation[arr_id][0]+=lon
		dictLocation[arr_id][1]+=lat
		dictLocation[arr_id][2]+=1
	    else:
		dictLocation[arr_id]=[lon,lat,1]

print "arr_id,lon,lat"
for key in dictLocation:
    slon = str(float(dictLocation[key][0])/float(dictLocation[key][2]))
    slat = str(float(dictLocation[key][1])/float(dictLocation[key][2]))
    print "{},{},{}".format(str(key),slon,slat)
