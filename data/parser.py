from collections import defaultdict
import random
from datetime import datetime
import time
import numpy as np

# SPIREL parser
# In[1]: txt file parsing

#dataset_name = 'gowalla_LA'
#dataset_name = 'taxi_checkin'
#dataset_name = 'foursquare-1M'
#dataset_name = 'taxi_medium'
dataset_name = 'yelp_RAPPOR'

file = open('../data/' + dataset_name + '.txt','r')
print("Open ", dataset_name, ".txt")

countU = defaultdict(lambda: 0)
countI = defaultdict(lambda: 0)

for line in file:
	split = line.split('\t')
	
	user = split[0]
	countU[user] += 1

	item = split[2]
	countI[item] += 1    
	
file.close()

# In[2]: initiailize dataset

file = open('../data/' + dataset_name + '.txt','r')

user_map = dict()
user_num = 0
item_map = dict()
item_num = 0
User = dict()

line_num = 0

for line in file:
	split = line.split('\t')
	
	user = split[0]
	checkin_time = split[1]
	time_stamp = time.mktime(datetime.strptime(checkin_time, '%Y-%m-%d %H:%M:%S').timetuple())
	#time_stamp = checkin_time
	item = split[2]
	
	#if countU[user] < 3 or countI[item] < 5 : continue
	if countU[user] < 3 : continue
	
	if user in user_map : user_id = user_map[user]
	else:
		user_id = user_num
		user_num+=1
		user_map[user] = user_id
		User[user_id] = []
	if item in item_map : item_id = item_map[item]
	else:
		item_id = item_num
		item_num += 1
		item_map[item] = item_id
	
	User[user_id].append([time_stamp, item_id])
	line_num += 1
	
for user_id in User.keys():
	User[user_id].sort(key=lambda x: x[0])

file.close()

# In[3]: dataset partitioning (train/test)

user_train = dict()
user_test = dict()

for user in User:
	user_train[user] = User[user][:-1]	
	user_test[user] = []
	user_test[user].append(User[user][-1])

dataset = [user_train, user_test, user_num, item_num]
np.save('../data/' + dataset_name + '.npy', dataset)

print("# user : ", user_num)
print("# item : ", item_num)
print("# line : ", line_num)
print ("finish")