
from collections import defaultdict
from itertools import combinations
import numpy as np
import re
import sys

file_name = input("Enter the name of the input file: ")
file = open(file_name, "r")
lines = file.readlines()


support_input = int(input("Enter the Minimum Support Count: "))
print("Support is set to be "+str(support_input)+"%")

matrix_data = []
ans = float('inf')
for line in lines:
	data = line.strip().split()
	ans = min(len(data) , ans)

for line in lines:
	data = line.strip().split()
	if len(data) > ans:
		last = data.pop()
		penul = data.pop()
		data.append(penul+'-'+last)
	matrix_data.append(data)

for line in matrix_data:
	for i , ele in enumerate(line):
		line[i] = "G" + str(i+1) + "_" + ele

item_freq = defaultdict(int)

for line in matrix_data:
	for i , ele in enumerate(line):
		item_freq[tuple([ele])]+=1

def find_freq_items(item_freq ,support_threshold):
	rows = len(matrix_data)
	freq_list = []
	for ele , value in item_freq.items():
		if (value / rows) >= (support_threshold/ 100):
			freq_list.append(ele)
			global_frequent_itemset_count[ele]+=value

	return freq_list

def get_combinations(frequent_itemset_list):
	new_combinations = []
	sorted_freq_itemset_list=sorted(frequent_itemset_list)
	for i in range(len(sorted_freq_itemset_list)):
		for j in range(i+1, len(sorted_freq_itemset_list)):
			if(sorted_freq_itemset_list[i][:itemset_length-2] == sorted_freq_itemset_list[j][:itemset_length-2]):
				new_combinations.append(tuple(sorted(set(sorted_freq_itemset_list[i]).union(set(sorted_freq_itemset_list[j])))))
	return new_combinations

def get_freq_items_counts(freq_items_candidate):
	item_freq  = defaultdict(int)
	count = 1
	for i , ele in enumerate(freq_items_candidate):
		for j  , eme in enumerate(matrix_data):
			if set(ele) <= set(eme):
				item_freq[ele]+=1
	return item_freq

global_frequent_itemset_count = defaultdict(int)
global_frequent_list = []
current_freq_list = find_freq_items(item_freq ,support_input)
print("number of length-1 frequent itemsets: ", len(current_freq_list))
for ele in current_freq_list:
	global_frequent_list.append(ele)
count = 1
itemset_length = 2
while len(current_freq_list) !=0:

	current_freq_item_candidates = get_combinations(current_freq_list)
	item_freq = get_freq_items_counts(current_freq_item_candidates)
	current_freq_list = find_freq_items(item_freq , support_input)
	print("number of length-{0} frequent itemsets: {1}".format(itemset_length , len(current_freq_list)))
	for ele in current_freq_list:
		global_frequent_list.append(ele)
	count+=1
	itemset_length+=1

print("number of all lengths frequent itemsets" , len(global_frequent_list))

user_confidence = int(input('Enter confidence percentage: '))
rules = set()
head = set()
body = set()
total_rules=0

for ele in global_frequent_list:
	length = 1
	while length <= len(ele):
		for all_comb in set(combinations(ele , length)):
			confidence = global_frequent_itemset_count[ele] / global_frequent_itemset_count[all_comb]

			if confidence >= user_confidence/100:
				if(len(list(set(ele) - set(all_comb)))!=0):
					rule = (",".join(list(all_comb))) + "->" + (",".join(list(set(ele)-set(all_comb))))
					total_rules = total_rules + 1
					rules.add(rule.upper())
					head.add(str(list(all_comb)))
					body.add(str(list(set(ele)-set(all_comb))))
		length = length+1

print("rule count ----" , total_rules)

def union_set(set1 , set2):
	return set(list(set1) + list(set2))


def intersection_set(set1 , set2):
	set_intersection = set1 & set2
	return set_intersection

def get_template_number(template_number):

	if template_number == 1:
		total_rules_obtained = set()
		p1 = (input('Enter First parameter for template 1: '))
		p2 = (input('Enter Second parameter for template 1: '))
		p3 = (input('Enter Third parameter for template 1: ').upper().split(','))

		if p1.upper() not in ('RULE','BODY','HEAD'):
			sys.exit("Invalid Parameter 1")
		for rule in rules:
			head_value = rule.split('->')[0].split(',')
			body_value = rule.split('->')[1].split(',')
			head_part = set(head_value)
			body_part = set(body_value)

			if p1.upper() == 'RULE':
				result_set = head_part.union(body_part)
			elif p1.upper() == 'BODY':
				result_set = body_part
			elif p1.upper() == 'HEAD':
				result_set = head_part
			if p2.upper() == 'ANY' and len(intersection_set(set(p3) , result_set)) > 0:
				total_rules_obtained.add(rule)
			elif p2.upper() == 'NONE' and  len(intersection_set(set(p3) , result_set)) == 0:
				total_rules_obtained.add(rule)
			elif p2.isdigit() and len(intersection_set(set(p3) , result_set)) == int(p2):
				total_rules_obtained.add(rule)
		return total_rules_obtained

	elif template_number == 2:
		total_rules_obtained = set()
		p1 = (input('Enter First parameter for template 2: '))
		p2 = (input('Enter Second parameter for template 2: '))

		#getting the length of param2
		p2_len = int(p2)
		if p1.upper() not in ('RULE','BODY','HEAD'):
			sys.exit("Invalid Parameter 1")
		for r in rules:
			head_value = r.split('->', 1)[0].split(',')
			body_value = r.split('->' , 1)[1].split(',')
			head_part = set(head_value)
			body_part = set(body_value)

			if p1.upper() == 'RULE':
				result_set = union_set(head_part , body_part)
			elif p1.upper() == 'BODY':
				result_set = body_part
			elif p1.upper() == 'HEAD':
				result_set = head_part
			if len(result_set) >= p2_len:
				total_rules_obtained.add(r)
		return total_rules_obtained

	elif template_number == 3:

		p1 = input('Enter the first parameter: ').upper()
		template_choices = re.split('(\d+)',p1)
		template_choices = list(filter(None , template_choices))
		if not template_choices[0].isdigit() or not template_choices[2].isdigit():
			print("First part is invalid")

		if template_choices[0] == '1':
			print("you have chosen template 1 for first part . please provide the relevant paramter details...")
			rule_set1 = get_template_number(1)
		elif template_choices[0] == '2':
			print("you have chosen get_template 2 for first part. please provide the relevant parameter details...")
			rule_set1 = get_template_number(2)

		if template_choices[2] == '1':
			print("you have chosen template 1 for second part . please provide the relevant paramter details...")
			rule_set2 = get_template_number(1)
		else:
			rule_set2 = get_template_number(2)

		if 'AND' in template_choices:
			return intersection_set(rule_set1 , rule_set2)

		return union_set(rule_set1 , rule_set2)








template_number = int(input('Enter the template number: '))
if template_number not in (1 , 2 , 3):
	print("Invalid template number")
	sys.exit("Invalid  template number")

template_rules = None
if template_number == 1:
	template_rules = get_template_number(template_number)
elif template_number == 2:
	template_rules = get_template_number(template_number)
else:
	template_rules = get_template_number(template_number)

print("the final set of template rules are:")
print(template_rules)
print("the count of the template rules are ..." , len(template_rules))









