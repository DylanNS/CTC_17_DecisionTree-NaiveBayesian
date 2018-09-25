"""
CES - 30 Lab 2 - Árvore de decisão e Redes baysianas

author: Dylan Nakandakari Sugimoto & Gabriel Adriano de Melo
Data: 24/09/2018
"""
from copy import copy
from math import log2
from graphviz import Digraph
from random import shuffle,seed

filename = "C:/Users/Dylan N. Sugimoto/Desktop/CTC_17_Lab2/dataset/connect_4/connect-4.data"
attribute_size = result_size = 3
validate_size = 10000


class Sample:

    def __init__(self,ID,list_values):
        self.id = ID
        list_values[-1] = list_values[-1].replace("\n","")
        self.values = list_values
        self.quant_attributes = self.calc_quant_attributes()

    def identify(self):
        return copy(self.id)
    def get_attributes(self):
        return copy(self.values[:-1])
    def get_result(self):
        return copy(self.values[-1])
    def print_values(self):
        print("values: ",self.values)
    def calc_quant_attributes(self):
        #last position is the experiment result. Not a attribute.
        return len(self.values) - 1
    def return_value(self,ID):
        return self.values[ID-1]

class Dataset:

    def __init__(self,filename,size_attributes = 3,result_size = 3):
        #attributes size is how much values it can assume.Like binary attribute size is 2.
        input_file = open(filename, 'r')
        self.dataset = self.build_dataset(input_file)
        self.filename = filename
        self.attributes_size = size_attributes
        self.result_size = result_size

    def shuffle(self,myseed):
        seed(myseed)
        shuffle(self.dataset)
    def build_dataset(self,file):

        dataset = []
        counter = 1
        for item in file:
            dataset.append(Sample(counter,item.split(",")))
            counter+=1
        return dataset
    def calc_quant_attributes(self):
        return self.dataset[1].quant_attributes
    
    def get_partition(self,p):
        
        if p ==0:
            return copy(self.dataset),[]
        subset_t = self.dataset[:-p]
        subset_v = self.dataset[-p:]
        return copy(subset_t),copy(subset_v)
    
    
    def get_dataset_size(self):
        return len(self.dataset)

class Attribute:

    def __init__(self,name,list_attribute_values):

        self.name = name
        self.list_attribute_values = list_attribute_values
        self.attribute_size = len(self.list_attribute_values)
        self.entropy_gain_v =[] 
        self.sample_set_attv = {}
        self.dic_popsup_ratio = {}
        self.dic_answer = {}
        self.build_dicanswer()
       
    
    def get_popsup_ratio(self,key):
        return copy(self.dic_popsup_ratio.get(key))
    
    def build_dicanswer(self):
        
        for v in self.list_attribute_values:
            self.dic_answer[v] = None
            self.sample_set_attv[v] = None
            self.dic_popsup_ratio[v] = None
       
            
    def get_attr_values(self):

        return copy(self.list_attribute_values)
    def calc_ratio_popsup(self,sample_set,value):
        
        total = len(sample_set)
        #Se o conjunto Sv for vazio
        if total==0:
            return 0
        n = 0
        for s in sample_set:
            #comparar valor do resultado
            if s.return_value(0) == value:
                n+=1
        return n/total
    
    def evaluate_entropy(self,ratio):
        #not zero
        if ratio:
            return -ratio*log2(ratio)
        #ratio zero -> entropy = 0
        return 0
    
    def calc_entropy(self,sample_set,list_values,key):
        #return entropy in bits
        list_popsup_ratio = [self.calc_ratio_popsup(sample_set,v) for v in list_values]
        self.dic_popsup_ratio[key] = list_popsup_ratio
        list_entropy = [self.evaluate_entropy(ratio) for ratio in list_popsup_ratio]
        return sum(list_entropy)

    def calc_sample_set_attv(self,sample_set,v):
        sample_set_attv = [s  for s in sample_set if s.return_value(self.name) == v]
        self.sample_set_attv[v] = copy(sample_set_attv)
        return sample_set_attv
    def calc_entropy_gain(self,sample_set, result_entropy_value,list_values):
        #calcula o ganho de entropia do atributo sobre o conjunto sample_set
        #sample_set: list of samples (its the domain)
        #result_entropy_values: sample_set's entropy or result's entropy on sample_set
        #list_values: list of values that result can assume.
        size_sample_set = len(sample_set)
        list_sample_set_v = [self.calc_sample_set_attv(sample_set,v) for v in self.list_attribute_values]
        list_entropy_gain_v = [-(len(sv)/size_sample_set)*self.calc_entropy(sv,list_values,v) for (sv,v) in zip(list_sample_set_v,self.list_attribute_values)]
        self.entropy_gain_v = list_entropy_gain_v
        
        return result_entropy_value + sum(list_entropy_gain_v)

class Decision_Tree:

    def __init__(self,trainning_dataset, list_attr):

        self.trainning_dataset = trainning_dataset
        #root tree
        self.root = None
        #list of all attributes and the last must be the result
        self.list_attr = list_attr
        #result attribute must be last in the attribute list (list_attr)
        self.result = self.list_attr[-1]

    def build_tree(self,list_attr2node_names,min_size_setv):

        #copy list attr
        list_attr = copy(self.list_attr[:-1])
        #get possible result values
        result_values = self.result.get_attr_values()
        #init list parent attr
        #calculate result entropy
        result_entropy = self.result.calc_entropy(self.trainning_dataset,result_values,"win")
        #calculate entropy gain for each attribute
        list_attribute_gain = [attr.calc_entropy_gain(self.trainning_dataset,result_entropy,result_values) for attr in list_attr]
        #get attribute with max entropy gain and remove from list_attr
        self.root = max_entropy_gain_attr = list_attr.pop(list_attribute_gain.index(max(list_attribute_gain)))
        list_parent_attr = []
        #init list Sample set
        list_sample_set = []
        list_parent_attr_value = []
        for value in max_entropy_gain_attr.get_attr_values():
                    popsup_ratio = max_entropy_gain_attr.get_popsup_ratio(value)
                    max_popsup_ratio = max(popsup_ratio)
                    if max_popsup_ratio > 0.99 or len(max_entropy_gain_attr.sample_set_attv.get(value)) < min_size_setv:
                        
                        index_max_popsup_ratio = popsup_ratio.index(max_popsup_ratio)
                        max_entropy_gain_attr.dic_answer[value] = result_values[index_max_popsup_ratio]
                    else:
                        list_parent_attr.append(max_entropy_gain_attr)
                        list_parent_attr_value.append(value)
                        list_sample_set.append(max_entropy_gain_attr.sample_set_attv.get(value))
        
        for sample_set,parent_attr_value,parent_attr in zip(list_sample_set,list_parent_attr_value,list_parent_attr):
            #print(len(sample_set),parent_attr_value,parent_attr.name)
            if len(list_attr) != 0:
                #calculate result entropy
                result_entropy = self.result.calc_entropy(sample_set,result_values,"win")
                #calculate entropy gain for each attribute
                list_attribute_gain = [attr.calc_entropy_gain(sample_set,result_entropy,result_values) for attr in list_attr]
                #get attribute with max entropy gain and remove from list_attr
                max_entropy_gain_attr = list_attr.pop(list_attribute_gain.index(max(list_attribute_gain)))
                #put child
                parent_attr.dic_answer[parent_attr_value] = max_entropy_gain_attr
               
                for value in max_entropy_gain_attr.get_attr_values():
                    #get value's popsup ratio list
                    popsup_ratio = max_entropy_gain_attr.get_popsup_ratio(value)
                    #get max popsup ratio
                    max_popsup_ratio = max(popsup_ratio)
                    #Verify if ratio is near to 1 or if Sv is small 
                    if max_popsup_ratio > 0.99 or len(max_entropy_gain_attr.sample_set_attv.get(value)) < min_size_setv:
                        #put leaf (answer)
                        index_max_popsup_ratio = popsup_ratio.index(max_popsup_ratio)
                        max_entropy_gain_attr.dic_answer[value] = result_values[index_max_popsup_ratio]
                    else:
                        #put intern node (question)
                        list_parent_attr.append(max_entropy_gain_attr)
                        list_parent_attr_value.append(value)
                        list_sample_set.append(max_entropy_gain_attr.sample_set_attv.get(value))
            else:
                #todos os atributos avaliados
                break
            
        #Fill the last tree level with leaf
        for attr in self.list_attr[:-1]:
            for key in attr.dic_answer:
                value = attr.dic_answer.get(key)
                if value is None:
                   
                    list_ratio = attr.dic_popsup_ratio.get(key)
                    i = list_ratio.index(max(list_ratio))
                    attr.dic_answer[key] = result_values[i]
        return
    
    def print_all_atributes_relations(self,filename, ext,list_attr2node_names):
        #Imprime todos os atributos
        f = Digraph("Decision Tree", filename)
        f.attr('node', shape='circle')
        f.node_attr.update(color='lightblue2', style='filled')
        for attr in self.list_attr[:-1]:
            for key in attr.dic_answer:
                value = attr.dic_answer.get(key)
                if isinstance(value,str):
                    f.edge(str(list_attr2node_names[attr.name]), value, label=key)
                else:
                     
                     f.edge(str(list_attr2node_names[attr.name]), str(list_attr2node_names[value.name]), label=key)
        f.format = ext
        f.render(filename,view = True)
    def print_tree(self,filename, ext,list_attr2node_names):
        #Imprime a arvore
        f = Digraph("Decision Tree", filename)
        f.attr('node', shape='circle')
        f.node_attr.update(color='lightblue2', style='filled')
        tree_question =[self.root]
        new_node = 0
        for attr in tree_question:
            for key in attr.dic_answer:
                value = attr.dic_answer.get(key)
                if isinstance(value,str):
                    name_new_node = str(new_node)
                    f.node(name_new_node,value)
                    f.edge(str(list_attr2node_names[attr.name]), name_new_node, label=key)
                    new_node+=1
                else:
                     
                     f.edge(str(list_attr2node_names[attr.name]), str(list_attr2node_names[value.name]), label=key)
                     tree_question.append(value)
        f.format = ext
        f.render(filename,view = True)
    def answer(self,sample):
        #first question is tree's root
        question = self.root
        
        while True:
            #ask to sample question and receive answer
            ans = sample.return_value(question.name)
            #check answer at question's dictionary answers
            next_step = question.dic_answer.get(ans)
            if isinstance(next_step,str):
                #next_step is string -> it's "win","loss" or "draw"
                return next_step
            else:
                #next_step is not string -> it's a question
                question = next_step
                

class Apriori:
    
    def __init__(self,trainning_dataset, result_attr):
        
        self.answer = True
        self.calc_answer(trainning_dataset,result_attr)
    def calc_answer(self,trainning_dataset,result_attr):
        
        score = {}
        for v in result_attr.list_attribute_values:
            score[v] = 0
        for s in trainning_dataset:
            score[s.get_result()]+=1
        self.answer = max(score,key=score.get)
    

dataset = Dataset(filename,attribute_size,result_size)
myseed=19
dataset.shuffle(myseed)
#possible values for attributes
list_attribute_values = ["x","b","o"]
list_attr = []
list_attr2node_names = {}
letter = 'a'
number = '1'
#fill attribute list and decoded attributes' names list
for index in range(dataset.calc_quant_attributes()):
    list_attr.append(Attribute(index+1,list_attribute_values))
    list_attr2node_names[index+1] = letter + number
    number = str(chr(ord(number)+1))
    if number > '6':
        number = '1'
        letter =str(chr(ord(letter)+1))

#possible values for result/feedback
list_result_values = ["win","loss","draw"]
#put feedback to the end of list_attr
list_attr.append(Attribute(len(list_attr)+1,list_result_values))   
#get trainning dataset and validate dataset     
trainning_data,validate_data = dataset.get_partition(validate_size)

#Instance Tree
dtree = Decision_Tree(trainning_data,list_attr)

#Build Tree
min_size_setv = 200
dtree.build_tree(list_attr2node_names,min_size_setv)
trainning_data_size = len(dataset.dataset)- validate_size

#Print Tree into pdf
filanem_out = "Decision_Tree" + "TamPoda" + str(min_size_setv) + "TamData" + str(trainning_data_size)
ext = 'pdf'
dtree.print_tree(filanem_out,ext,list_attr2node_names)


#validate Dtree
ncorrect_t = {key:0 for key in list_result_values}
matrixc = {key:copy(ncorrect_t) for key in list_result_values}
dic_validate = copy(ncorrect_t)
total = len(validate_data)
total_collum = copy(ncorrect_t)
for s in validate_data:

    ans = dtree.answer(s)
    validate = s.get_result()
    if ans == validate:
        ncorrect_t[validate]+=1
    dic = matrixc.get(validate)
    dic[ans] +=1
    dic_validate[validate] +=1
total_correct_t = sum(ncorrect_t.values())
print("Conjunto de Validacao")
for k in dic_validate:
    print(k,"\t",dic_validate.get(k))
print("Decision Tree Validate")
print("Acertos: ",ncorrect_t," (",total_correct_t /total,") ")
print("Total: ",total)

#Instance A priori
ap = Apriori(trainning_data,list_attr[-1])

#validate A priori classifier
ncorrect_ap = {key:0 for key in list_result_values}
ans = ap.answer
for s in validate_data:
    validate = s.get_result()
    if ans == validate:
        ncorrect_ap[validate] +=1
total_correct_ap = sum(ncorrect_ap.values())
print("A Priori")
print("Acertos: ",ncorrect_ap," (",total_correct_ap/ total,") ")
print("Total: ",total)
print("Is Decision Tree has higher score than A priori ? ",total_correct_t  > total_correct_ap)
print("Is Decision Tree has same score than A priori ? ",total_correct_t  == total_correct_ap)
print("Is Decision Tree has lower score than A priori ? ",total_correct_t  < total_correct_ap)

print("Matrix de Confusão")

string = ""
for key in matrixc:
    string += "\t"+str(key)
string +="\t Total"
print(string)

total_real = []
for key in matrixc:
    string  =str(key)
    count = 0
    for k in matrixc.get(key):
        string +="\t"+str(matrixc.get(key).get(k))
        count +=matrixc.get(key).get(k)
        total_collum[k] +=matrixc.get(key).get(k)
    total_real.append(count)
    string+="\t"+str(count)
    print(string)
string = "Total"
for key in total_collum:
    string += "\t"+str(total_collum.get(key))
print(string)

total = sum(total_collum.values())
ratio = [total_collum.get(k)/total for k in total_collum]
pe = sum([v*r for v,r in zip(total_real,ratio)]) 
po = total_correct_t
kappa = (po - pe)/(sum(total_real)-pe)
print("Kappa: ",kappa)