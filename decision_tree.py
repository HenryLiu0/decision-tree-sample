from math import log
from collections import Counter
import copy
import csv
import carTreePlotter
import re

def calEntropy(dataSet):
    """
    输入：二维数据集
    输出：二维数据集标签的熵
    描述：
    计算数据集的标签的香农熵；香农熵越大，数据集越混乱；
    在计算 splitinfo 和通过计算熵减选择信息增益最大的属性时可以用到
    """
    entryNum = len(dataSet)
    labelsCount = {}
    for entry in dataSet:
        label = entry[-1]
        if label not in labelsCount.keys():
            labelsCount[label] = 0
        labelsCount[label] += 1             # labelsCount -> {'0' : 3, '1' : 4}
    entropy = 0.0
    for key in labelsCount:
        propotion = float(labelsCount[key])/entryNum   # propotion 特定标签占总标签比例
        entropy -= propotion * log(propotion, 2)
    return entropy

def calGini(dataSet):
    """
    输入：二维数据集
    输出：二维数据集的基尼系数
    描述：计算数据集的基尼系数，基尼系数越大数据集越混乱
    """
    entryNum = len(dataSet)
    labelsCount = {}
    for entry in dataSet:
        label = entry[-1]
        if label not in labelsCount.keys():
            labelsCount[label] = 0
        labelsCount[label] += 1
    gini = 1.0
    for key in labelsCount:
        p = float(labelsCount[key])/entryNum
        gini -= p * p    # 1-p1^2-p2^2
    return gini

def splitDataSet(dataSet, col, value):
    """
    输入：二维数据集，属性列index，值
    输出：从dataSet分离出来的subDataSet
    描述：
    将dataSet的col列中与value相同的样本组成一个新的subDataSet
    CART的分离方法与普通方法并无区别
    """
    subDataSet = []
    for entry in dataSet:
        if entry[col] == value:      # 将col属性中值为value的行挑出
            subEntry = entry[:col]
            subEntry.extend(entry[col+1:])
            subDataSet.append(subEntry)
    return subDataSet

def selectBestAttrIndex(dataSet, algorithm):
    """
    输入：二维数据集
    输出：熵减最大的属性在 dataSet 中的下标
    描述：
    先计算dataSet的熵，然后通过属性数目，遍历计算按照每个属性划分得到的熵；
    比较得到熵减最大的属性，返回它在dataSet中属性的index。
    """
    if algorithm == 'ID3':
        return selectBestAttrIndex_ID3(dataSet)
    elif algorithm == 'C4.5':
        return selectBestAttrIndex_C45(dataSet)
    elif algorithm == 'CART':
        return selectBestAttrIndex_CART(dataSet)

def selectBestAttrIndex_ID3(dataSet):
    labelNum = len(dataSet[0])-1            # 属性attribute数目
    oldEntropy = calEntropy(dataSet)
    bestIndex = -1
    maxInfoGain = 0.0
    for index in range(labelNum):
        newEntropy = 0.0
        attrValueList = [entry[index] for entry in dataSet]     # 获得dataSet中每个属性的所有value的列表
        attrValueSet = set(attrValueList)  # 获得value列表的不重复set，在ID3和C4.5中遍历计算每个value的熵，CART中用value进行二分类计算gini系数
        for uniqueValue in attrValueSet:
            subDataSet = splitDataSet(dataSet, index, uniqueValue)  # 分离出col=index, value = uniqueValue 的数据集
            p = float(len(subDataSet)) / len(dataSet)   # 计算子数据集占总数据比例
            newEntropy += p * calEntropy(subDataSet)
        infoGain = oldEntropy - newEntropy
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            bestIndex = index
    return bestIndex

def selectBestAttrIndex_C45(dataSet):
    labelNum = len(dataSet[0])-1
    oldEntropy = calEntropy(dataSet)
    bestIndex = -1
    maxInfoGainRotio = 0.0
    for index in range(labelNum):
        newEntropy = 0.0
        splitInfo = 0.0
        attrValueList = [entry[index] for entry in dataSet]
        attrValueSet = set(attrValueList)
        for uniqueValue in attrValueSet:
            subDataSet = splitDataSet(dataSet, index, uniqueValue)
            p = float(len(subDataSet)) / len(dataSet)
            newEntropy += p * calEntropy(subDataSet)
            splitInfo -= p * log(p, 2)       # index标签的熵
        infoGain = oldEntropy - newEntropy
        if splitInfo == 0.0:
            continue
        infoGainRatio = infoGain / splitInfo   # 计算信息增益
        if infoGainRatio > maxInfoGainRotio:
            maxInfoGainRotio = infoGainRatio
            bestIndex = index
    return bestIndex

def selectBestAttrIndex_CART(dataSet):
    labelNum = len(dataSet[0])-1
    bestIndex = -1
    minGini = float("inf")    # 所有attribute 中最小gini系数
    for index in range(labelNum):
        attrValueList = [entry[index] for entry in dataSet]
        attrValueSet = set(attrValueList)
        newGini = 0.0
        for uniqueValue in attrValueSet:
            subDataSet = splitDataSet(dataSet, index, uniqueValue)
            p = float(len(subDataSet)) / len(dataSet)
            newGini += p * calGini(subDataSet)
        if newGini < minGini:
            minGini = newGini
            bestIndex = index
    return bestIndex

def createTree(dataSet, oriAttr, oriAttrUniValSet, algorithm = 'ID3'):
    attr = oriAttr[:]     # 输入的一份拷贝，不改动输入的属性
    attrUniValSet = oriAttrUniValSet[:]
    labelList = [entry[-1] for entry in dataSet]
    if len(labelList) == labelList.count(labelList[0]):  # 1. 所有样本标签相同，那么该节点为记为该标签叶子节点
        return labelList[0]
    if len(attr) == 0:                 # 2. 没有可以分类的属性
        return Counter(labelList).most_common(1)[0][0]    # 返回出现次数最多的标签
    # dataSet 为空？dataSet 中所有属性的收益相同？
    bestAttrIndex = selectBestAttrIndex(dataSet, algorithm)        # 获得收益最大的属性下标，2. 数据集中所有样本在所有属性上增益相同
    bestAttr = attr[bestAttrIndex]                      # 获得收益最大属性
    resTree = {bestAttr : {}}                           # 构建字典树
    del(attr[bestAttrIndex])                            # 删除收益最大属性，与split后的dataSet相同长度
    valueSet = attrUniValSet[bestAttrIndex]      #B1
    del(attrUniValSet[bestAttrIndex])            #B1
    for value in valueSet:    # 为每个value创建分支
        subDataSet = splitDataSet(dataSet, bestAttrIndex, value)
        if len(subDataSet) == 0:    # 3. 数据集为空，预测标签为父节点出现最多的标签
            resTree[bestAttr][value] = Counter(labelList).most_common(1)[0][0]
        else:
            cpyAttr = attr[:]      # 创建attr的副本，避免直接传需要用到的引用进函数  #B1
            resTree[bestAttr][value] = createTree(subDataSet, cpyAttr, attrUniValSet, algorithm)    # 分支字典 {attribute0 : {low : {}, med : {}, high : {}, vhigh : {}}}   #B1 B2
    return resTree

def createAttrUniValSet(dataSet):
    attrUniValSet = []
    for attrIndex in range(len(dataSet[0])-1):   # 遍历每个属性
        attrList = [entry[attrIndex] for entry in dataSet]
        attrUniValSet.append(set(attrList))
    return attrUniValSet

def classifierVec(testVec, attr, tree):
    tempTree = copy.deepcopy(tree)       # 深复制
    while(isinstance(tempTree, dict)):
        nodeName = list(tempTree.keys())[0]    # 获得标签 outlook {'outlook':{}}
        nodeAttrIndex = attr.index(nodeName)   # 获得标签 outlook 在 attr 的下标 0
        branch = testVec[nodeAttrIndex]        # 获得分支值 2 ，用于{2:{windy:{}}}
        tempTree = tempTree[nodeName][branch]
    return tempTree

def classifierSet(testDataSet, attr, tree):
    resLabel = []
    for testVec in testDataSet:
        resLabel.append(classifierVec(testVec, attr, tree))
    return resLabel

def saveTree(path, tree):
    with open(path, 'w') as wf:
        wf.write(repr(tree))     # 将决策树字典结果当做字符串写入文件
    # print("Write done!\nThe file looks like:")
    # with open(path, 'r') as rf:
    #     sample = rf.read()
    #     print(sample)

def loadTree(path):
    with open(path, 'r') as rf:
        tree = eval(rf.read())
        return tree

def loadCarDataSet(path):
    with open(path, 'r') as csvfile:
        entries = csv.reader(csvfile)
        dataSet = list(entries)          # 获得数据集二维列表
        attr = ['attr' + str(i) for i in range(len(dataSet[0])-1)]    # 获得属性向量
        return dataSet, attr

def saveCarDataRes(path, carDataSetRes):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(carDataSetRes)

def calAccuracy(dataSet, resVec):
    if len(dataSet) != len(resVec):
        print("Length of dataSet no equal length of resVec!")
        return
    dataLabelVec = [entry[-1] for entry in dataSet]
    correctCount = 0
    for i in range(len(resVec)):
        if dataSet[i][-1] == resVec[i]:
            correctCount += 1
    accuracy = float(correctCount)/len(resVec)
    return accuracy


# main函数中的选择函数
def mainTrainTree():
    print("说明：训练集是train.csv，验证集是validate.csv，由Car_train.csv随机分配得到，比例为3:1")
    print("使用train.csv建立决策树")
    carDataSet, carAttr = loadCarDataSet('./data/train.csv')
    carUniValSet = createAttrUniValSet(carDataSet)

    print("正在训练ID3决策树...", end='')
    car_ID3_Tree = createTree(carDataSet, carAttr, carUniValSet)
    saveTree('./output/car_ID3_Tree/car_ID3_Tree.txt', car_ID3_Tree)
    print("完成，保存为'./output/car_ID3_Tree/car_ID3_Tree.txt'")
    print("正在绘制ID3决策树图像...", end='')
    carTreePlotter.createPlot(car_ID3_Tree, "./output/car_ID3_Tree/car_ID3_Tree.png")
    print("完成，保存为'./output/car_ID3_Tree/car_ID3_Tree.png'")

    print("正在训练C4.5决策树...", end='')
    car_C45_Tree = createTree(carDataSet, carAttr, carUniValSet, 'C4.5')
    saveTree('./output/car_C45_Tree/car_C45_Tree.txt', car_C45_Tree)
    print("完成，保存为'./output/car_ID3_Tree/car_C45_Tree.txt'")
    print("正在绘制C4.5决策树图像...", end='')
    carTreePlotter.createPlot(car_C45_Tree, "./output/car_C45_Tree/car_C45_Tree.png")
    print("完成，保存为'./output/car_ID3_Tree/car_C45_Tree.png'")

    print("正在训练CART决策树...", end='')
    car_CART_Tree = createTree(carDataSet, carAttr, carUniValSet, 'CART')
    saveTree('./output/car_CART_Tree/car_CART_Tree.txt', car_CART_Tree)
    print("完成，保存为'./output/car_ID3_Tree/car_CART_Tree.txt'")
    print("正在绘制CART决策树图像...", end='')
    carTreePlotter.createPlot(car_CART_Tree, "./output/car_CART_Tree/car_CART_Tree.png")
    print("完成，保存为'./output/car_CART_Tree/car_CART_Tree.png'")
    

def mainCalAccu():
    carTestSet, carTestAttr = loadCarDataSet('./data/validate.csv')
    print("正在用ID3决策树计算验证集...", end='')
    car_ID3_Tree = loadTree('./output/car_ID3_Tree/car_ID3_Tree.txt')
    car_ID3_SetRes = classifierSet(carTestSet, carTestAttr, car_ID3_Tree)
    car_ID3_accuracy = calAccuracy(carTestSet, car_ID3_SetRes)
    print("完成，准确率为 %f" % car_ID3_accuracy)

    print("正在用C4.5决策树计算验证集...", end='')
    car_C45_Tree = loadTree('./output/car_C45_Tree/car_C45_Tree.txt')
    car_C45_SetRes = classifierSet(carTestSet, carTestAttr, car_C45_Tree)
    car_C45_accuracy = calAccuracy(carTestSet, car_C45_SetRes)
    print("完成，准确率为 %f" % car_C45_accuracy)

    print("正在用CART决策树计算验证集...", end='')
    car_CART_Tree = loadTree("./output/car_CART_Tree/car_CART_Tree.txt")
    car_CART_SetRes = classifierSet(carTestSet, carTestAttr, car_CART_Tree)
    car_CART_accuracy = calAccuracy(carTestSet, car_CART_SetRes)
    print("完成，准确率为 %f" % car_CART_accuracy)
    

def mainGenRes():
    carDataSet, carAttr = loadCarDataSet('./data/Car_test.csv')
    print("正在用ID3决策树生成测试集预测结果...", end='')
    car_ID3_Tree = loadTree('./output/car_ID3_Tree/car_ID3_Tree.txt')
    car_ID3_SetRes = classifierSet(carDataSet, carAttr, car_ID3_Tree)
    saveCarDataRes('./output/car_ID3_Tree/ID3_predict.csv', car_ID3_SetRes)
    print("完成，保存为'./output/car_ID3_Tree/ID3_predict.csv'")

    print("正在用C4.5决策树生成测试集预测结果...", end='')
    car_C45_Tree = loadTree('./output/car_C45_Tree/car_C45_Tree.txt')
    car_C45_SetRes = classifierSet(carDataSet, carAttr, car_C45_Tree)
    saveCarDataRes('./output/car_C45_Tree/C45_predict.csv', car_C45_SetRes)
    print("完成，保存为'./output/car_C45_Tree/C45_predict.csv'")

    print("正在用CART决策树生成测试集预测结果...", end='')
    car_CART_Tree = loadTree('./output/car_CART_Tree/car_CART_Tree.txt')
    car_CART_SetRes = classifierSet(carDataSet, carAttr, car_CART_Tree)
    saveCarDataRes('./output/car_CART_Tree/CART_predict.csv', car_CART_SetRes)
    print("完成，保存为'./output/car_CART_Tree/CART_predict.csv'")

def main():
    trained = True
    while True:
        activeNumStr = input("1.训练决策树\t2.计算准确率\t3.生成测试集预测结果\t4.退出\n")
        if re.match(r'^[1-4]$', activeNumStr):
            activeNum = int(activeNumStr)
            if activeNum == 1:
                mainTrainTree()
                trained = True
            elif activeNum == 4:
                break
            else:
                if trained:
                    if activeNum == 2:
                        mainCalAccu()
                    elif activeNum == 3:
                        mainGenRes()
                else:
                    print("请先训练决策树")
        else:
            print("输入不匹配：", end='')


main()