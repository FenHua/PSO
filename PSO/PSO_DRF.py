#coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.11.17
功能：利用粒子群(PSO)算法优化腐蚀剂量响应函数中的7个参数，建立四个环境
    因素和腐蚀速率之间的非线性关系。7个参数和误差均有取值范围的限定。
"""
from __future__ import division
import numpy as np
import math
import copy as npcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from xlrd import open_workbook
from xlutils.copy import copy
import time


class MyPSO(object):
    """
    pop_size：鸟群规模
    factor_size：解的维度
    wmax, wmin ：惯性权值 w 的取值范围
    c1, c2：学习参数
    iter：最大迭代次数
    """
    def __init__(self,pop_size,factor_size,wmax,wmin,c1,c2,iter,data):
        self.pop_size = pop_size
        self.factor_size = factor_size
        self.wmax = wmax
        self.wmin = wmin
        self.c1 = c1
        self.c2 = c2
        self.iter = iter
        # 四个环境因子和腐蚀速率：T,RH,PD,SD,CORRO
        # 这里拟合class3等级的腐蚀剂量响应函数
        self.data =  np.array([[27.4,87,8,127,36],
                                [7.7,68,54,56,37],
                                [13.1,54,50,58,45]])

    # 初始化每只鸟的速度，每只鸟最好的速度，全局最好速度
    def Initpart(self):
        initvect, initvectdict, initbestpdict, initbestg, initerrodict = [], {}, {}, [], {}
        initvect = np.random.random_sample(self.factor_size)
        initbestg = npcopy.deepcopy(initvect)
        for i in xrange(self.pop_size):
            initvectdict[i] = initvect
            initbestpdict[i] = npcopy.deepcopy(initvect)
            initerrodict[i] = float('inf')
        return initvectdict,initbestpdict,initbestg,initerrodict

    # 定义待优化函数，计算误差
    def Func(self,factor):
        samplenum = np.shape(self.data)[0]
        erro = []
        for i in xrange(samplenum):
            if self.data[i][0] < 10:
                fsti = 0.15 * (self.data[i][0] - 10)
            else:
                fsti = -0.054 * (self.data[i][0] - 10)
            corrosiionRatei = factor[0] * self.data[i][2]**factor[1] * math.e**(factor[2] * self.data[i][1] + fsti) + \
                              factor[3] * self.data[i][3]**factor[4] * math.e**(factor[5] * self.data[i][1] + \
                              factor[-1]*self.data[i][0])
            erroi = abs(self.data[i][-1] - corrosiionRatei)
            erro.append(erroi)
        erroavg = sum(erro)/samplenum
        num = 0
        for i in range(samplenum):
            if erro[i] <= 10:
                num += 1
        if num >= samplenum:
            return erroavg,True,erro
        else:
            return  erroavg,False,erro

    # 更新每只鸟的速度方向，采用线性自适应惯性权重取值法
    def Vector(self,iternow,vectnow,bestp,bestg):
        r1, r2 = np.random.random(), np.random.random()
        vectnext = {}
        wnow = self.wmax - (self.wmax - self.wmin) / self.iter * iternow
        for i in xrange(self.pop_size):
            vectnext[i] = wnow * vectnow[i] + self.c1*r1*(bestp[i] - vectnow[i]) + self.c2*r2*(bestg - vectnow[i])
            # 对7个参数取值范围进行限定
            vectnext[i][0] = 1.3 if vectnext[i][0] < 0.5 else vectnext[i][0]
            vectnext[i][0] = vectnext[i][0]+1 if vectnext[i][0] < 1 else vectnext[i][0]
            vectnext[i][0] = 1.82 if vectnext[i][0] > 1.82 else vectnext[i][0]
            vectnext[i][1] = 0.2 if vectnext[i][1] < 0.1 else vectnext[i][1]
            vectnext[i][1] = vectnext[i][1]+0.3 if vectnext[i][1] < 0.2 else vectnext[i][1]
            vectnext[i][1] = 0.8 if vectnext[i][1] > 0.8 else vectnext[i][1]
            vectnext[i][2] = 0.01 if vectnext[i][2] < 0.01 else vectnext[i][2]
            vectnext[i][2] = 0.03 if vectnext[i][2] > 0.03 else vectnext[i][2]
            vectnext[i][3] = 0.05 if vectnext[i][3] < 0.05 else vectnext[i][3]
            vectnext[i][3] = 0.2 if vectnext[i][3] > 0.2 else vectnext[i][3]
            vectnext[i][4] = 0.53 if vectnext[i][4] < 0.53 else vectnext[i][4]
            vectnext[i][4] = 0.8 if vectnext[i][4] > 0.8 else vectnext[i][4]
            vectnext[i][5] = 0.01 if vectnext[i][5] < 0.01 else vectnext[i][5]
            vectnext[i][5] = 0.05 if vectnext[i][5] > 0.05 else vectnext[i][5]
            vectnext[i][6] = 0.02 if vectnext[i][6] < 0.02 else vectnext[i][6]
            vectnext[i][6] = 0.05 if vectnext[i][6] > 0.05 else vectnext[i][6]
        return vectnext

    # 绘制迭代误差图
    def Ploterro(self,errodict):
        mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        plt.plot(errodict.keys(),errodict.values(),'r-',linewidth=1.5,markersize=5)
        ax.set_xlabel(u'迭代次数',fontsize=18)
        ax.set_ylabel(u'误差',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0,)
        plt.ylim(0,20)
        # plt.title(u"迭代次数-误差",fontsize=18)
        plt.grid(True)
        plt.show()

    # 将合适的参数和误差保存至txt和excell
    def Save(self,bestg,erro):
        # 保存至txt
        factorandrerro = bestg.ravel().tolist()
        factorandrerro.extend(erro)
        file = open(r'PSO_DRF.txt','a') #追加模式
        for i in range(np.shape(factorandrerro)[0]):
            if i == 7:
                file.write("   ")
                file.write(str(round(factorandrerro[i],3)))
                file.write(" ")
            else:
                file.write(str(round(factorandrerro[i],3)))
                file.write(" ")
        file.write("\n")
        file.close()
        # 保存至excell，先将原文件复制再写入
        bookago = open_workbook('PSO_DRF.xls',formatting_info=True) #设置为 True 保留原格式
        sheetago = bookago.sheet_by_index(0)
        rowsago = sheetago.nrows    # 获取当前sheet有多少行，在下一行进行写入
        booknow = copy(bookago)
        sheetnow = booknow.get_sheet(0)
        for i in range(np.shape(factorandrerro)[0]):
            sheetnow.write(rowsago,i,str(round(factorandrerro[i],3)))
        booknow.save('PSO_DRF.xls')

    # 主程序入口
    def Run(self):
        erromin = float('inf')
        itererro,itererrodic = [],{}
        vectdict, bestp, bestg, errodict = self.Initpart()
        for iter in xrange(self.iter):
            print("iter = " + str(iter))
            for j in xrange(self.pop_size):
                erroj,trueorfalse,erro = self.Func(vectdict[j])
                # 得到每个鸟的最优速度
                if erroj < errodict[j]:
                    errodict[j] = erroj
                    bestp[j] = vectdict[j]
                else:
                    pass
                itererro.append(erroj)
            itererrodic[iter] = min(itererro)
            # 得到鸟群此次迭代及之前迭代历史最优速度
            for part, erro in errodict.iteritems():
                if erro < erromin:
                    erromin = erro
                    bestg = bestp[part]
                else:
                    pass
            erroavg,trueorfalse,erro = self.Func(bestg)
            if erromin <= 10 and trueorfalse == True and 1.3<bestg[0]<1.82 and \
                        0.2<bestg[1]<0.8 and 0.01<bestg[2]<0.03 and 0.05<bestg[3]<0.2 \
                    and  0.4<bestg[4]<0.8 and 0.01<bestg[5]<0.05 and 0.02<bestg[-1]<0.05:
                # 如果7个参数都在取值范围之内。且误差均小于10，则保存该参数和误差，并绘制每次迭代后的误差曲线
                print("Best erro = " + str(erromin))
                print("Best vector = " + str([round(a,4) for a in bestg.tolist()]))
                self.Save(bestg,erro)
                self.Ploterro(itererrodic)
                break
            else:
                vectdict = self.Vector(iter,vectdict,bestp,bestg)

if __name__ == "__main__":
    starttime = time.time()
    for i in xrange(5):
        a = MyPSO(100,7,0.9,0.4,2.05,2.05,300,data=None)
        a.Run()
    endtime = time.time()
    print("Runtime = " + str(endtime - starttime))
