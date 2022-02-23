import numpy as np
import scipy as sp
import pandas as pd
import math
import cmath
import random
from scipy.io import loadmat
import time
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
import cufflinks as cf
cf.go_offline()


def plot_stepcharacteristics(ddf,yfinal, zeroline, tol52mult):
    obj = stepsresponseparameters(ddf,yfinal,zeroline)
    fig = go.Figure()
    fig1 = go.Line( x=ddf['t'].values.tolist(), y=ddf['y'].values.tolist(),name="Real Power")
    fig4 = go.Scatter(x=[obj.get_peak_val_t_os()[1]], y=[obj.get_peak_val_t_os()[0]], name="peak",\
                      mode="markers+text",text=["Peak"])
    tmp=  obj.get_tsettle(0.05*tol52mult)
    fig2 = go.Scatter(x=[tmp[0]], y=[tmp[1]], name = "tsettle with 5% tolerance",\
                      mode="markers+text",text=["ts<br>{:.1f}%".format(0.05*tol52mult*100)])
    tmp2 = obj.get_tsettle(0.02*tol52mult)
    fig3 = go.Scatter(x=[tmp2[0]], y=[tmp2[1]], name="tsettle with 2% tolerance",\
                     mode="markers+text",text=["ts<br>{:.1f}%".format(0.02*tol52mult*100)])
    fig5 = go.Scatter(x=[obj.get_under_val_t_us()[1]], y=[obj.get_under_val_t_us()[0]], \
                      name="under1", mode="markers+text",text=["Settling<br>min1"])

    fig.add_trace(fig1)
    fig.add_trace(fig2)
    fig.add_trace(fig3)
    fig.add_trace(fig4)
    fig.add_trace(fig5)
    
    if not(obj.identify_under_before_over()):  

        temp = obj.get_tr()
        if temp == None:
            print("cannot calculate risetime")
        else:
            fig7 = go.Scatter(x=[temp[1]], y=[temp[3]], name="risetime 10%",\
                          mode="markers+text",text=["tr<br>10%"])
            fig.add_trace(fig7)
            fig8 = go.Scatter(x=[temp[2]], y=[temp[4]], name="risetime 90%",\
                              mode="markers+text",text=["tr<br>90%"])
            fig9 = go.Line( x=[temp[1],temp[2]] , y=[0,0],name="RiseTime(tr)", mode="lines+markers")

            fig.add_trace(fig8)
            fig.add_trace(fig9)
        fig.update_layout(showlegend=False)
        fig.show()
    else:
        fig6 = go.Scatter(x=[obj.get_under_val_t_us_second()[1]], y=[obj.get_under_val_t_us_second()[0]], name="under2",\
                          mode="markers+text",text=["Settling<br>min"])        
        fig.add_trace(fig6)
        fig.update_layout(showlegend=False)
        fig.show()
        print("Cannot calculate rise time for this signal yet??")
        
        
def preprocessing(signalfile):
    cdf = pd.read_csv(signalfile, index_col=0)
    cdf.head()
    cdf['Frequency'].iplot(title="Frequency")
    cdf['Power Real'].iplot(title="Power Real")
    pattern = '%Y-%m-%d  %H:%M:%S.%f%z'
    date_time = '2019-12-19 17:27:08.833000+00:00'
    ts = [x.split("+")[0] for x in cdf['Timestamp'].values]
    def make_epoch(date_time):
        pattern = '%Y-%m-%d  %H:%M:%S.%f'
        pattern2 = '%Y-%m-%d  %H:%M:%S'
        try:
            epoch = int(time.mktime(time.strptime(date_time, pattern))) 
        except: 
            epoch = int(time.mktime(time.strptime(date_time, pattern2)))
        return epoch

    from datetime import datetime
    def makedatetime(x):
        try:
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    ts = [makedatetime(x) for x in ts]

    tss = []
    tss.append(0.0)
    for i in range(1,len(ts)):
        d2 = ts[i]
        d1 = ts[i-1]
        diff = timedelta.total_seconds(d2-d1)
        tss.append(diff)

    taxis = []
    for i in range(len(tss)):
        if i == 0:
            taxis.append(0.0)
        else:
            taxis.append(taxis[i-1]+tss[i])
    return ([cdf, taxis])
class stepsresponseparameters():

    def __init__(self, ddf, yfinal, zeroline=0.0):
        self.ddf = pd.DataFrame(ddf)
        self.yval =  ddf['y'].values
        self.tval = ddf['t'].values
        self.yfinal = yfinal
        self.peak = None
        self.peaktime = None
        self.overshoot = None
        self.under = None
        self.under2 = None
        self.undertime = None
        self.undershoot = None  
        self.zeroline = zeroline

    def plot_ts(self):
        
        self.ddf.set_index(['t']).iplot()
        
        
    
    def get_peak_val_t_os(self):
        self.peak = np.max(self.yval)
        self.peaktime = self.tval[self.yval.tolist().index(self.peak)]
        self.overshoot = self.peak - self.yfinal
        
        return (self.peak, self.peaktime, self.overshoot)

    def get_under_val_t_us(self):
        if self.identify_under_before_over():
            self.under = np.min(self.yval)
#             if self.under < self.zeroline
            self.undertime = self.tval[self.yval.tolist().index(self.under)]
            self.undershoot = self.yfinal - self.under
            return (self.under, self.undertime, self.undershoot)
        else:
            self.under, self.undertime, self.undershoot  = self.get_under_val_t_us_second()
            return (self.under, self.undertime, self.undershoot)
            

    def get_mc(self,l1, l2):
        # get slope and constant for y=m*x + c 
        m = float((l2[1] - l1[1])) / float(l2[0] - l1[0])
        c = (l2[1] - (m * l2[0]))
        return m, c

    def get_t9(self):
        if self.nineelem_val == self.ninef_def:    
            t9 = self.tval[self.nineelem]
            y9 = self.ninef_def
        if self.nineelem_val >self.ninef_def:    
            m,c = self.get_mc((self.tval[self.nineelem-1], self.ynew_bp[self.nineelem-1]), (self.tval[self.nineelem], self.nineelem_val))
            t9 = (self.ninef_def-c)/m  
            y9 = self.nineelem_val
        if self.nineelem_val < self.ninef_def: 
            m,c = self.get_mc((self.tval[self.nineelem], self.ynew_bp[self.nineelem]), (self.tval[self.nineelem+1], self.ynew_bp[self.nineelem+1]))
            t9 = (self.ninef_def-c)/m
            y9 = self.nineelem_val
            
        return([t9, y9])

    def get_t1(self):
        if self.oneelem_val == self.onef_def:    
            t1 = self.tval[self.oneelem]
            y1 = self.oneelem
        if self.oneelem_val > self.onef_def:    
            m,c = self.get_mc((self.tval[self.oneelem-1], self.ynew_bp[self.oneelem-1]), (self.tval[self.oneelem], self.oneelem_val))
            t1 = (self.onef_def-c)/m  
            y1 = self.onef_def
            if t1 < self.tval[self.oneelem-1]: 
                t1 = self.tval[self.oneelem]
                y1 = self.oneelem
        if self.oneelem_val < self.onef_def: 
            m,c = self.get_mc((self.tval[self.oneelem], self.ynew_bp[self.oneelem]), (self.tval[self.oneelem+1], self.ynew_bp[self.oneelem+1]))
            t1 = (self.onef_def-c)/m
            y1 = self.onef_def
            if t1 > self.tval[self.oneelem+1]:
                t1 = self.tval[self.oneelem+1 ]
                y1 = self.oneelem+1 
        return([t1, y1])

    def get_tr(self):
        if self.identify_under_before_over():
            print("Rise time calculations for such signals has not yet been implementd")
            return(None)
        else:
            self.onef_def = 0.1*self.yfinal
            self.ninef_def = 0.9*self.yfinal
            if self.onef_def < self.zeroline:
                print("Rise time calculations for such signals has not yet been implementd, t10% is less than zeroline")
                return(None)               
            if self.peak == None:
                self.get_peak_val_t_os()
            self.ynew_bp = self.yval[: self.yval.tolist().index(self.peak)]
            y2 = [x if x>self.yfinal*0.05 else 0.0 for x in self.ynew_bp ]
            self.oneelem = min(range(len(y2)), key=lambda i: abs(y2[i]-self.onef_def))
            self.nineelem = min(range(len(y2)), key=lambda i: abs(y2[i]-self.ninef_def))
            self.oneelem_val = self.yval[self.oneelem]
            self.nineelem_val = self.yval[self.nineelem]       
            t1,y1 = self.get_t1()
            t9, y9 = self.get_t9()
            tr = t9 - t1 
            return(tr, t1, t9, y1, y9)
        
    def get_tsettle(self,tol):
        if self.under == None:
            self.get_under_val_t_us()
        if self.identify_under_before_over():
            self.get_under_val_t_us_second()
            yval_au = self.yval[self.yval.tolist().index(self.under2) : ]
            ysettleu = self.yfinal + self.yfinal*tol
            ysettlel = self.yfinal - self.yfinal*tol 
            for i in range(len(yval_au)):
                tmp = yval_au[i:]
                found = [x for x in tmp if (x>=ysettlel and  x<=ysettleu)]
                if (len(found) == len(tmp)):
                    break
            ts = self.tval[i-1+self.yval.tolist().index(self.under2)]
            ys = self.yval[i-1+self.yval.tolist().index(self.under2)]
            
        else:
            yval_au = self.yval[self.yval.tolist().index(self.under) : ]
            ysettleu = self.yfinal + self.yfinal*tol
            ysettlel = self.yfinal - self.yfinal*tol 
            for i in range(len(yval_au)):
                tmp = yval_au[i:]
                found = [x for x in tmp if (x>=ysettlel and  x<=ysettleu)]
                if (len(found) == len(tmp)):
                    break
            ts = self.tval[i-1+self.yval.tolist().index(self.under)]
            ys = self.yval[i-1+self.yval.tolist().index(self.under)]


        return ([ts,ys])

    def identify_under_before_over(self):
        maxs = np.max(self.yval)
        mins = np.min(self.yval)
        
        if (self.yval.tolist().index(mins) < self.yval.tolist().index(maxs)): 
            if mins < self.zeroline:
                return(False)
            else:
                return(True)
        else:
            return(False)
        
    def get_under_val_t_us_second(self):
        new_list = self.yval[self.yval.tolist().index(self.peak):]
        self.under2 = np.min(new_list)
        self.undertime2 = self.tval[self.yval.tolist().index(self.under2)]
        self.undershoot2 = self.yfinal - self.under2
        return (self.under2, self.undertime2, self.undershoot2)