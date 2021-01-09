import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd.DataFrame(marathon_2015_2017,columns=['M/F',  'Age',  'Pace',  '10K', '20K',  '30K',  'Official Time']).sort_values(by=['Official Time'])

record['M/F'] = record[____].map({'M': 1, ___: _})
# Dataframe to List
record_list = record.values.tolist()

grad_fig = Figure(figsize=(6,6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(0, 2000)
grad_ax.set_ylim(0, 10000)
grad_ax.set_title('Cost Gradient Decent')
grad_ax.set_ylabel("Total Cost")
grad_ax.set_xlabel("Number of Traning")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())
        
    # X and Y data from 0km to 30km
    
    x_train_1 = [ r[_] for r in record_list]
    x_train_2 = [ r[_] for r in record_list]
    x_train_3 = [ r[_] for r in record_list]

    y_train = [ r[__] for r in record_list]

    # Try to find values for W and b
    W1 = tf.Variable(tf.random_normal([1]), name="weight1")
    W2 = tf.Variable(tf.random_normal([1]), name="weight2")
    W3 = tf.Variable(tf.random_normal([1]), name="weight3")

    b = tf.Variable(tf.random_normal([1]), name="bias")
    
    # placeholders for a tensor
    X1 = tf.placeholder(tf.float32, shape=[____])
    X2 = tf.placeholder(tf.float32, shape=[____])
    X3 = tf.placeholder(tf.float32, shape=[____])

    Y = tf.placeholder(tf.float32, shape=[____])
    
    # Our hypothesis
    hypothesis = _____________________ + b
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(_________ - Y))
    
    # optimizer
    train = tf.train.GradientDescentOptimizer(__________=t_r).minimize(____)
    
    # Launch a session.
    with tf.______() as sess:
        # Initializes global variables
        sess.___(tf.______________________())
    
        # Fit the line
        log_ScrolledText.insert(END, "%10s %6i %20s %10.8f" % ('\nNo. of train is', (t_t-1), ', learing rate is ', t_r)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
        log_ScrolledText.insert(END, "%20s %20s" % ('Step', 'Cost')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val = sess.run([train, ____,  ________], feed_dict={X1: x_train_1, X2: x_train_2, X3: x_train_3, Y: y_train})
    
            if step % 100 == 0:
                print(step, cost_val, h_val) 
                g_xdata.append(step)
                g_ydata.append(cost_val)
                log_ScrolledText.insert(END, "%20i %20.5f" % (step, cost_val)+'\n')
        #gn.set_data(g_xdata, g_ydata)
        grad_ax.plot(g_xdata, g_ydata, 'ro')
        grad_ax.set_title('The minimum cost is '+str(cost_val)+' at '+str(step)+'times')
        grad_fig.canvas.draw() 
        # Testing our model
        winner = record_list[0]
        print(winner)
        time = sess.run(________, feed_dict={X1: [winner[_]], X2: [winner[_]], X3: [winner[_]]})
        # time = sess.run(hypothesis, feed_dict={X1: [1], X2: [25], X3: [296]})
        log_ScrolledText.insert(END, "%20s" % ('\n\nThe Winner Records Prediction\n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%20s %20s %20s" % ('Real record', 'ML Prediction', 'Variation(Second)')+'\n\n')
        log_ScrolledText.insert(END, "%20s %20s %20i" % (seconds_to_hhmmss(y_train[0]), seconds_to_hhmmss(time[_]), (y_train[_] - time[_]))+'\n')
        
    
#main
main = Tk()
main.title("Marathon Records")
main.geometry()

label=Label(main, text='Multi Variable Linear Regression Concept')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

t_tVal  = IntVar(value=2000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=1)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=0)

t_rVal  = DoubleVar(value=1e-6)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-6, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=3)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=2)

Button(main,text="Machine Learing", height=2,command=lambda:learing()).grid(row=2, column=0, columnspan=4, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=4)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=4, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


